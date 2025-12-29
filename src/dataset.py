import torch
import zarr
import numpy as np
from torch.utils.data import Dataset

class StudentDataset(Dataset):
    def __init__(self, zarr_path, sequence_length=4, cache_in_ram=True, is_train=True, 
                 return_future_actions=False, pred_horizon=16):
        """
        Args:
            return_future_actions (bool): 是否返回未来动作序列（用于 Diffusion 训练）
            pred_horizon (int): 预测未来的步数 (通常为 16)
        """
        self.seq_len = sequence_length
        self.is_train = is_train
        self.return_future_actions = return_future_actions
        self.pred_horizon = pred_horizon
        
        # 打开 Zarr
        print(f"Loading dataset from {zarr_path}...")
        self.data_root = zarr.open(zarr_path, mode='r')
        total_frames = self.data_root['point_cloud'].shape[0]
        
        # 计算有效索引：确保有足够的历史帧
        # 注意：如果需要预测未来，理论上还需要确保未来有数据，
        # 但通常我们允许读到最后，然后用 Padding 处理未来不足的情况
        self.valid_indices = list(range(total_frames - sequence_length + 1))
        
        # --- 数据加载 ---
        if cache_in_ram:
            print("Caching data in RAM...")
            self.point_clouds = torch.from_numpy(self.data_root['point_cloud'][:]).float()
            self.agent_pos = torch.from_numpy(self.data_root['agent_pos'][:]).float()   
            self.targets = torch.from_numpy(self.data_root['fused_feature'][:]).float() 
            self.tactile = torch.from_numpy(self.data_root['tactile'][:]).float()
            
            # [新增] 加载 Action 数据
            if 'action' in self.data_root:
                self.actions = torch.from_numpy(self.data_root['action'][:]).float()
            else:
                if self.return_future_actions:
                    raise ValueError("Dataset missing 'action' key, but return_future_actions=True!")
                self.actions = None
        else:
            self.point_clouds = self.data_root['point_cloud']
            self.agent_pos = self.data_root['agent_pos']
            self.targets = self.data_root['fused_feature']
            self.tactile = self.data_root['tactile']
            self.actions = self.data_root['action'] if 'action' in self.data_root else None
            
        self.in_ram = cache_in_ram
        print(f"Dataset loaded. Total frames: {total_frames}, Valid samples: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_frame = self.valid_indices[idx]
        end_frame = start_frame + self.seq_len
        
        # ==========================
        # 1. 获取历史输入 (PC & Prop)
        # ==========================
        if self.in_ram:
            pc_seq = self.point_clouds[start_frame : end_frame].clone()
            q_seq = self.agent_pos[start_frame : end_frame].clone()
            
            # 构造速度/动作意图
            if start_frame == 0:
                first_vel = torch.zeros_like(self.agent_pos[0:1])
                future_vel = self.agent_pos[start_frame+1 : end_frame] - self.agent_pos[start_frame : end_frame-1]
                prev_action_seq = torch.cat([first_vel, future_vel], dim=0)
            else:
                prev_action_seq = self.agent_pos[start_frame : end_frame] - self.agent_pos[start_frame-1 : end_frame-1]

            target_z = self.targets[end_frame - 1].clone()
            target_tactile = self.tactile[end_frame - 1].clone()

        else:
            # Numpy 模式
            pc_seq = torch.from_numpy(self.point_clouds[start_frame : end_frame]).float()
            q_seq_np = self.agent_pos[start_frame : end_frame]
            
            if start_frame == 0:
                first_vel = np.zeros_like(self.agent_pos[0:1])
                future_vel = self.agent_pos[start_frame+1 : end_frame] - self.agent_pos[start_frame : end_frame-1]
                prev_action_np = np.concatenate([first_vel, future_vel], axis=0)
            else:
                prev_action_np = self.agent_pos[start_frame : end_frame] - self.agent_pos[start_frame-1 : end_frame-1]
            
            q_seq = torch.from_numpy(q_seq_np).float()
            prev_action_seq = torch.from_numpy(prev_action_np).float()
            target_z = torch.from_numpy(self.targets[end_frame - 1]).float()
            target_tactile = torch.from_numpy(self.tactile[end_frame - 1]).float()

        # 构造 Proprioception 输入
        prop_seq = torch.cat([q_seq, prev_action_seq], dim=-1)

        # 数据增强 (仅针对点云)
        if self.is_train and not self.return_future_actions: 
            # 微调时通常减少增强，或者保持一致，这里保留逻辑
            noise = torch.randn_like(pc_seq) * 0.002 
            pc_seq += noise

        # ==========================
        # 2. 分支返回
        # ==========================
        
        # A. 如果是微调模式 (Fine-tuning / Diffusion Training)
        if self.return_future_actions:
            # 索引计算: 从当前帧(end_frame)开始往后取 pred_horizon 帧
            current_t = end_frame # 实际上是 end_frame - 1 对应当前时刻，但预测通常从下一刻或当前刻开始，看DP3定义。
            # 通常：Observations是 [t-T+1, ... t]，Action预测是 [t, ... t+H-1] 或 [t+1, ... t+H]
            # 这里假设预测从当前帧动作开始
            
            # 由于 self.valid_indices 保证了历史帧存在，但没保证未来帧存在，需要处理 Padding
            total_len = len(self.actions) if self.in_ram else self.data_root['action'].shape[0]
            
            act_start = end_frame - 1 # 对齐：最后历史帧的时刻
            act_end = act_start + self.pred_horizon
            
            if act_end > total_len:
                # 需要 Padding
                valid_len = total_len - act_start
                if self.in_ram:
                    valid_acts = self.actions[act_start:].clone()
                else:
                    valid_acts = torch.from_numpy(self.actions[act_start:]).float()
                
                # 重复最后一帧进行填充
                if valid_len > 0:
                    pad_len = self.pred_horizon - valid_len
                    last_act = valid_acts[-1].unsqueeze(0)
                    action_seq = torch.cat([valid_acts, last_act.repeat(pad_len, 1)], dim=0)
                else:
                    # 极端的边缘情况，虽然 valid_indices 应该避免
                    action_seq = torch.zeros((self.pred_horizon, self.actions.shape[1]))
            else:
                if self.in_ram:
                    action_seq = self.actions[act_start : act_end].clone()
                else:
                    action_seq = torch.from_numpy(self.actions[act_start : act_end]).float()
            
            # 返回字典格式，方便 Agent 处理
            return {
                'pc': pc_seq, 
                'prop': prop_seq, 
                'action': action_seq  # [Pred_Horizon, Action_Dim]
            }

        # B. 如果是蒸馏模式 (Distillation)
        else:
            return pc_seq, prop_seq, target_z, target_tactile