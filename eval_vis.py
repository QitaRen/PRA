import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from src.model import ActionGuidedStudentPolicy
from src.diffusion_policy import DiffusionPolicy 
from src.dataset import StudentDataset

class StudentDiffusionAgent(torch.nn.Module):
    def __init__(self, encoder, policy):
        super().__init__()
        self.encoder = encoder
        self.policy = policy
    
    @torch.no_grad()
    def predict_action(self, batch):
        global_cond_student, _ = self.encoder(batch['pc'], batch['prop'])
        global_cond_expanded = global_cond_student.repeat(1, 5)
        result = self.policy.predict_action(global_cond_expanded)
        return result

def visualize_trajectory(pred_action_long, gt_action_long, save_path):
    """
    绘制 1000 帧完整任务流程对比图
    """
    horizon = min(pred_action_long.shape[0], gt_action_long.shape[0])
    dim = pred_action_long.shape[1]
    
    # 截断
    pred_action = pred_action_long[:horizon]
    gt_action = gt_action_long[:horizon]
    
    # 布局：3列，自动计算行数
    n_cols = 3
    n_rows = int(np.ceil(dim / n_cols))
    
    # 画布高度适当增加，因为横轴现在很长
    plt.figure(figsize=(24, 3.5 * n_rows))
    
    for i in range(dim):
        plt.subplot(n_rows, n_cols, i+1)
        
        # === [修改点] 绘图样式优化 ===
        # 1000帧不适合用点(marker)，改用纯线
        # GT: 红色，稍微透明
        plt.plot(np.arange(horizon), gt_action[:, i], 'r-', label='Ground Truth', linewidth=1.5, alpha=0.6)
        
        # Pred: 蓝色，实线，强调
        plt.plot(np.arange(horizon), pred_action[:, i], 'b-', label='Student (RHC)', linewidth=1.2, alpha=0.9)
        
        # 误差计算
        mae = np.mean(np.abs(pred_action[:, i] - gt_action[:, i]))
        
        # 填充误差
        plt.fill_between(range(horizon), gt_action[:, i], pred_action[:, i], color='gray', alpha=0.2)
        
        # 标题
        plt.title(f"Joint {i+1} | Seq MAE: {mae:.4f}", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlabel("Time Step (Frame)", fontsize=10)
        plt.xlim(0, horizon)
        
        # === 智能纵轴逻辑 ===
        data_min = min(gt_action[:, i].min(), pred_action[:, i].min())
        data_max = max(gt_action[:, i].max(), pred_action[:, i].max())
        data_span = data_max - data_min
        
        NOISE_FLOOR = 0.02
        mid = (data_max + data_min) / 2
        
        if data_span < NOISE_FLOOR:
            plt.ylim(mid - 0.05, mid + 0.05)
        else:
            padding = data_span * 0.1
            plt.ylim(data_min - padding, data_max + padding)

        if i == 0:
            plt.legend(loc='upper right')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {save_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    print("Loading models...")
    student_encoder = ActionGuidedStudentPolicy(
        sequence_length=args.seq_len,
        visual_input_dim=3,
        prop_input_dim=26,      
        tactile_output_dim=12*40*3,
        embed_dim=256,
        dropout=0.0
    ).to(device)
    
    diffusion_policy = DiffusionPolicy(
        action_dim=13,
        obs_dim=320,
        pred_horizon=8,  
        num_inference_steps=10,
        down_dims=(512, 1024, 2048),
        diffusion_step_embed_dim=128
    ).to(device)
    
    agent = StudentDiffusionAgent(student_encoder, diffusion_policy).to(device)
    
    # 2. 加载权重
    print(f"Loading checkpoint: {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    
    # 3. 加载数据集
    dataset = StudentDataset(
        args.data_path, 
        sequence_length=args.seq_len, 
        cache_in_ram=True, 
        is_train=False, 
        return_future_actions=True, 
        pred_horizon=8
    )
    
    # 拟合 Normalizer
    from torch.utils.data import DataLoader
    # 稍微加大 batch_size 加速拟合
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    print("Fitting normalizer...")
    agent.policy.fit_normalizer(loader)
    agent.to(device)
    
    # ==========================================
    # 4. 1000帧滚动预测配置
    # ==========================================
    VIS_HORIZON = 1000   # 查看 1000 帧 (约 30-50秒)
    EXEC_STEPS = 4       # 每次执行 4 帧
    
    # 检查数据集是否够长
    if len(dataset) < VIS_HORIZON + 10:
        print(f"⚠️ Warning: Dataset size ({len(dataset)}) is close to or smaller than VIS_HORIZON ({VIS_HORIZON}). Reducing horizon.")
        VIS_HORIZON = len(dataset) - 50
    
    os.makedirs("vis_eval_1000frame", exist_ok=True)
    
    # 随机选起始点
    # valid_indices 必须保证后面有 1000 帧
    # 这里我们尝试找几个间隔较远的起始点，避免都在同一个 episode
    max_start_idx = len(dataset) - VIS_HORIZON - 1
    if max_start_idx <= 0:
        raise ValueError("Dataset is too short for 1000 frames visualization!")

    # 随机采样 3 个片段
    indices = np.random.choice(range(max_start_idx), 3, replace=False)
    
    print(f"Running Long-Horizon Inference ({VIS_HORIZON} frames)... This may take a minute.")
    
    for i, start_idx in enumerate(indices):
        print(f"Processing sample {i+1}/3 (Start Index: {start_idx})...")
        pred_actions_buffer = []
        
        # RHC 循环 (0 -> 1000)
        for t in range(0, VIS_HORIZON, EXEC_STEPS):
            current_dataset_idx = start_idx + t
            
            data = dataset[current_dataset_idx]
            batch = {
                'pc': data['pc'].unsqueeze(0).to(device),
                'prop': data['prop'].unsqueeze(0).to(device),
                'action': data['action'].unsqueeze(0).to(device)
            }
            
            result = agent.predict_action(batch)
            pred_8_frames = result['action_pred'][0].cpu().numpy()
            
            # 截取需要的帧数
            steps_to_take = min(EXEC_STEPS, VIS_HORIZON - t)
            pred_chunk = pred_8_frames[:steps_to_take]
            pred_actions_buffer.append(pred_chunk)
            
        pred_action_long = np.concatenate(pred_actions_buffer, axis=0)
        
        # 获取对应长度的 GT
        history_start_frame = dataset.valid_indices[start_idx]
        pred_start_frame = history_start_frame + dataset.seq_len 
        
        act_idx_start = pred_start_frame - 1
        act_idx_end = act_idx_start + VIS_HORIZON
        
        # 确保 GT 不越界
        if act_idx_end > len(dataset.actions):
            act_idx_end = len(dataset.actions)
            pred_action_long = pred_action_long[:(act_idx_end - act_idx_start)]
            
        gt_action_long = dataset.actions[act_idx_start : act_idx_end].cpu().numpy()

        # 绘图
        visualize_trajectory(pred_action_long, gt_action_long, f"vis_eval_1000frame/sample_{i}_1000frames.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/bread/student_dataset.zarr")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints_finetune/finetune_epoch_50.pth")
    parser.add_argument("--seq_len", type=int, default=4)
    args = parser.parse_args()
    
    main(args)