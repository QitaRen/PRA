import torch
import torch.nn as nn
import torch.nn.functional as F

class TactileDecoder(nn.Module):
    """
    [辅助模块] 隐式特征 -> 触觉图像 (12x40x3)
    """
    def __init__(self, latent_dim=128, output_dim=1440):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class PhysicallyRecurrentStudentPolicy(nn.Module):
    def __init__(
        self, 
        sequence_length=4,     
        visual_input_dim=3,    
        prop_input_dim=26,
        tactile_latent_dim=128, # 隐式物理记忆维度
        tactile_image_shape=(12, 40, 3), 
        embed_dim=256,
        transformer_layers=3,
        nhead=4,
        dropout=0.1
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.tactile_latent_dim = tactile_latent_dim
        self.tactile_flat_dim = tactile_image_shape[0] * tactile_image_shape[1] * tactile_image_shape[2]

        # ----------------------------------------------------------------
        # 1. Encoders (Frame-wise processing)
        # ----------------------------------------------------------------
        self.vis_conv1 = nn.Conv1d(visual_input_dim, 64, 1)
        self.vis_conv2 = nn.Conv1d(64, 128, 1)
        self.vis_conv3 = nn.Conv1d(128, embed_dim, 1)
        self.vis_bn1 = nn.BatchNorm1d(64)
        self.vis_bn2 = nn.BatchNorm1d(128)
        self.vis_bn3 = nn.BatchNorm1d(embed_dim)
        
        self.prop_mlp = nn.Sequential(
            nn.Linear(prop_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim), 
            nn.LayerNorm(embed_dim)
        )

        # ----------------------------------------------------------------
        # 2. PRA Modules (Recurrent Logic)
        # ----------------------------------------------------------------
        # [冷启动] 可学习初始状态，用于 t=0
        self.h_tact_init = nn.Parameter(torch.zeros(1, tactile_latent_dim))
        
        # 将 h_tact 映射为 Attention 的 Query
        self.query_proj = nn.Linear(tactile_latent_dim, embed_dim)

        # 单步 Cross-Attention: Query=Tactile, Key/Value=Visual
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # 触觉状态更新器 (从当前融合特征预测下一时刻的物理状态)
        # Fused_Feat(t) -> h_tact(t+1)
        self.tactile_rnn_cell = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, tactile_latent_dim),
            nn.LayerNorm(tactile_latent_dim)
        )

        # ----------------------------------------------------------------
        # 3. Temporal & Output
        # ----------------------------------------------------------------
        self.pos_emb = nn.Parameter(torch.zeros(1, sequence_length, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=512, 
            dropout=dropout, batch_first=True, norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # [主任务头] Latent Alignment (Student -> Teacher Z)
        self.latent_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 320) # 假设 Teacher 特征维度是 320 (根据 dataset target_z 调整)
        )

        # [辅助任务头] 触觉重建 (用于 Loss)
        self.tactile_decoder = TactileDecoder(latent_dim=tactile_latent_dim, output_dim=self.tactile_flat_dim)

    def forward_visual_frame(self, x):
        # x: (B, 3, N) -> PointNet layers
        x = F.relu(self.vis_bn1(self.vis_conv1(x)))
        x = F.relu(self.vis_bn2(self.vis_conv2(x)))
        x = self.vis_bn3(self.vis_conv3(x))
        x = torch.max(x, 2)[0] # Global Max Pool -> (B, embed_dim)
        return x

    def forward(self, pc_seq, prop_seq):
        """
        Args:
            pc_seq: (B, T, N, 3)
            prop_seq: (B, T, D_prop)
        Returns:
            pred_z: (B, 320) 蒸馏特征
            pred_tactile_img: (B, 12, 40, 3) 最后一帧的触觉重建
        """
        B, T, N, C = pc_seq.shape
        
        # 1. 预处理所有帧的视觉和本体特征 (Batch-Parallel)
        # Visual: (B, T, N, 3) -> (B*T, 3, N)
        pc_flat = pc_seq.view(B*T, N, C).permute(0, 2, 1)
        vis_feats_flat = self.forward_visual_frame(pc_flat) # (B*T, embed)
        vis_feats = vis_feats_flat.view(B, T, self.embed_dim)
        
        # Prop: (B, T, D) -> (B, T, embed)
        prop_feats = self.prop_mlp(prop_seq)

        # 2. 【核心】物理循环注意力回路 (Physically-Recurrent Loop)
        # 我们必须在时间维度上显式循环，因为 h_tact[t] 依赖于 h_tact[t-1]
        
        # 初始化 h_tact (使用冷启动参数)
        h_tact = self.h_tact_init.expand(B, -1) # (B, latent)
        
        fused_sequence = []
        
        for t in range(T):
            # A. 准备 Query (来自上一时刻的物理记忆)
            query = self.query_proj(h_tact).unsqueeze(1) # (B, 1, embed)
            
            # B. 准备 Key/Value (当前时刻的视觉)
            curr_vis = vis_feats[:, t, :].unsqueeze(1)   # (B, 1, embed)
            
            # C. Cross Attention: "凭借感觉去看"
            # visual 只有当前帧，所以不需要 mask
            attn_out, _ = self.cross_attn(query=query, key=curr_vis, value=curr_vis)
            
            # D. Residual Fusion (Base on Proprioception)
            curr_prop = prop_feats[:, t, :].unsqueeze(1) # (B, 1, embed)
            fused_step = self.attn_norm(attn_out + curr_prop) # (B, 1, embed)
            fused_sequence.append(fused_step)
            
            # E. 【闭环】更新物理状态 h_tact
            # 使用当前融合的特征，去预测/更新这一刻感受到的触觉状态
            # 这个 h_tact 将作为下一次循环的 Query
            h_tact = self.tactile_rnn_cell(fused_step.squeeze(1)) # (B, latent)

        # 3. 时序整合 (Transformer)
        # 堆叠序列: (B, T, embed)
        fused_seq = torch.cat(fused_sequence, dim=1)
        
        # 加位置编码
        fused_seq = fused_seq + self.pos_emb[:, :T, :]
        
        # Transformer 处理
        trans_out = self.transformer(fused_seq)
        
        # 取最后一帧作为全局上下文
        global_context = trans_out[:, -1, :] 

        # 4. 输出头
        # A. 主任务: 预测 Teacher 特征
        pred_z = self.latent_head(global_context)
        
        # B. 辅助任务: 重建最后一帧的触觉 (用于 Loss)
        # 注意：这里的 h_tact 已经是循环最后一步更新出来的，对应时刻 T 的触觉状态
        pred_tactile_flat = self.tactile_decoder(h_tact)
        pred_tactile_img = pred_tactile_flat.view(B, 12, 40, 3)

        return pred_z, pred_tactile_img