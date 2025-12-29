import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionGuidedStudentPolicy(nn.Module):
    def __init__(
        self, 
        sequence_length=4,     
        visual_input_dim=3,    
        prop_input_dim=26,     # 13(Joints) + 13(Prev_Action)
        
        # [修正]: 触觉是一张图 [12, 40, 3] -> Flatten = 1440
        tactile_output_dim=1440, 
        
        embed_dim=256,
        transformer_layers=3,
        nhead=4,
        dropout=0.1
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
        # ----------------------------------------------------------------
        # 1. Encoders 
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
        # 2. Innovation 1: Cross-Attention
        # ----------------------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # ----------------------------------------------------------------
        # 3. Temporal Reasoning
        # ----------------------------------------------------------------
        self.pos_emb = nn.Parameter(torch.zeros(1, sequence_length, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=512, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # ----------------------------------------------------------------
        # 4. Innovation 2: Dual Heads Output
        # ----------------------------------------------------------------
        
        # [Head A]: Latent Alignment (主任务)
        self.latent_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 320)
        )

        # [Head B]: Tactile Reconstruction (辅助任务) - 生成图像
        # 输入: embed_dim -> 输出: 1440 (12*40*3)
        self.tactile_recon_head = nn.Sequential(
            nn.Linear(embed_dim, 512), # 增加一层隐层以增强图像生成能力
            nn.ReLU(),
            nn.Linear(512, tactile_output_dim) 
        )

    def forward_visual_stream(self, pc_seq):
        """
        处理点云序列
        Input: (B, T, 1024, 3) -> 这里的 1024 是点数 N，3 是通道 C
        """
        # 1. 解包维度
        # 注意：这里我们用 generic 的写法，防止 shape 顺序不同
        # 你的数据形状是 (Batch, Time, Points, Channels)
        B, T, N, C = pc_seq.shape 
        
        # 2. 合并 Batch 和 Time 维度
        x = pc_seq.view(B * T, N, C) # -> (B*T, 1024, 3)
        
        # 3. [关键修正] 交换维度 (Transpose/Permute)
        # PointNet 需要 (Batch, Channels, Points)
        # 即把最后一维(3) 换到中间去
        x = x.permute(0, 2, 1)       # -> (B*T, 3, 1024)
        
        # 4. 进入卷积层 (现在 x 的 shape 符合 [B*T, 3, 1024] 了)
        x = F.relu(self.vis_bn1(self.vis_conv1(x)))
        x = F.relu(self.vis_bn2(self.vis_conv2(x)))
        x = self.vis_bn3(self.vis_conv3(x))
        
        # 5. Global Max Pooling
        # max over the points dimension (last dimension)
        x = torch.max(x, 2, keepdim=False)[0]  # (B*T, embed_dim)
        
        # 6. 恢复时序维度
        x = x.view(B, T, self.embed_dim)
        return x

    def forward(self, pc_seq, prop_seq):
        # 1. Encode
        vis_feat = self.forward_visual_stream(pc_seq) 
        prop_feat = self.prop_mlp(prop_seq)           

        # 2. Cross-Attention
        attn_out, _ = self.cross_attn(query=prop_feat, key=vis_feat, value=vis_feat)
        fused_feat = self.attn_norm(attn_out + prop_feat)

        # 3. Transformer
        fused_feat = fused_feat + self.pos_emb[:, :fused_feat.shape[1], :]
        trans_out = self.transformer(fused_feat)
        context = trans_out[:, -1, :] 

        # 4. Heads
        latent_pred = self.latent_head(context)
        
        # 触觉图像生成
        tactile_flat = self.tactile_recon_head(context) # (B, 1440)
        
        # [关键]: Reshape 为图像形状 (B, 12, 40, 3)
        tactile_pred = tactile_flat.view(-1, 12, 40, 3)

        return latent_pred, tactile_pred