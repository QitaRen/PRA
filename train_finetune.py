import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import swanlab
import numpy as np

# === 引入项目模块 ===
from src.model import ActionGuidedStudentPolicy
from src.diffusion_policy import DiffusionPolicy 
from src.dataset import StudentDataset

# ==============================================================================
# 1. 定义联合 Agent (Wrapper) - 修复维度匹配问题
# ==============================================================================
class StudentDiffusionAgent(nn.Module):
    def __init__(self, encoder, policy):
        super().__init__()
        self.encoder = encoder
        self.policy = policy
        
    def forward(self, batch):
        # 1. Student 编码 -> (B, 320)
        global_cond_student, _ = self.encoder(batch['pc'], batch['prop'])
        
        # 2. [关键修复] 维度适配
        # Teacher Config: n_obs_steps=5, obs_as_global_cond=True
        # Teacher Expects: (B, 320 * 5) = (B, 1600)
        # 我们把 Student 的单帧特征重复 5 次来"欺骗" Teacher
        global_cond_expanded = global_cond_student.repeat(1, 5) # (B, 1600)
        
        # 3. 计算 Loss
        loss_dict = self.policy.compute_loss(batch['action'], global_cond_expanded)
        return loss_dict

    @torch.no_grad()
    def predict_action(self, batch):
        global_cond_student, _ = self.encoder(batch['pc'], batch['prop'])
        global_cond_expanded = global_cond_student.repeat(1, 5)
        result = self.policy.predict_action(global_cond_expanded)
        return result

# ==============================================================================
# 2. 训练主程序
# ==============================================================================
def train_finetune(args):
    swanlab.init(
        project="End2End_Finetune",
        experiment_name=args.exp_name,
        config=vars(args),
        mode="local"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # A. Load Student (320 dim output)
    print(f"📦 Loading Student: {args.student_ckpt}")
    tactile_flat_dim = 12 * 40 * 3
    student_encoder = ActionGuidedStudentPolicy(
        sequence_length=args.seq_len,
        visual_input_dim=3,
        prop_input_dim=26,      
        tactile_output_dim=tactile_flat_dim,
        embed_dim=256,
        dropout=args.dropout
    ).to(device)
    student_ckpt = torch.load(args.student_ckpt, map_location=device)
    if 'state_dict' in student_ckpt:
        student_encoder.load_state_dict(student_ckpt['state_dict'])
    else:
        student_encoder.load_state_dict(student_ckpt)

    # B. Load Teacher (Use Config Parameters!)
    print(f"📦 Loading Teacher: {args.teacher_ckpt}")
    diffusion_policy = DiffusionPolicy(
        action_dim=13,
        obs_dim=320, # 基础 obs 维度，Policy 内部会 *5 变成 1600
        pred_horizon=args.pred_horizon, # 必须是 8
        num_inference_steps=10,
        down_dims=(512, 1024, 2048),    # Teacher Config
        diffusion_step_embed_dim=128,   # Teacher Config
    ).to(device)
    
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device)
    state_dict = teacher_ckpt['state_dict'] if 'state_dict' in teacher_ckpt else teacher_ckpt
    
    # 加载权重 (Strict=True 应该也能过了，如果不放心可 False)
    missing, unexpected = diffusion_policy.model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"⚠️ Missing keys: {len(missing)} (Check if critical)")
    else:
        print("✅ Teacher Weights Perfectly Matched!")

    # C. Optimizer
    agent = StudentDiffusionAgent(student_encoder, diffusion_policy).to(device)
    optimizer = optim.AdamW([
        {'params': agent.encoder.parameters(), 'lr': 1e-5},
        {'params': agent.policy.parameters(),  'lr': 1e-5}
    ], weight_decay=1e-6) # Config weight_decay is 1e-6
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # D. Dataset
    # [关键] pred_horizon 必须设为 8，与 Teacher 训练时一致
    print(f"📂 Loading Dataset (Horizon={args.pred_horizon})...")
    dataset = StudentDataset(
        args.data_path, 
        sequence_length=args.seq_len, 
        cache_in_ram=True, 
        is_train=True,
        return_future_actions=True,
        pred_horizon=args.pred_horizon 
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # Fit Normalizer
    print("📊 Fitting Action Normalizer...")
    agent.policy.fit_normalizer(train_loader)
    
    # [关键修复步骤]: Fit Normalizer 会导致参数掉回 CPU，必须重新移动到 GPU！
    agent.to(device)
    print("🔄 Re-moved Agent to Device (Fixing Normalizer device mismatch)")

    # E. Training Loop
    print(f"🔥 Start Training for {args.epochs} epochs...")
    global_step = 0
    for epoch in range(args.epochs):
        agent.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()
            
            loss_dict = agent(batch)
            loss = loss_dict['loss']
            
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            swanlab.log({"train/loss": loss.item(), "global_step": global_step})
            global_step += 1
            
        lr_scheduler.step()
        print(f"📉 Avg Loss: {epoch_loss / len(train_loader):.5f}")
        
        if (epoch + 1) % args.save_interval == 0:
            torch.save(agent.state_dict(), os.path.join(args.save_dir, f"finetune_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/bread/student_dataset.zarr")
    parser.add_argument("--student_ckpt", type=str, default="checkpoints_pro/student_best_pro.pth")
    # 这里指向您的 Teacher Checkpoint
    parser.add_argument("--teacher_ckpt", type=str, default="teacher_checkpoints/action_model_best.pth", help="Path to teacher's action_model_latest.pth")
    parser.add_argument("--save_dir", type=str, default="checkpoints_finetune")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128) 
    parser.add_argument("--seq_len", type=int, default=4)
    # [关键修改] 默认值改为 8
    parser.add_argument("--pred_horizon", type=int, default=8, help="Must match Teacher Config 'horizon'")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--exp_name", type=str, default="Student_End2End_v2")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train_finetune(args)