import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import swanlab

from src.model import PhysicallyRecurrentStudentPolicy
from src.dataset import StudentDataset

def train(args):
    # 1. 初始化
    swanlab.init(
        project="pra_distillation",
        experiment_name=args.exp_name,
        description="PRA Stage 1: Representation Learning (No Action Head)",
        config=vars(args),
        mode="local"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 数据准备 (Dataset 不变)
    # dataset[i] 返回: (pc, prop, target_z, target_tactile)
    # target_tactile 是最后一帧 (frame T)
    ds_train_source = StudentDataset(args.data_path, args.seq_len, cache_in_ram=True, is_train=True)
    ds_val_source = StudentDataset(args.data_path, args.seq_len, cache_in_ram=True, is_train=False)
    
    total_samples = len(ds_train_source)
    # 简单划分 Train/Val
    split_index = int(0.9 * total_samples)
    train_dataset = Subset(ds_train_source, list(range(0, split_index)))
    val_dataset = Subset(ds_val_source, list(range(split_index, total_samples)))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. 模型初始化 (不包含 Action Head)
    model = PhysicallyRecurrentStudentPolicy(
        sequence_length=args.seq_len,
        visual_input_dim=3,
        prop_input_dim=26,
        tactile_latent_dim=128, 
        embed_dim=256,
        dropout=args.dropout
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Start PRA Distillation Training...")
    
    # 4. 训练循环
    for epoch in range(args.epochs):
        model.train()
        loss_tracker = {'total': 0, 'distill': 0, 'recon': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            # Dataset 返回 4 项
            pc_seq, prop_seq, target_z, target_tactile = batch
            
            pc_seq = pc_seq.to(device)
            prop_seq = prop_seq.to(device)
            target_z = target_z.to(device)
            target_tactile = target_tactile.to(device) # 这是最后一帧的 GT
            
            optimizer.zero_grad()
            
            # Forward
            # 模型内部会自动进行循环，最终输出 (B, 320) 和 (B, 12, 40, 3)
            pred_z, pred_tactile_img = model(pc_seq, prop_seq)
            
            # --- Loss Calculation ---
            
            # A. Distillation Loss (Z) - 核心任务
            l_mse = F.mse_loss(pred_z, target_z)
            # 添加 Cosine Similarity Loss 帮助方向对齐
            l_cos = 1.0 - F.cosine_similarity(pred_z, target_z, dim=-1).mean()
            l_distill = l_mse + 0.1 * l_cos
            
            # B. Tactile Reconstruction Loss (Aux) - 辅助 PRA 闭环
            # 监督最后一帧的触觉预测
            l_recon = F.mse_loss(pred_tactile_img, target_tactile)
            
            # Total Loss
            total_loss = l_distill + 0.2 * l_recon
            
            total_loss.backward()
            optimizer.step()
            
            # Logging
            loss_tracker['total'] += total_loss.item()
            loss_tracker['distill'] += l_distill.item()
            loss_tracker['recon'] += l_recon.item()
            
            pbar.set_postfix({
                'Dist': f"{l_distill.item():.4f}", 
                'Rec': f"{l_recon.item():.4f}"
            })

        # --- Validation ---
        model.eval()
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                pc_seq, prop_seq, target_z, target_tactile = batch
                pc_seq = pc_seq.to(device)
                prop_seq = prop_seq.to(device)
                target_z = target_z.to(device)
                target_tactile = target_tactile.to(device)
                
                pred_z, pred_tactile_img = model(pc_seq, prop_seq)
                
                l_dis = F.mse_loss(pred_z, target_z) + 0.1 * (1.0 - F.cosine_similarity(pred_z, target_z, dim=-1).mean())
                l_rec = F.mse_loss(pred_tactile_img, target_tactile)
                
                val_total += (l_dis + 0.2 * l_rec).item()
        
        avg_val = val_total / len(val_loader)
        avg_train_loss = loss_tracker['total'] / len(train_loader)
        
        swanlab.log({
            "train/loss": avg_train_loss,
            "train/distill_loss": loss_tracker['distill'] / len(train_loader),
            "train/recon_loss": loss_tracker['recon'] / len(train_loader),
            "val/loss": avg_val,
            "epoch": epoch + 1
        })
        
        # Save Best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(args.save_dir, "student_best_pra.pth"))
            print(f"Epoch {epoch+1}: New Best Val Loss: {avg_val:.5f}")

    # Save Last
    torch.save(model.state_dict(), os.path.join(args.save_dir, "student_last_pra.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/bread/student_dataset.zarr")
    parser.add_argument("--save_dir", type=str, default="checkpoints_pra")
    parser.add_argument("--epochs", type=int, default=250) 
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1) 
    parser.add_argument("--exp_name", type=str, default="PRA_Distillation_NoAction")
    
    args = parser.parse_args()
    train(args)