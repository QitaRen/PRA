import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import swanlab

from src.model import ActionGuidedStudentPolicy
from src.dataset import StudentDataset

def train(args):
    # 1. 初始化 SwanLab
    swanlab.init(
        project="tactile_distillation_pro",
        experiment_name=args.exp_name,
        description="Action-Guided + Tactile Image Recon (12x40x3)",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "architecture": "ActionGuided_CrossAttn_ImageRecon"
        },
        mode="local"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 准备数据
    ds_train_source = StudentDataset(args.data_path, args.seq_len, cache_in_ram=True, is_train=True)
    ds_val_source = StudentDataset(args.data_path, args.seq_len, cache_in_ram=True, is_train=False)
    
    total_samples = len(ds_train_source)
    split_index = 1000 if total_samples > 1000 else int(0.9 * total_samples)

    train_dataset = Subset(ds_train_source, list(range(0, split_index)))
    val_dataset = Subset(ds_val_source, list(range(split_index, total_samples)))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 3. 初始化模型
    # 计算触觉扁平维度: 12 * 40 * 3 = 1440
    tactile_flat_dim = 12 * 40 * 3
    
    model = ActionGuidedStudentPolicy(
        sequence_length=args.seq_len,
        visual_input_dim=3,
        prop_input_dim=26,      
        tactile_output_dim=tactile_flat_dim, # 传入 1440
        embed_dim=256,
        dropout=args.dropout
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 4. 训练循环
    for epoch in range(args.epochs):
        model.train()
        loss_tracker = {'mse': 0, 'cos': 0, 'recon': 0, 'total': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for pc_seq, prop_seq, target_z, target_tactile in pbar:
            pc_seq, prop_seq, target_z, target_tactile = \
                pc_seq.to(device), prop_seq.to(device), target_z.to(device), target_tactile.to(device)
            
            optimizer.zero_grad()
            
            # Forward: pred_tactile 形状已经是 (B, 12, 40, 3)
            pred_z, pred_tactile = model(pc_seq, prop_seq)
            
            # Loss A: Distillation
            l_mse = F.mse_loss(pred_z, target_z)
            l_cos = 1.0 - F.cosine_similarity(pred_z, target_z, dim=-1).mean()
            loss_distill = l_mse + 0.1 * l_cos
            
            # Loss B: Tactile Image Reconstruction
            # PyTorch MSE Loss 自动处理 (B, 12, 40, 3) 这种高维张量，无需 flatten
            loss_recon = F.mse_loss(pred_tactile, target_tactile)
            
            total_loss = loss_distill + 0.2 * loss_recon
            
            total_loss.backward()
            optimizer.step()
            
            # Update Tracker
            loss_tracker['total'] += total_loss.item()
            loss_tracker['mse'] += l_mse.item()
            loss_tracker['recon'] += loss_recon.item()
            
            pbar.set_postfix({
                'L_all': f"{total_loss.item():.4f}", 
                'Rec': f"{loss_recon.item():.4f}"
            })
            
        # Log & Validate
        avg_train = loss_tracker['total'] / len(train_loader)
        
        # Validation
        model.eval()
        val_total = 0
        with torch.no_grad():
            for pc_seq, prop_seq, target_z, target_tactile in val_loader:
                pc_seq, prop_seq, target_z, target_tactile = \
                    pc_seq.to(device), prop_seq.to(device), target_z.to(device), target_tactile.to(device)
                
                pred_z, pred_tactile = model(pc_seq, prop_seq)
                
                l_dis = F.mse_loss(pred_z, target_z) + \
                        0.1 * (1.0 - F.cosine_similarity(pred_z, target_z, dim=-1).mean())
                l_rec = F.mse_loss(pred_tactile, target_tactile)
                
                val_total += (l_dis + 0.2 * l_rec).item()
        
        avg_val = val_total / len(val_loader)
        
        swanlab.log({
            "train/loss": avg_train,
            "train/recon_loss": loss_tracker['recon'] / len(train_loader),
            "val/loss": avg_val,
            "epoch": epoch + 1
        })
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(args.save_dir, "student_best_pro.pth"))
            print(f"Epoch {epoch+1}: New Best Val Loss: {avg_val:.5f}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "student_last_pro.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/bread/student_dataset.zarr")
    parser.add_argument("--save_dir", type=str, default="checkpoints_pro")
    parser.add_argument("--epochs", type=int, default=250) 
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1) 
    parser.add_argument("--exp_name", type=str, default="ActionGuided_Student_ImageRecon")
    
    args = parser.parse_args()
    train(args)