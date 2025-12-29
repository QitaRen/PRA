import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random
from src.model import ActionGuidedStudentPolicy
from src.dataset import StudentDataset

def visualize_reconstruction(model, dataset, device, num_samples=3):
    model.eval()
    
    # 随机挑选几个样本
    indices = random.sample(range(len(dataset)), num_samples)
    
    plt.figure(figsize=(12, 4 * num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 加载数据
            pc_seq, prop_seq, target_z, target_tactile = dataset[idx]
            
            # 增加 Batch 维度
            pc_seq = pc_seq.unsqueeze(0).to(device)
            prop_seq = prop_seq.unsqueeze(0).to(device)
            
            # 推理
            _, pred_tactile = model(pc_seq, prop_seq)
            
            # 转为 Numpy (B, 12, 40, 3) -> (12, 40, 3)
            gt_img = target_tactile.cpu().numpy()
            pred_img = pred_tactile[0].cpu().numpy()
            
            # --- 归一化处理以便绘图 ---
            # 触觉数据范围可能很大，为了可视化，我们将其归一化到 0-1
            # 也可以根据实际数据的物理意义调整，这里使用 min-max
            v_min = min(gt_img.min(), pred_img.min())
            v_max = max(gt_img.max(), pred_img.max())
            
            # 绘图：Ground Truth
            plt.subplot(num_samples, 2, 2*i + 1)
            # 假设是 RGB，如果数值超出 0-1/0-255，imshow 会自动截断或归一化
            # 这里我们手动归一化一下以防万一
            show_gt = (gt_img - v_min) / (v_max - v_min + 1e-6)
            plt.imshow(show_gt) 
            plt.title(f"Sample {idx}: Ground Truth Tactile")
            plt.axis('off')
            
            # 绘图：Prediction (Hallucination)
            plt.subplot(num_samples, 2, 2*i + 2)
            show_pred = (pred_img - v_min) / (v_max - v_min + 1e-6)
            plt.imshow(show_pred)
            plt.title(f"Sample {idx}: predicted (Hallucination)")
            plt.axis('off')
            
    plt.tight_layout()
    os.makedirs("vis_results", exist_ok=True)
    save_path = "vis_results/tactile_recon_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved visualization to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 注意：确保这里的路径和参数与 train.py 一致
    parser.add_argument("--checkpoint", type=str, default="checkpoints_pro/student_best_pro.pth")
    parser.add_argument("--data_path", type=str, default="data/bread/student_dataset.zarr")
    parser.add_argument("--seq_len", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载数据集
    dataset = StudentDataset(args.data_path, args.seq_len, cache_in_ram=False, is_train=False)
    
    # 2. 加载模型
    # 必须与 train.py 中的参数完全一致
    model = ActionGuidedStudentPolicy(
        sequence_length=args.seq_len,
        visual_input_dim=3,
        prop_input_dim=26,      
        tactile_output_dim=1440, # 12*40*3
        embed_dim=256,
        dropout=0.0
    ).to(device)
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    
    # 3. 运行可视化
    visualize_reconstruction(model, dataset, device)