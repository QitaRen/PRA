import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import seaborn as sns 

# [修改点 1]: 引入新模型
from src.model import ActionGuidedStudentPolicy
from src.dataset import StudentDataset

def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)

def linear_hsic(K, L):
    K_c = centering(K)
    L_c = centering(L)
    return torch.sum(K_c * L_c)

def calculate_cka_score(X, Y):
    gram_x = torch.matmul(X, X.T)
    gram_y = torch.matmul(Y, Y.T)

    print("Computing HSIC (this may take a moment)...")
    hsic_xy = linear_hsic(gram_x, gram_y)
    hsic_xx = linear_hsic(gram_x, gram_x)
    hsic_yy = linear_hsic(gram_y, gram_y)

    cka_score = hsic_xy / (torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy))
    return cka_score.item(), gram_x, gram_y

def collect_features(model, dataset, device, max_samples=1000):
    teacher_feats = []
    student_feats = []
    
    print(f"Collecting {max_samples} samples from validation set...")
    
    idxs = np.arange(len(dataset))
    if len(idxs) > max_samples:
        np.random.shuffle(idxs)
        idxs = idxs[:max_samples]
    
    idxs.sort() 
    
    with torch.no_grad():
        for i in tqdm(idxs):
            # [修改点 2]: 解包 4 个变量
            # pc, prop, target_z, tactile_gt
            pc_seq, prop_seq, target_z, _ = dataset[i]
            
            pc_seq = pc_seq.unsqueeze(0).to(device)
            prop_seq = prop_seq.unsqueeze(0).to(device)
            target_z = target_z.to(device)
            
            # [修改点 3]: 模型返回双头，只取 latent
            pred_z, _ = model(pc_seq, prop_seq)
            
            teacher_feats.append(target_z)
            student_feats.append(pred_z[0])
            
    return torch.stack(teacher_feats), torch.stack(student_feats)

def plot_gram_matrices(gram_x, gram_y, cka_score, save_path):
    g_x = gram_x.cpu().numpy()
    g_y = gram_y.cpu().numpy()
    
    g_x = (g_x - g_x.min()) / (g_x.max() - g_x.min())
    g_y = (g_y - g_y.min()) / (g_y.max() - g_y.min())

    plt.figure(figsize=(14, 6))
    plt.suptitle(f"CKA Similarity Analysis (Score: {cka_score:.4f})", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.title("Teacher Representation Structure (Gram Matrix)")
    sns.heatmap(g_x, cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
    plt.xlabel("Sample Index (Time ->)")
    plt.ylabel("Sample Index (Time ->)")

    plt.subplot(1, 2, 2)
    plt.title("Student Representation Structure (Gram Matrix)")
    sns.heatmap(g_y, cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
    plt.xlabel("Sample Index (Time ->)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved CKA visualization to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    # 指向新的 checkpoint 目录
    parser.add_argument("--checkpoint", type=str, default="checkpoints_pro/student_best_pro.pth")
    parser.add_argument("--data_path", type=str, default="data/bread/student_dataset.zarr")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples for CKA calculation")
    parser.add_argument("--seq_len", type=int, default=4)
    args = parser.parse_args()

    # 1. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # [修改点 4]: 参数适配 ActionGuidedStudentPolicy
    tactile_flat_dim = 12 * 40 * 3
    model = ActionGuidedStudentPolicy(
        sequence_length=args.seq_len,
        visual_input_dim=3,
        prop_input_dim=26,      # 13+13
        tactile_output_dim=tactile_flat_dim,
        embed_dim=256,
        dropout=0.0
    ).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 2. 加载数据
    dataset = StudentDataset(args.data_path, sequence_length=args.seq_len, cache_in_ram=False, is_train=False)
    
    # 3. 收集特征
    teacher_feats, student_feats = collect_features(model, dataset, device, max_samples=args.samples)
    
    # 4. 计算 CKA
    cka_score, gram_x, gram_y = calculate_cka_score(teacher_feats, student_feats)
    
    print("-" * 30)
    print(f"🎯 Linear CKA Score: {cka_score:.5f}")
    print("-" * 30)
    
    if cka_score > 0.9:
        print("✅ Excellent Alignment! Student perfectly captures Teacher's structure.")
    elif cka_score > 0.7:
        print("👌 Good Alignment. Major structures are preserved.")
    else:
        print("⚠️ Poor Alignment. Representation structures differ significantly.")

    # 5. 绘图
    os.makedirs("vis_results", exist_ok=True)
    plot_gram_matrices(gram_x, gram_y, cka_score, "vis_results/cka_analysis_pro.png")

if __name__ == "__main__":
    main()