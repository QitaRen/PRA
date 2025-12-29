import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os
from tqdm import tqdm

# [修改点 1]: 引入新模型
from src.model import ActionGuidedStudentPolicy
from src.dataset import StudentDataset

def collect_features(model, dataset, device, max_samples=1000):
    teacher_feats = []
    student_feats = []
    indices = [] 
    
    print("Running inference on validation set...")
    
    idxs = np.arange(len(dataset))
    if len(idxs) > max_samples:
        np.random.shuffle(idxs)
        idxs = idxs[:max_samples]
    
    # 排序以便观察轨迹
    idxs.sort()
    
    with torch.no_grad():
        for i in tqdm(idxs):
            # [修改点 2]: 解包 4 个变量
            pc_seq, prop_seq, target_z, _ = dataset[i]
            
            pc_seq = pc_seq.unsqueeze(0).to(device)
            prop_seq = prop_seq.unsqueeze(0).to(device)
            
            # [修改点 3]: 模型返回双头，只取 latent
            pred_z, _ = model(pc_seq, prop_seq)
            
            teacher_feats.append(target_z.cpu().numpy())
            student_feats.append(pred_z[0].cpu().numpy())
            indices.append(i)
            
    return np.array(teacher_feats), np.array(student_feats), np.array(indices)

def plot_tsne(teacher_feats, student_feats, indices, save_path):
    print("Computing t-SNE (this may take a moment)...")
    
    combined_feats = np.concatenate([teacher_feats, student_feats], axis=0)
    n_samples = len(teacher_feats)
    
    # 建议: perplexity 可以稍微调小一点如果点数不多，或者调大一点看全局
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(combined_feats)
    
    t_embed = tsne_results[:n_samples]
    s_embed = tsne_results[n_samples:]
    
    plt.figure(figsize=(12, 6))
    
    # 图 1: 对齐情况
    plt.subplot(1, 2, 1)
    plt.title("Feature Alignment: Teacher vs Student")
    plt.scatter(t_embed[:, 0], t_embed[:, 1], c='red', label='Teacher (GT)', alpha=0.3, s=15)
    plt.scatter(s_embed[:, 0], s_embed[:, 1], c='blue', label='Student (Pred)', alpha=0.5, s=15, marker='x')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 图 2: 轨迹结构
    plt.subplot(1, 2, 2)
    plt.title("Manifold Structure (Colored by Time)")
    # 这里把两个 Embedding 拼起来画，看整体的时间流形
    all_embed = np.concatenate([t_embed, s_embed], axis=0)
    all_indices = np.concatenate([indices, indices])
    
    sc = plt.scatter(all_embed[:, 0], all_embed[:, 1], c=all_indices, cmap='viridis', s=10, alpha=0.6)
    plt.colorbar(sc, label='Frame Index')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved t-SNE plot to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    # 指向新的 checkpoint
    parser.add_argument("--checkpoint", type=str, default="checkpoints_pro/student_best_pro.pth")
    parser.add_argument("--data_path", type=str, default="data/bread/student_dataset.zarr")
    parser.add_argument("--samples", type=int, default=1000, help="Visualize max N samples")
    parser.add_argument("--seq_len", type=int, default=4)
    args = parser.parse_args()

    # 1. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # [修改点 4]: 参数适配 ActionGuidedStudentPolicy
    tactile_flat_dim = 12 * 40 * 3
    model = ActionGuidedStudentPolicy(
        sequence_length=args.seq_len,
        visual_input_dim=3,
        prop_input_dim=26,      
        tactile_output_dim=tactile_flat_dim,
        embed_dim=256,
        dropout=0.0 
    ).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 2. 加载验证集
    dataset = StudentDataset(args.data_path, sequence_length=args.seq_len, cache_in_ram=False, is_train=False)
    
    # 3. 收集特征
    t_feats, s_feats, idxs = collect_features(model, dataset, device, max_samples=args.samples)
    
    # 4. 绘图
    os.makedirs("vis_results", exist_ok=True)
    plot_tsne(t_feats, s_feats, idxs, "vis_results/tsne_analysis_pro.png")

if __name__ == "__main__":
    main()