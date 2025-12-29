import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
import argparse
import os
from tqdm import tqdm

# [修改点 1]: 引入新的模型类
from src.model import ActionGuidedStudentPolicy
from src.dataset import StudentDataset

def collect_features(model, dataset, device, max_samples=1000):
    """
    收集 Teacher (Target) 和 Student (Pred) 的特征向量
    适配 Action-Guided 模型
    """
    teacher_feats = []
    student_feats = []
    indices = [] # 记录时间步
    
    print(f"Collecting {max_samples} samples from validation set...")
    
    # 随机采样
    idxs = np.arange(len(dataset))
    if len(idxs) > max_samples:
        np.random.shuffle(idxs)
        idxs = idxs[:max_samples]
    
    # 按索引排序，这样画出来的轨迹在时间上是连续的
    idxs.sort()
    
    with torch.no_grad():
        for i in tqdm(idxs):
            # [修改点 2]: 数据集现在返回 4 个值，我们只需要前3个
            # pc, prop, teacher_z, tactile_gt
            pc_seq, prop_seq, target_z, _ = dataset[i]
            
            # 增加 Batch 维度
            pc_seq = pc_seq.unsqueeze(0).to(device)
            prop_seq = prop_seq.unsqueeze(0).to(device)
            target_z = target_z.to(device)
            
            # [修改点 3]: 模型现在返回 (latent, tactile_img)，我们只取 latent
            pred_z, _ = model(pc_seq, prop_seq)
            
            # 存入列表
            teacher_feats.append(target_z.cpu().numpy())
            student_feats.append(pred_z[0].cpu().numpy()) # pred_z 是 (1, 320)
            indices.append(i)
            
    return np.array(teacher_feats), np.array(student_feats), np.array(indices)

def plot_umap_analysis(teacher_feats, student_feats, indices, save_path):
    """
    运行 UMAP 并绘图 (这部分逻辑通用，无需大改)
    """
    print("Running UMAP dimensionality reduction...")
    
    # 1. 合并数据 (Target + Pred)
    n_samples = len(teacher_feats)
    combined_feats = np.concatenate([teacher_feats, student_feats], axis=0)
    
    # 2. UMAP 参数设置
    reducer = umap.UMAP(
        n_neighbors=30, 
        min_dist=0.1, 
        n_components=2, 
        metric='cosine', # 继续使用 Cosine 距离，因为我们用了 Cosine Loss
        random_state=42
    )
    
    embedding = reducer.fit_transform(combined_feats)
    
    # 分离结果
    t_embed = embedding[:n_samples]
    s_embed = embedding[n_samples:]
    
    # --- 开始绘图 ---
    plt.figure(figsize=(14, 6))
    
    # 图 1: 对齐情况 (Alignment)
    plt.subplot(1, 2, 1)
    plt.title(f"UMAP Alignment: Teacher vs Student\n(n_samples={n_samples})")
    # Teacher: 红色
    plt.scatter(t_embed[:, 0], t_embed[:, 1], c='red', label='Teacher (GT)', alpha=0.2, s=20)
    # Student: 蓝色
    plt.scatter(s_embed[:, 0], s_embed[:, 1], c='blue', label='Student (Pred)', alpha=0.5, s=5)
    plt.legend()
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.grid(True, alpha=0.3)
    
    # 图 2: 轨迹流形 (Manifold Structure)
    plt.subplot(1, 2, 2)
    plt.title("Manifold Dynamics (Colored by Time)")
    
    # 拼接坐标和时间索引
    all_x = np.concatenate([t_embed[:, 0], s_embed[:, 0]])
    all_y = np.concatenate([t_embed[:, 1], s_embed[:, 1]])
    all_c = np.concatenate([indices, indices]) 
    
    sc = plt.scatter(all_x, all_y, c=all_c, cmap='plasma', s=10, alpha=0.5)
    plt.colorbar(sc, label='Time Step Index')
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved UMAP plot to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    # 注意这里指向 checkpoints_pro 文件夹
    parser.add_argument("--checkpoint", type=str, default="checkpoints_pro/student_best_pro.pth")
    parser.add_argument("--data_path", type=str, default="data/bread/student_dataset.zarr")
    parser.add_argument("--samples", type=int, default=2000, help="Number of samples")
    parser.add_argument("--seq_len", type=int, default=4, help="Sequence length used in training")
    args = parser.parse_args()

    # 1. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # [修改点 4]: 初始化 ActionGuidedStudentPolicy
    # 参数必须与 train.py 完全一致
    tactile_flat_dim = 12 * 40 * 3
    model = ActionGuidedStudentPolicy(
        sequence_length=args.seq_len,
        visual_input_dim=3,
        prop_input_dim=26,      # 13(Joints) + 13(Prev_Action)
        tactile_output_dim=tactile_flat_dim,
        embed_dim=256,
        dropout=0.0 # 推理时 Dropout 设为 0
    ).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 加载数据
    dataset = StudentDataset(args.data_path, sequence_length=args.seq_len, cache_in_ram=False, is_train=False)
    
    # 3. 收集特征
    t_feats, s_feats, idxs = collect_features(model, dataset, device, max_samples=args.samples)
    
    # 4. UMAP 分析
    os.makedirs("vis_results", exist_ok=True)
    plot_umap_analysis(t_feats, s_feats, idxs, "vis_results/umap_analysis_pro.png")

if __name__ == "__main__":
    main()