import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import argparse

# 引入你的模型和数据定义
from src.model import DualStreamStudentPolicy
from src.dataset import StudentDataset

class PointCloudGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # 注册 Hook
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] is the gradient wrt output of the layer
        self.gradients = grad_output[0]

    def __call__(self, pc_seq, prop_seq):
        self.model.zero_grad()
        
        # Forward pass
        # pc_seq: [1, T, 3, 1024]
        output = self.model(pc_seq, prop_seq)
        
        # --- 关键点：我们只对“隐式触觉特征”感兴趣 ---
        # Teacher 输出的前256维是视触觉融合特征，后64维是本体。
        # 我们最大化前 256 维的平均激活值，看模型是为了谁在努力。
        target = output[0, :256].mean() 
        
        # Backward pass
        target.backward()
        
        # Grad-CAM Calculation
        # Gradients: [Batch*T, 256, 1024] (因为模型内部把 B和T 合并处理了)
        # Activations: [Batch*T, 256, 1024]
        
        # 我们只看序列的最后一帧 (Current Frame)
        # model.point_encoder 处理后形状是 [B*T, 256, 1024]
        # 取最后一个 sample
        grads = self.gradients[-1]      # [256, 1024]
        acts = self.activations[-1]     # [256, 1024]
        
        # 1. Global Average Pooling on gradients (获取通道权重)
        weights = torch.mean(grads, dim=1, keepdim=True) # [256, 1]
        
        # 2. 加权激活图
        cam = torch.sum(weights * acts, dim=0) # [1024]
        
        # 3. ReLU (只关注正向贡献)
        cam = torch.relu(cam)
        
        # 4. 归一化到 0-1
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.cpu().detach().numpy()

def save_colored_ply(points, heatmap, filename):
    """
    将带有热力图颜色的点云保存为 PLY 文件
    points: (1024, 3)
    heatmap: (1024,) 0~1 value
    """
    # 使用 matplotlib 的 'jet' 或 'viridis' 颜色映射
    colormap = cm.get_cmap('jet')
    colors = colormap(heatmap)[:, :3] # 取 RGB, 舍弃 Alpha
    
    # 写入 PLY Header
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(points.shape[0]):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(int)
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
    
    print(f"Saved heatmap to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/student_best.pth")
    parser.add_argument("--data_path", type=str, default="data/bread/student_dataset.zarr")
    # 想要可视化的样本索引 (建议选几个接触发生时的索引)
    parser.add_argument("--sample_idx", type=int, default=100) 
    args = parser.parse_args()

    # 1. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamStudentPolicy(sequence_length=4).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    # --- Hook 目标层 ---
    # model.point_encoder 是一个 Sequential
    # 结构: [Conv, BN, ReLU, ..., Conv(最后), BN, ReLU(target), MaxPool]
    # 我们想看 MaxPool 之前的那个 ReLU 的输出
    # 在你的代码里，point_encoder 最后一层是 AdaptiveMaxPool1d
    # 倒数第二层是 ReLU
    target_layer = model.point_encoder[-2] 
    
    grad_cam = PointCloudGradCAM(model, target_layer)

    # 2. 加载数据 (取一条)
    dataset = StudentDataset(args.data_path, sequence_length=4, cache_in_ram=False, is_train=False)
    pc_seq, prop_seq, _ = dataset[args.sample_idx]
    
    # 增加 Batch 维度: [1, T, 3, 1024]
    pc_seq = pc_seq.unsqueeze(0).to(device) 
    # 注意：Dataset里如果是 (T, 1024, 3) 还是 (T, 3, 1024)？
    # 看 dataset.py 这里的 dataset 返回的是 Tensor
    # 如果你的 dataset 返回的是 (T, 1024, 3)，需要在 model forward 里处理，
    # 但 model forward 第一步做了 transpose。
    # 我们这里保持和 train loop 一致即可。
    
    prop_seq = prop_seq.unsqueeze(0).to(device)

    # 3. 运行 Grad-CAM
    heatmap = grad_cam(pc_seq, prop_seq)
    
    # 4. 保存可视化结果
    # 取出最后一帧的点云坐标用于画图
    # pc_seq shape [1, 4, 1024, 3]
    last_frame_points = pc_seq[0, -1, :, :].cpu().numpy() # (1024, 3)
    
    os.makedirs("vis_results", exist_ok=True)
    save_colored_ply(last_frame_points, heatmap, f"vis_results/cam_sample_{args.sample_idx}.ply")

if __name__ == "__main__":
    main()