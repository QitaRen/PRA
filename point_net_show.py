import zarr
import numpy as np
import argparse
import os

def save_ply_xyz(points, filename):
    """
    将纯坐标点云保存为 PLY 文件 (不带颜色)
    points: (N, 3) numpy array
    """
    num_points = points.shape[0]
    
    with open(filename, 'w') as f:
        # 写入 PLY 头信息
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        # 写入点坐标
        for i in range(num_points):
            x, y, z = points[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
            
    print(f"✅ 已保存: {filename}")
    print(f"   点数: {num_points}")
    print(f"   范围: X[{points[:,0].min():.3f}, {points[:,0].max():.3f}] "
          f"Y[{points[:,1].min():.3f}, {points[:,1].max():.3f}] "
          f"Z[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")

def main():
    parser = argparse.ArgumentParser(description="提取Zarr中的原始点云帧为PLY文件")
    parser.add_argument("--zarr_path", type=str, default="data/bread/student_dataset.zarr", 
                        help="Zarr数据集路径")
    parser.add_argument("--index", type=int, default=0, 
                        help="要提取的帧索引 (Frame Index)")
    parser.add_argument("--output", type=str, default="raw_point_cloud.ply", 
                        help="输出文件名")
    
    args = parser.parse_args()

    # 1. 检查路径
    if not os.path.exists(args.zarr_path):
        print(f"❌ 错误: 找不到路径 {args.zarr_path}")
        return

    # 2. 加载 Zarr
    try:
        root = zarr.open(args.zarr_path, mode='r')
        # 根据你之前的截图，点云数据存储在 'point_cloud' 键下
        # 形状应该是 (Total_Frames, 1024, 3)
        pc_array = root['point_cloud']
    except Exception as e:
        print(f"❌ 打开 Zarr 失败: {e}")
        return

    total_frames = pc_array.shape[0]
    print(f"📂 数据集加载成功，总帧数: {total_frames}")

    # 3. 检查索引越界
    if args.index < 0 or args.index >= total_frames:
        print(f"❌ 错误: 索引 {args.index} 超出范围 (0 - {total_frames-1})")
        return

    # 4. 提取数据
    # 注意：Zarr 支持切片读取，不会把整个数据集加载到内存
    raw_points = pc_array[args.index] # 形状应该为 (1024, 3)

    # 5. 保存
    save_ply_xyz(raw_points, args.output)

if __name__ == "__main__":
    main()