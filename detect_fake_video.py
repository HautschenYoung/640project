import os
import cv2
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functools import partial
from pathlib import Path
from glob import glob
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "dust3r"))

# ================= 关键修补：解决 PyTorch 2.6+ Pickle Error =================
# 在导入 dust3r 之前，我们给 torch.load 打个补丁，强制允许加载非 Tensor 对象。
# 这样无需修改 dust3r 库源码即可加载旧版 checkpoint。
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # 如果调用中没有指定 weights_only，且当前 torch 版本支持该参数，强制设为 False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# 应用补丁
torch.load = patched_torch_load
print("[System] 已应用 PyTorch weights_only=False 补丁以兼容 DUSt3R 模型。")

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# ================= 配置区域 =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 请修改为你下载的模型路径
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "dust3r" / "checkpoints" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

if not LOCAL_MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {LOCAL_MODEL_PATH}\nPut the checkpoint at this path or update MODEL_PATH.")

IMG_SIZE = 512  # 图像处理分辨率
INPUT_ROOT_DIR = PROJECT_ROOT / "preprocessed_frames"  # 对应 Step 1 的输出目录
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.float32): return float(obj)
        return super().default(obj)

def analyze_folder(folder_path, model):
    print(f"\n正在分析文件夹: {folder_path} ...")
    
    # 1. 加载图片
    image_files = sorted(glob(os.path.join(folder_path, '*.jpg')) + glob(os.path.join(folder_path, '*.png')))
    if len(image_files) < 2:
        print("图片数量不足，跳过")
        return None

    train_imgs = load_images(image_files, size=IMG_SIZE)
    
    # 2. 推理
    pairs = make_pairs(train_imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, DEVICE, batch_size=1)
    
    # 3. 全局对齐
    scene = global_aligner(output, device=DEVICE)
    scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
    
    # ================= 修复开始：使用标准 API 获取数据 =================
    
    # --- 修复 1: 轨迹获取 (Physics) ---
    # 不要遍历 scene.imgs，直接调用 scene.get_im_poses()
    # 它返回的是一个 Tensor shape [N, 4, 4]，代表 World-to-Camera 矩阵
    w2c_matrices = scene.get_im_poses().detach().cpu().numpy()
    
    traj_points = []
    for w2c in w2c_matrices:
        # DUSt3R 输出的是 World-to-Camera，我们需要求逆得到 Camera-to-World (相机在世界中的位置)
        c2w = np.linalg.inv(w2c)
        traj_points.append(c2w[:3, 3]) # 取平移向量 (x, y, z)
    
    traj_points = np.array(traj_points)
    
    # 计算抖动分数
    if len(traj_points) < 3:
        jitter_score = 0.0 # 帧数太少无法计算二阶差分
    else:
        velocities = np.diff(traj_points, axis=0)
        accelerations = np.diff(velocities, axis=0)
        jitter_score = np.mean(np.linalg.norm(accelerations, axis=1))
    
    # --- 修复 2: 深度/置信度获取 (Geometry) ---
    # 不要访问 img.conf，直接访问 scene.im_conf
    # scene.im_conf 是一个包含所有帧置信度图的 ParameterList
    confs = []
    for conf_map in scene.im_conf:
        # conf_map 是 (H, W) 或 (1, H, W) 的 Tensor
        # 需要 exp 因为内部存储的通常是 log-confidence，或者直接取值均值作为相对指标
        # 这里为了稳定，直接取 mean
        confs.append(conf_map.detach().cpu().numpy().mean())
    
    # 计算深度不一致性 (1 - 平均置信度)
    depth_inconsistency = 1.0 - np.mean(confs)
    
    # ================= 修复结束 =================
    
    return {
        "name": os.path.basename(folder_path),
        "jitter_score": float(jitter_score),
        "depth_error": float(depth_inconsistency),
        "trajectory": traj_points
    }

def visualize_comparison(results):
    """将所有视频的分析结果画在一张图上对比"""
    if not results: return
    
    fig = plt.figure(figsize=(15, 6))
    
    # 子图1: 3D 轨迹
    ax1 = fig.add_subplot(121, projection='3d')
    for res in results:
        traj = np.array(res['trajectory'])
        traj = traj - traj.mean(axis=0) # 中心化，方便对比形状
        ax1.plot(traj[:,0], traj[:,1], traj[:,2], marker='o', label=res['name'])
    ax1.set_title("Normalized Camera Trajectories")
    ax1.legend()
    
    # 子图2: 分数对比
    ax2 = fig.add_subplot(122)
    names = [r['name'] for r in results]
    jitters = [r['jitter_score'] * 100 for r in results] # 放大一点方便看
    depths = [r['depth_error'] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax2.bar(x - width/2, jitters, width, label='Jitter (x100)', alpha=0.7)
    ax2.bar(x + width/2, depths, width, label='Depth Inconsistency', alpha=0.7)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15)
    ax2.set_title("Inconsistency Scores (Higher = Fake/Unstable)")
    ax2.legend()
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / "final_analysis_result.png"
    plt.savefig(str(save_path))
    print("\n[Output] 图表已保存为 final_analysis_result.png")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(INPUT_ROOT_DIR):
        print(f"错误: 找不到文件夹 {INPUT_ROOT_DIR}，请先运行 step1_preprocess.py")
        exit()

    # 加载模型 (这里使用的是本地路径，配合上面的 Patch)
    print(f"正在加载本地模型: {LOCAL_MODEL_PATH} ...")
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("错误: 找不到模型文件，请检查 LOCAL_MODEL_PATH 路径！")
        exit()
        
    model = AsymmetricCroCo3DStereo.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)
    
    # 扫描所有预处理好的子文件夹
    subfolders = [f.path for f in os.scandir(INPUT_ROOT_DIR) if f.is_dir()]
    results = []
    
    for folder in subfolders:
        res = analyze_folder(folder, model)
        if res:
            results.append(res)
            
    # 保存数据到 JSON
    save_path = OUTPUT_DIR / "analysis_report.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    print("\n[Output] 数据已保存为 analysis_report.json")

    print("\n========== 最终检测报告 ==========")
    for r in results:
        print(f"[{r['name']}] 轨迹抖动: {r['jitter_score']:.5f} | 深度不一致性: {r['depth_error']:.5f}")
    
    # 可视化
    visualize_comparison(results)
