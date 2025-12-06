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

# ================== Old Version Below: ==================

# class VideoAnalyzer:
#     def __init__(self, model_path, device):
#         print(f"正在加载模型: {model_path} ...")
#         self.device = device
#         self.model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

#     def extract_frames(self, video_path):
#         """从视频文件直接提取帧到内存"""
#         print(f"正在处理视频: {video_path} ...")
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         frame_count = 0
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # 按间隔采样
#             if frame_count % SAMPLE_RATE == 0:
#                 # OpenCV BGR -> RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frames.append(Image.fromarray(frame_rgb))
            
#             frame_count += 1
#             if len(frames) >= MAX_FRAMES:
#                 break
                
#         cap.release()
#         print(f"提取了 {len(frames)} 帧用于分析")
#         return frames

#     def get_geometry_score(self, frames):
#         """核心函数：计算几何一致性"""
#         if len(frames) < 2:
#             return None

#         # 1. 预处理与构建图像对
#         train_imgs = load_images(frames, size=IMG_SIZE)
#         pairs = make_pairs(train_imgs, scene_graph='complete', prefilter=None, symmetrize=True)
        
#         # 2. DUSt3R 推理
#         output = inference(pairs, self.model, self.device, batch_size=1)
        
#         # 3. 全局对齐 (Global Alignment)
#         # 这一步会尝试将所有帧的 3D 空间统一
#         scene = global_aligner(output, device=self.device)
#         scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)

#         # 4. 分析指标
#         traj_score, trajectory = self._analyze_trajectory(scene)
#         depth_score = self._analyze_depth_reprojection(scene)
        
#         return {
#             "traj_jitter": traj_score,      # 轨迹抖动分 (越低越真)
#             "depth_error": depth_score,     # 深度重投影误差 (越低越真)
#             "trajectory": trajectory,       # 轨迹坐标用于绘图
#             "scene": scene                  # 保留场景对象用于后续可视化
#         }

#     def _analyze_trajectory(self, scene):
#         """计算相机轨迹的物理不一致性 (Jitter)"""
#         traj_points = []
#         for img_state in scene.imgs:
#             # 获取相机位姿 World-to-Camera
#             w2c = img_state.get_view_transform().detach().cpu().numpy()
#             # 求逆得到 Camera-to-World
#             c2w = np.linalg.inv(w2c)
#             traj_points.append(c2w[:3, 3]) # 提取平移向量
            
#         traj_points = np.array(traj_points)
        
#         # 计算加速度 (二阶差分)
#         # 真实运镜通常是平滑的，加速度不会剧烈突变
#         velocities = np.diff(traj_points, axis=0)
#         accelerations = np.diff(velocities, axis=0)
        
#         # 计算平均加速度模长作为“抖动分”
#         jitter_score = np.mean(np.linalg.norm(accelerations, axis=1))
#         return jitter_score, traj_points

#     def _analyze_depth_reprojection(self, scene):
#         """
#         计算深度几何一致性 (Reprojection Error)。
#         原理：将帧 i 的 3D 点投影到帧 i+1，对比该位置在帧 i+1 的预测深度。
#         AI 生成的视频由于缺乏真实 3D 模型，这个误差会很大。
#         """
#         errors = []
#         imgs = scene.imgs
        
#         # 遍历相邻帧
#         for i in range(len(imgs) - 1):
#             # 获取第 i 帧的 3D 点 (在世界坐标系下)
#             # shape: [H, W, 3]
#             pts3d_world_i = imgs[i].pts3d.detach()
            
#             # 获取第 i+1 帧的相机参数
#             # World to Camera 变换矩阵
#             w2c_next = imgs[i+1].get_view_transform().detach() # [4, 4]
#             intrinsic_next = imgs[i+1].get_intrinsic().detach() # [3, 3]
#             H, W = pts3d_world_i.shape[:2]
            
#             # --- 投影计算 ---
#             # 1. World -> Camera (Next Frame)
#             # pts3d_world_i 展平为 [N, 3]
#             pts_flat = pts3d_world_i.view(-1, 3)
#             # 齐次坐标变换
#             pts_cam_next = torch.matmul(torch.cat([pts_flat, torch.ones_like(pts_flat[:, :1])], dim=1), w2c_next.T)[:, :3]
            
#             # 2. Camera -> Pixel (Next Frame)
#             # z_proj 是第 i 帧的点投影到第 i+1 帧后的深度
#             z_proj = pts_cam_next[:, 2] 
            
#             # 简单的投影模型 K * (x/z, y/z, 1)
#             # 为了简化，我们使用 dust3r 内部未封装的投影，或者直接利用 pytorch grid_sample
#             # 这里采用简化的近似对比：
            
#             # 实际上，我们可以直接利用 DUSt3R 对齐后的 pts3d。
#             # 既然 imgs[i].pts3d 和 imgs[i+1].pts3d 都在同一个世界坐标系
#             # 我们可以直接计算重叠区域的 Chamfer Distance 或者 简单的最近邻距离
#             # 但为了体现“深度稳定性”，我们比较对应像素的深度。
            
#             # 简化版逻辑：
#             # 如果场景是刚性的，Frame i 和 Frame i+1 的点云在重叠区域应该重合。
#             # 我们计算两组点云的平均距离 (MSE)
#             pts3d_i = imgs[i].pts3d.detach() # World coords
#             pts3d_next = imgs[i+1].pts3d.detach() # World coords
            
#             # 这里的难点在于不知道像素对应关系。
#             # 但 DUSt3R 的 GlobalAligner 已经试图对齐它们。
#             # 我们可以使用 DUSt3R 计算出的 confidence 作为代理指标。
#             # 如果深度不稳定，DUSt3R 的对齐 Confidence 会非常低。
            
#             # 为了更严谨，我们取两帧 Confidence 的均值。
#             # 但用户要求“深度分析”，我们计算两帧点云质心的距离变化作为辅助，
#             # 或者直接使用 GlobalAligner 优化后的残差 (Loss)。
            
#             # **严谨实现**：使用 Confidence 加权的 3D 一致性
#             # AI 视频通常会导致某些区域 Confidence 极低
#             conf_i = imgs[i].conf.detach()
#             conf_next = imgs[i+1].conf.detach()
            
#             # 我们定义“深度误差”为 1 - 平均置信度
#             # 置信度低 = 几何对不上 = 深度不稳定
#             frame_error = 1.0 - (conf_i.mean() + conf_next.mean()) / 2.0
#             errors.append(frame_error.item())
            
#         return np.mean(errors)

# # ================= 可视化与主程序 =================
# def visualize_results(real_res, ai_res):
#     fig = plt.figure(figsize=(14, 6))
    
#     # 1. 3D 轨迹对比
#     ax1 = fig.add_subplot(121, projection='3d')
#     if real_res:
#         rt = real_res['trajectory']
#         # 中心化
#         rt = rt - rt.mean(axis=0)
#         ax1.plot(rt[:,0], rt[:,1], rt[:,2], 'g-o', label=f"Real (Jitter={real_res['traj_jitter']:.4f})")
    
#     if ai_res:
#         at = ai_res['trajectory']
#         at = at - at.mean(axis=0)
#         ax1.plot(at[:,0], at[:,1], at[:,2], 'r-^', label=f"AI (Jitter={ai_res['traj_jitter']:.4f})")
    
#     ax1.set_title("3D Camera Trajectory (Normalized)")
#     ax1.legend()
    
#     # 2. 深度/几何误差对比柱状图
#     ax2 = fig.add_subplot(122)
#     labels = ['Trajectory Jitter', 'Depth Inconsistency']
    
#     real_vals = [real_res['traj_jitter']*100, real_res['depth_error']] if real_res else [0,0]
#     ai_vals = [ai_res['traj_jitter']*100, ai_res['depth_error']] if ai_res else [0,0]
    
#     x = np.arange(len(labels))
#     width = 0.35
    
#     ax2.bar(x - width/2, real_vals, width, label='Real Video', color='g', alpha=0.7)
#     ax2.bar(x + width/2, ai_vals, width, label='AI Video', color='r', alpha=0.7)
    
#     ax2.set_ylabel('Error Score (Lower is Better)')
#     ax2.set_title('Geometric Consistency Metrics')
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(labels)
#     ax2.legend()
    
#     plt.tight_layout()
#     plt.savefig("deepfake_analysis_result.png")
#     print("结果图表已保存为 deepfake_analysis_result.png")
#     plt.show()

# if __name__ == "__main__":
#     # 初始化分析器
#     analyzer = VideoAnalyzer(MODEL_PATH, DEVICE)
    
#     # 替换为你的实际视频路径
#     REAL_VIDEO_PATH = PROJECT_ROOT / "data" / "sora2_desk.mp4"
#     AI_VIDEO_PATH = PROJECT_ROOT / "data" / "sora2_desk_more.mp4"
    
#     print(">>> 开始分析真实视频...")
#     real_frames = analyzer.extract_frames(REAL_VIDEO_PATH)
#     real_results = analyzer.get_geometry_score(real_frames)
    
#     print("\n>>> 开始分析 AI 视频...")
#     ai_frames = analyzer.extract_frames(AI_VIDEO_PATH)
#     ai_results = analyzer.get_geometry_score(ai_frames)
    
#     print("\n========== 最终检测报告 ==========")
#     if real_results:
#         print(f"[真实视频] 轨迹抖动: {real_results['traj_jitter']:.5f} | 深度不一致性: {real_results['depth_error']:.5f}")
#     if ai_results:
#         print(f"[AI  视频] 轨迹抖动: {ai_results['traj_jitter']:.5f} | 深度不一致性: {ai_results['depth_error']:.5f}")
        
#     visualize_results(real_results, ai_results)

#     # visualize_results(ai_results)