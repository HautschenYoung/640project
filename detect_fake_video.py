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

# ================= Critical Patch: Fix PyTorch 2.6+ Pickle Error =================
# Before importing dust3r, we patch torch.load to force allowing non-Tensor objects.
# This allows loading old checkpoints without modifying dust3r source code.
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # If weights_only is not specified in the call, and the current torch version supports it, force it to False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Apply patch
torch.load = patched_torch_load
print("[System] Applied PyTorch weights_only=False patch for DUSt3R model compatibility.")

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# ================= Configuration Area =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "dust3r" / "checkpoints" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

if not LOCAL_MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {LOCAL_MODEL_PATH}\nPut the checkpoint at this path or update MODEL_PATH.")

IMG_SIZE = 512  # Image processing resolution
INPUT_ROOT_DIR = PROJECT_ROOT / "preprocessed_frames"  # Corresponds to preprocessing output directory
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.float32): return float(obj)
        return super().default(obj)

def analyze_folder(folder_path, model):
    print(f"\nAnalyzing folder: {folder_path} ...")
    
    # 1. Load images
    image_files = sorted(glob(os.path.join(folder_path, '*.jpg')) + glob(os.path.join(folder_path, '*.png')))
    if len(image_files) < 2:
        print("Insufficient number of images, skipping")
        return None

    train_imgs = load_images(image_files, size=IMG_SIZE)
    
    # 2. Inference
    pairs = make_pairs(train_imgs, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, DEVICE, batch_size=1)
    
    # 3. Global alignment
    scene = global_aligner(output, device=DEVICE)
    scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
    
    # ================= Use Standard API to Get Data =================
    
    # --- 1: Trajectory Retrieval (Physics) ---
    # Directly call scene.get_im_poses()
    # It returns a Tensor shape [N, 4, 4], representing World-to-Camera matrices
    w2c_matrices = scene.get_im_poses().detach().cpu().numpy()
    
    traj_points = []
    for w2c in w2c_matrices:
        # DUSt3R outputs World-to-Camera, we need to invert it to get Camera-to-World (camera position in world)
        c2w = np.linalg.inv(w2c)
        traj_points.append(c2w[:3, 3]) # Get translation vector (x, y, z)
    
    traj_points = np.array(traj_points)
    
    # Calculate jitter score
    if len(traj_points) < 3:
        jitter_score = 0.0 # Too few frames to calculate second-order difference
    else:
        velocities = np.diff(traj_points, axis=0)
        accelerations = np.diff(velocities, axis=0)
        jitter_score = np.mean(np.linalg.norm(accelerations, axis=1))
    
    # --- 2: Depth/Confidence Retrieval (Geometry) ---
    # Directly access scene.im_conf
    # scene.im_conf is a ParameterList containing confidence maps for all frames
    confs = []
    for conf_map in scene.im_conf:
        # conf_map is (H, W) or (1, H, W) Tensor
        # Need exp because internal storage is usually log-confidence, or take mean directly as relative metric
        # Here for stability, we take mean directly
        confs.append(conf_map.detach().cpu().numpy().mean())
    
    # Calculate depth inconsistency (1 - average confidence)
    depth_inconsistency = 1.0 - np.mean(confs)
    
    # ================= Fix End =================
    
    return {
        "name": os.path.basename(folder_path),
        "jitter_score": float(jitter_score),
        "depth_error": float(depth_inconsistency),
        "trajectory": traj_points
    }

def visualize_comparison(results):
    """Plot analysis results of all videos on one chart for comparison"""
    if not results: return
    
    fig = plt.figure(figsize=(15, 6))
    
    # Subplot 1: 3D Trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    for res in results:
        traj = np.array(res['trajectory'])
        traj = traj - traj.mean(axis=0) # Center to compare shapes
        ax1.plot(traj[:,0], traj[:,1], traj[:,2], marker='o', label=res['name'])
    ax1.set_title("Normalized Camera Trajectories")
    ax1.legend()
    
    # Subplot 2: Score Comparison
    ax2 = fig.add_subplot(122)
    names = [r['name'] for r in results]
    jitters = [r['jitter_score'] * 100 for r in results] # Scale up for visibility
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
    print("\n[Output] Chart saved as final_analysis_result.png")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(INPUT_ROOT_DIR):
        print(f"Error: Folder {INPUT_ROOT_DIR} not found, please run step1_preprocess.py first")
        exit()

    # Load model (using local path here, with the Patch above)
    print(f"Loading local model: {LOCAL_MODEL_PATH} ...")
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Error: Model file not found, please check LOCAL_MODEL_PATH!")
        exit()
        
    model = AsymmetricCroCo3DStereo.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)
    
    # Scan all preprocessed subfolders
    subfolders = [f.path for f in os.scandir(INPUT_ROOT_DIR) if f.is_dir()]
    results = []
    
    for folder in subfolders:
        res = analyze_folder(folder, model)
        if res:
            results.append(res)
            
    # Save data to JSON
    save_path = OUTPUT_DIR / "analysis_report.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    print("\n[Output] Data saved as analysis_report.json")

    print("\n========== Final Detection Report ==========")
    for r in results:
        print(f"[{r['name']}] Trajectory Jitter: {r['jitter_score']:.5f} | Depth Inconsistency: {r['depth_error']:.5f}")
    
    # Visualize
    visualize_comparison(results)
