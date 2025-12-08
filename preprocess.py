import cv2
import os
import glob
import argparse
from pathlib import Path

# ================= 配置 =================
# 输入视频所在的文件夹 (脚本会自动寻找该文件夹下的 .mp4, .mov, .avi)
PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_INPUT_DIR = PROJECT_ROOT / "data" 

# 输出图片保存的根目录
OUTPUT_ROOT_DIR = PROJECT_ROOT / "preprocessed_frames"

# 抽帧设置
# 每隔几帧取一张
# 每个视频最多提取多少帧 (防止显存爆炸)
RESIZE_HEIGHT = 512   # 统一调整高度 (保持长宽比)，None 则不缩放

def process_video(video_path, sample_rate=2, max_frame=20, reset_output=False):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(OUTPUT_ROOT_DIR, video_name)
    
    if os.path.exists(save_dir) and not reset_output:
        print(f"[跳过] 文件夹已存在: {save_dir}")
        return
    elif os.path.exists(save_dir) and reset_output:
        import shutil
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    print(f"正在处理: {video_name} ...")

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 采样逻辑
        if frame_idx % sample_rate == 0:
            # 缩放处理 (可选)
            if RESIZE_HEIGHT:
                h, w = frame.shape[:2]
                scale = RESIZE_HEIGHT / h
                new_w = int(w * scale)
                frame = cv2.resize(frame, (new_w, RESIZE_HEIGHT))
            
            # 保存图片 (按顺序命名 00000.jpg)
            save_path = os.path.join(save_dir, f"{saved_count:05d}.jpg")
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
            if saved_count >= max_frame:
                break
        
        frame_idx += 1

    cap.release()
    print(f" -> 已保存 {saved_count} 帧到 {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频预处理脚本")
    parser.add_argument("--sample_rate", type=int, default=2, help="每隔几帧取一张")
    parser.add_argument("--max_frames", type=int, default=20, help="每个视频最多提取多少帧")
    parser.add_argument("-r", action="store_true", default=False, help="Indicator for reset output. If set, will delete existing output folder before processing.")
    args = parser.parse_args()
    SAMPLE_RATE = args.sample_rate
    MAX_FRAMES = args.max_frames
    # 确保有输入文件夹
    if not os.path.exists(VIDEO_INPUT_DIR):
        os.makedirs(VIDEO_INPUT_DIR)
        print(f"请将你的视频文件放入 '{VIDEO_INPUT_DIR}' 文件夹中，然后重新运行。")
    else:
        # 扫描所有视频文件
        video_files = glob.glob(os.path.join(VIDEO_INPUT_DIR, "*.*"))
        video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print(f"在 '{VIDEO_INPUT_DIR}' 中没找到视频文件。")
        else:
            for v_path in video_files:
                process_video(v_path, sample_rate=SAMPLE_RATE, max_frame=MAX_FRAMES, reset_output=args.r)
            print("\n预处理完成！现在请运行 step2_analyze.py")