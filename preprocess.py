import cv2
import os
import glob
import argparse
from pathlib import Path

# ================= Configuration =================
# Folder containing input videos (script automatically finds .mp4, .mov, .avi)
PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_INPUT_DIR = PROJECT_ROOT / "data" 

# Root directory for saving output images
OUTPUT_ROOT_DIR = PROJECT_ROOT / "preprocessed_frames"

RESIZE_HEIGHT = 512   # Uniform height adjustment (maintain aspect ratio)

def process_video(video_path, sample_rate=2, max_frame=20, reset_output=False):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(OUTPUT_ROOT_DIR, video_name)
    
    if os.path.exists(save_dir) and not reset_output:
        print(f"[Skip] Folder already exists: {save_dir}")
        return
    elif os.path.exists(save_dir) and reset_output:
        import shutil
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    print(f"Processing: {video_name} ...")

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sampling logic
        if frame_idx % sample_rate == 0:
            # Resize processing
            if RESIZE_HEIGHT:
                h, w = frame.shape[:2]
                scale = RESIZE_HEIGHT / h
                new_w = int(w * scale)
                frame = cv2.resize(frame, (new_w, RESIZE_HEIGHT))
            
            # Save image (named sequentially 00000.jpg)
            save_path = os.path.join(save_dir, f"{saved_count:05d}.jpg")
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
            if saved_count >= max_frame:
                break
        
        frame_idx += 1

    cap.release()
    print(f" -> Saved {saved_count} frames to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video preprocessing script")
    parser.add_argument("--sample_rate", type=int, default=2, help="Take one frame every few frames")
    parser.add_argument("--max_frames", type=int, default=20, help="Maximum frames to extract per video")
    parser.add_argument("-r", action="store_true", default=False, help="Indicator for reset output. If set, will delete existing output folder before processing.")
    args = parser.parse_args()
    SAMPLE_RATE = args.sample_rate
    MAX_FRAMES = args.max_frames
    # Ensure input folder exists
    if not os.path.exists(VIDEO_INPUT_DIR):
        os.makedirs(VIDEO_INPUT_DIR)
        print(f"Please put your video files in '{VIDEO_INPUT_DIR}' folder and re-run.")
    else:
        # Scan all video files
        video_files = glob.glob(os.path.join(VIDEO_INPUT_DIR, "*.*"))
        video_files = [f for f in video_files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print(f"No video files found in '{VIDEO_INPUT_DIR}'.")
        else:
            for v_path in video_files:
                process_video(v_path, sample_rate=SAMPLE_RATE, max_frame=MAX_FRAMES, reset_output=args.r)
            print("\nPreprocessing complete! Now please run detect_fake_video.py")