# Final Project: Detecting AI-Generated Videos via Geometric Consistency and Physics Constraints

## The structure of this project
```Plaintext
Project/
├── dust3r                  <-- submodule for 3D reconstruction
│   └── checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth  <-- model file
├── data/
│   ├── real_*.mp4          <-- real videos
│   └── sora2_*.mp4         <-- AI generated videos
├── preprocessed_frames/
├── results/                <-- stores the analysis results
├── preprocess.py           <-- extract 20 consecutive frames for each videos in data/
└── detect_fake_video.py    <-- analyzing using frames in preprocessed_frames/
```

## Setup

### 1. Install DUSt3R

Refer the README.md of dust3r for installation.

1. First go to the directory
```bash
cd dust3r
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n dust3r python=3.11 cmake=3.14.0
conda activate dust3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# Optional: you can also install additional packages to:
# - add support for HEIC images
# - add pyrender, used to render depthmap in some datasets preprocessing
# - add required packages for visloc.py
pip install -r requirements_optional.txt
```

3. Compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

4. Download the `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` model:
```bash
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
```

### 2. Preprocess the data

dust3r requires image frames as input and does not accept video input directly. Therefore, use `preprocess.py` to extract frames from videos. `preprocess.py` can accept 3 arguments and takes videos in `data/` as input and outputs 20 frames into individual folders in `preprocessed_frames/` per video. Zero argument also would work and the preprocess would run in default arguments.
* `--max_frames` argument indicates how many frames are to be extract. Defaultly set to 20.
* `--sample_rate` argument indicates the frequency of frame extraction. Defaultly set to 2.
* `-r` argument indicates whether to delete existing output and regenerate frames. If set, will delete existing output folder before processing.

Sample usage:
```bash
python preprocess.py --max_frames 10 -r
```

### 3. Run analysis

Directly run `detect_fake_video.py` to analyze the physics and geometry of videos. `detect_fake_video.py` requires no argument. 

Note that dust3r performs pairwise operation during analysis, and thus the number of total frames heavily influences the runtime. Specifically, 20 frames take 30~40 minutes on my PC (4070 laptop) to be analyzed and the whole experiment analyzes 4 videos and thus would take about 2 hours to be completed. The results would be both stored in the `results/` folder and presented in an interactive window after execution.

 To get a quick verification of reproducibility, use `python preprocess.py --max_frames 10` to only generate 10 frames per video. Such configuration would reduce runtime to be within minutes but may produce less accurate results. 