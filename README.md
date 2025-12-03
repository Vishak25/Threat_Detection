# Camera Anomaly Detection V2 - Hopper Deployment

This guide explains how to deploy and run the project on the GMU Hopper cluster.

## Prerequisites
- **Local Machine**: macOS/Linux with `scp` and `rsync`.
- **Remote Machine**: GMU Hopper (`hopper.orc.gmu.edu`).
- **Data**: DCSASS Dataset (or subset).

## 1. Sync Code to Hopper
**Run these commands in your LOCAL Terminal (macOS):**

```bash
# 1. Navigate to project directory
cd /Users/vishaknandakumar/Documents/camera_anomaly_detection_v2

# 2. Sync code (excluding data)
# Note: We use rsync because macOS scp doesn't support --exclude
rsync -av --exclude='data' --exclude='data.zip' --exclude='.git' --exclude='.venv' --exclude='__pycache__' ./ vnandak@hopper.orc.gmu.edu:~/surveillance/camera_anomaly_detection_v2/

# 3. Sync subset data (if needed)
# If you created a subset zip locally:
scp subset_data.zip vnandak@hopper.orc.gmu.edu:~/surveillance/camera_anomaly_detection_v2/
```

## 2. Setup Environment on Hopper
**Run these commands in your SSH Terminal (Hopper):**

```bash
# 1. Connect to Hopper
ssh vnandak@hopper.orc.gmu.edu

# 2. Go to project directory
cd ~/surveillance/camera_anomaly_detection_v2

# 3. Unzip data (if using subset)
unzip -o subset_data.zip

# 4. Run Setup Script (Installs dependencies)
bash hopper_setup.sh
```

## 3. Run Training
**Run these commands in your SSH Terminal (Hopper):**

**CRITICAL**: You must export the library paths before running python, otherwise TensorFlow won't find the GPU.

```bash
# 1. Load Modules
module load cuda/12.6.3
module load cudnn/9.6.0.74-12.6.3

# 2. Export Library Path (Critical for GPU)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib:$(python3 -c "import nvidia.cublas; print(nvidia.cublas.__path__[0])")/lib:$(python3 -c "import nvidia.cufft; print(nvidia.cufft.__path__[0])")/lib:$(python3 -c "import nvidia.cusparse; print(nvidia.cusparse.__path__[0])")/lib:$(python3 -c "import nvidia.cusolver; print(nvidia.cusolver.__path__[0])")/lib:$(python3 -c "import nvidia.curand; print(nvidia.curand.__path__[0])")/lib

# 3. Run Visualization
python3 hopper_visualize.py

# 4. Run Training Script
python3 hopper_train.py
```

## Environment Reference (Golden Configuration)
These are the exact versions that are confirmed to work on Hopper:

*   **Cluster Modules**:
    *   `cuda/12.6.3`
    *   `cudnn/9.6.0.74-12.6.3`
*   **Python Packages**:
    *   `tensorflow==2.16.1`
    *   `numpy<2.0.0` (Pinned to avoid conflict)
    *   `scipy<1.12` (Pinned to avoid conflict)
*   **GPU Libraries**:
    *   We use the **pip-installed** NVIDIA libraries (e.g., `nvidia-cudnn-cu12==8.9.7`) by exporting `LD_LIBRARY_PATH`. This is more reliable than system modules for TF 2.16.

## Troubleshooting
- **"Killed"**: This means the process ran out of System RAM (not GPU memory).
    - **Fix**: The script `hopper_train.py` now has `max_frames=1000` to prevent loading huge videos. Ensure you synced the latest version.
- **"Cannot dlopen some GPU libraries"**: You forgot to run the `export LD_LIBRARY_PATH...` command above.




