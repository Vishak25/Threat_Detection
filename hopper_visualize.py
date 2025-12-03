import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import random
from config import config
from model import MILScoringHead

# Setup
sys.path.append(os.getcwd())
config.BATCH_SIZE = 1

# --- Label Loader (Copied from hopper_train.py) ---
def load_ground_truth():
    label_dir = Path.home() / "surveillance/camera_anomaly_detection_v2/data/DCSASS Dataset/Labels"
    if not label_dir.exists():
        local_fallback = Path("data/DCSASS Dataset/Labels")
        if local_fallback.exists():
            label_dir = local_fallback
            
    print(f"Loading labels from {label_dir}...")
    video_to_label = {}
    if not label_dir.exists():
        return video_to_label

    for csv_path in label_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path, header=None, names=["name", "class", "label"])
            for _, row in df.iterrows():
                video_to_label[str(row["name"])] = int(row["label"])
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
    return video_to_label

GROUND_TRUTH = load_ground_truth()

# --- Feature Loader (Copied from hopper_train.py) ---
def make_feature_dataset(feature_dir, split="train", val_split=0.2, seed=1337):
    feature_dir = Path(feature_dir)
    files = sorted(list(feature_dir.rglob("*.npy")))
    
    # Shuffle
    random.seed(seed)
    random.shuffle(files)
    
    # Split
    split_idx = int(len(files) * (1 - val_split))
    if split == "train":
        files = files[:split_idx]
    else:
        files = files[split_idx:]
        
    print(f"Found {len(files)} files for split '{split}'")
    
    def generator():
        for f in files:
            try:
                feats = np.load(f)
                # Get Label from CSV
                label = GROUND_TRUTH.get(f.stem, 1)
                yield feats, np.int32(label), str(f.name)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue

    output_signature = (
        tf.TensorSpec(shape=(None, 2048), dtype=tf.float16),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.string),
    )
    
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.batch(1).prefetch(tf.data.AUTOTUNE)
    return ds

# --- Load Data ---
SCRATCH_DIR = Path("/scratch/vnandak/features")
HOME_DIR = Path("features")

if SCRATCH_DIR.exists():
    FEATURES_DIR = SCRATCH_DIR
    print(f"Using features from scratch: {FEATURES_DIR}")
elif HOME_DIR.exists():
    FEATURES_DIR = HOME_DIR
    print(f"Using features from home/local: {FEATURES_DIR}")
else:
    FEATURES_DIR = Path("features")
    print(f"Using default features path: {FEATURES_DIR}")

print("Loading validation dataset...")
val_ds = make_feature_dataset(FEATURES_DIR, split="val")

# --- Load Model ---
print("Loading MIL Scoring Head...")
mil_head = MILScoringHead()
mil_head.build((None, 2048))

ckpt_dir = config.CHECKPOINT_DIR
latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
if not latest_ckpt:
    ckpts = sorted(list(ckpt_dir.glob("model_epoch_*.weights.h5")), key=lambda p: int(p.name.split('_')[-1].split('.')[0]))
    if ckpts:
        latest_ckpt = str(ckpts[-1])

if latest_ckpt:
    print(f"Loading weights from {latest_ckpt}")
    try:
        mil_head.load_weights(latest_ckpt)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)
else:
    print("No checkpoints found!")
    sys.exit(1)

# --- Visualize ---
print("\n--- Visualizing Predictions ---")
print(f"{'Video ID':<30} | {'Max Score':<10} | {'Avg Score':<10} | {'Pred':<5} | {'Label':<5}")
print("-" * 70)

for i, (bag, label, vid_id_tensor) in enumerate(val_ds.take(20)):
    vid_id = vid_id_tensor.numpy()[0].decode('utf-8')
    instances = bag[0] # (N, 2048)
    
    # Predict
    # Cast to float32
    instances = tf.cast(instances, tf.float32)
    scores = mil_head(instances, training=False)
    scores = scores.numpy().flatten()
    
    max_score = np.max(scores)
    avg_score = np.mean(scores)
    pred_label = 1 if max_score > 0.5 else 0
    true_label = int(label.numpy()[0])
    
    print(f"{vid_id:<30} | {max_score:.4f}     | {avg_score:.4f}     | {pred_label:<5} | {true_label:<5}")
    
    if pred_label != true_label or true_label == 1:
        print(f"  Scores (First 10): {scores[:10]}")
        print(f"  Scores (Max Index): {np.argmax(scores)} (Val: {max_score:.4f})")
        print("-" * 70)

