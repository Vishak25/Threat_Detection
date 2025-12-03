import os
import sys
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import numpy as np
import glob
import random
import pandas as pd

# Add current directory to path
sys.path.append(os.getcwd())

from config import config
from model import MILScoringHead

# --- Configuration ---
config.EPOCHS = 20
config.BATCH_SIZE = 1 # MIL standard
config.LEARNING_RATE = 1e-4 # Adjust as needed

print("--- Configuration ---")
config.display()

# --- Environment Check ---
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Enable Mixed Precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed precision policy set to: {policy.name}")

# --- Label Loader ---
def load_ground_truth():
    # Path to labels on Hopper
    # User uploaded to: ~/surveillance/camera_anomaly_detection_v2/data/DCSASS Dataset/Labels
    label_dir = Path.home() / "surveillance/camera_anomaly_detection_v2/data/DCSASS Dataset/Labels"
    
    # Fallback for local testing if home path doesn't exist
    if not label_dir.exists():
        local_fallback = Path("data/DCSASS Dataset/Labels")
        if local_fallback.exists():
            label_dir = local_fallback
        
    print(f"Loading labels from {label_dir}...")
    video_to_label = {}
    
    if not label_dir.exists():
        print("WARNING: Label directory not found! All labels will default to 1 (Anomaly).")
        return video_to_label

    # We only care about the classes we are training on, but loading all is fine/fast
    csv_files = list(label_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files.")

    for csv_path in csv_files:
        try:
            # CSV Format: name, class, label (0 or 1)
            # Example: Arrest001_x264_0,Arrest,0
            df = pd.read_csv(csv_path, header=None, names=["name", "class", "label"])
            for _, row in df.iterrows():
                video_to_label[str(row["name"])] = int(row["label"])
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            
    print(f"Loaded {len(video_to_label)} ground truth labels.")
    return video_to_label

# Load labels once
GROUND_TRUTH = load_ground_truth()

# --- Feature Loader ---
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
                # Load features: (N, 2048)
                feats = np.load(f)
                
                # Get Label from CSV
                # f.stem is the filename without extension (e.g. Arrest001_x264_0)
                # Default to 1 (Anomaly) if not found, but log it?
                label = GROUND_TRUTH.get(f.stem, 1)
                
                # Yield: (Bag, Label, VidID)
                yield feats, np.int32(label), str(f.name)
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue

    output_signature = (
        tf.TensorSpec(shape=(None, 2048), dtype=tf.float16), # Features are float16
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.string),
    )
    
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.batch(1).prefetch(tf.data.AUTOTUNE)
    return ds

# --- Load Data ---
print("Creating datasets...")
# Use scratch directory for features to avoid quota issues
# We will assume the user uploads to /scratch/vnandak/features
# Or we can check if it exists there, otherwise fallback to local
SCRATCH_DIR = Path("/scratch/vnandak/features")
HOME_DIR = Path("features")

if SCRATCH_DIR.exists():
    FEATURES_DIR = SCRATCH_DIR
    print(f"Using features from scratch: {FEATURES_DIR}")
elif HOME_DIR.exists():
    FEATURES_DIR = HOME_DIR
    print(f"Using features from home/local: {FEATURES_DIR}")
else:
    # Fallback for when running on local machine
    FEATURES_DIR = Path("features")
    print(f"Using default features path: {FEATURES_DIR}")

train_ds = make_feature_dataset(FEATURES_DIR, split="train")
val_ds = make_feature_dataset(FEATURES_DIR, split="val")
print("Datasets created.")

# --- Build Model ---
# We use the head directly since inputs are already features
model = MILScoringHead()
# Build the model to print summary
model.build((None, 2048))
model.summary()

# --- Training Loop ---
optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
bce_loss = tf.keras.losses.BinaryCrossentropy()

# Regularization weights
LAMBDA_SPARSITY = 8e-5
LAMBDA_SMOOTHNESS = 0.01

@tf.function
def train_step(bag, label):
    # bag: (1, N, 2048)
    instances = bag[0] # (N, 2048)
    
    with tf.GradientTape() as tape:
        # Cast to float32 for stability if needed, but model handles mixed precision
        # instances = tf.cast(instances, tf.float32) 
        
        scores = model(instances, training=True) # (N, 1)
        
        # Cast scores to float32 for loss calculation stability and to avoid mixed precision issues
        scores = tf.cast(scores, tf.float32)
        
        # 1. Classification Loss (Push max score to 1 for Anomaly bags)
        max_score = tf.reduce_max(scores)
        max_score_expanded = tf.expand_dims(max_score, 0)
        cls_loss = bce_loss(label, max_score_expanded)
        
        # 2. Sparsity Loss (Push average score to 0)
        sparsity_loss = tf.reduce_mean(scores)
        
        # 3. Smoothness Loss (Push adjacent scores to be similar)
        if tf.shape(scores)[0] > 1:
            smoothness_loss = tf.reduce_mean(tf.square(scores[1:] - scores[:-1]))
        else:
            smoothness_loss = 0.0
        
        # Losses are already float32 now
        
        # Total Loss
        loss = cls_loss + (LAMBDA_SPARSITY * sparsity_loss) + (LAMBDA_SMOOTHNESS * smoothness_loss)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, max_score

# Create checkpoint directory
config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print("Starting training...")
for epoch in range(config.EPOCHS):
    print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
    
    # Training
    total_loss = 0.0
    steps = 0
    for bag, label, vid_id in tqdm(train_ds, desc="Training"):
        loss, score = train_step(bag, label)
        total_loss += loss
        steps += 1
        
    avg_loss = total_loss / steps if steps > 0 else 0
    print(f"Train Loss: {avg_loss:.4f}")
    
    # Validation (Run every 5 epochs)
    if (epoch + 1) % 5 == 0:
        val_loss = 0.0
        val_steps = 0
        correct = 0
        total = 0
        
        for bag, label, vid_id in tqdm(val_ds, desc="Validation"):
            instances = bag[0]
            scores = model(instances, training=False)
            max_score = tf.reduce_max(scores)
            max_score_expanded = tf.expand_dims(max_score, 0)
            loss = bce_loss(label, max_score_expanded)
            
            val_loss += loss
            val_steps += 1
            
            pred = 1 if max_score > 0.5 else 0
            if pred == label[0]:
                correct += 1
            total += 1
            
        if val_steps > 0:
            print(f"Val Loss: {val_loss/val_steps:.4f}, Val Acc: {correct/total:.4f}")
        else:
            print("Val Loss: N/A, Val Acc: N/A")
    
    # Save Checkpoint
    if (epoch + 1) % 5 == 0:
        ckpt_path = config.CHECKPOINT_DIR / f"model_epoch_{epoch+1}.weights.h5"
        model.save_weights(str(ckpt_path))
        print(f"Saved checkpoint to {ckpt_path}")
