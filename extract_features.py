import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import dcsass_loader
from config import config

# Setup
sys.path.append(os.getcwd())

def extract_features():
    # 1. Setup Model (ResNet50V2)
    print("Loading ResNet50V2...")
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        pooling="avg" # Global Average Pooling -> (2048,)
    )
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    
    # 2. Setup Paths
    data_root = Path(config.DATA_DIR)
    output_root = Path("features")
    output_root.mkdir(exist_ok=True)
    
    print(f"Data Root: {data_root}")
    print(f"Output Root: {output_root}")
    
    # 3. Get Video List (Using loader logic)
    # We want ALL videos, so we'll just scan the directories manually or use loader
    # Let's use the loader's metadata function but ignore splits
    metadata = dcsass_loader._load_metadata(data_root)
    
    # Filter for the labels we care about (Arrest, Fighting) + Normal if available
    # Or just extract everything found
    
    print(f"Found {len(metadata)} videos.")
    
    # 4. Extraction Loop
    for entry in tqdm(metadata, desc="Extracting Features"):
        label = entry["label"]
        video_path = Path(entry["path"])
        
        # Create output directory
        save_dir = output_root / label
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f"{video_path.stem}.npy"
        
        if save_path.exists():
            continue
            
        try:
            # Decode Video
            # We use the same logic as training: T=16, stride=16, max_frames=1000
            # BUT for extraction, we usually want *all* features and handle sampling during training.
            # However, to save space, let's stick to the training sampling strategy OR 
            # just extract frames at a fixed fps.
            # Let's extract ALL frames (or max 1000) and save them. 
            # The loader usually handles sampling. 
            # Better approach: Save (N, 2048) where N is number of frames (subsampled).
            
            # Let's use dcsass_loader.decode_video_opencv directly
            frames = dcsass_loader.decode_video_opencv(
                str(video_path), 
                target_size=(224, 224),
                max_frames=1000 # Safety limit
            )
            
            if len(frames) == 0:
                # If no frames, still delete video if it's broken/empty to save space? 
                # Maybe safer to keep for inspection, but for now let's skip.
                continue
                
            # Preprocess for ResNet
            # frames is a list of arrays, convert to stack
            frames_arr = np.array(frames, dtype=np.float32)
            frames_preprocessed = tf.keras.applications.resnet_v2.preprocess_input(frames_arr)
            
            # Extract Features
            # Batch processing to avoid OOM on local machine
            features = model.predict(frames_preprocessed, batch_size=32, verbose=0)
            
            # Save
            np.save(save_path, features.astype(np.float16)) # Save as float16 to save space
            
            # Delete source video to free up space
            if video_path.exists():
                video_path.unlink()
            
        except Exception as e:
            # If we run out of space, we need to know
            if "Disk quota exceeded" in str(e):
                print(f"CRITICAL ERROR: Disk Full processing {video_path}")
                # Try to delete the partial file if it was created
                if save_path.exists():
                    save_path.unlink()
                raise e # Stop execution
            print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    extract_features()
