import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import dcsass_loader
from config import config

sys.path.append(os.getcwd())


def extract_features():
    """Extract ResNet50V2 features from videos and save as .npy files."""
    print("Loading ResNet50V2...")
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    model = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    
    data_root = Path(config.DATA_DIR)
    output_root = Path("features")
    output_root.mkdir(exist_ok=True)
    
    print(f"Data Root: {data_root}")
    print(f"Output Root: {output_root}")
    
    metadata = dcsass_loader._load_metadata(data_root)
    print(f"Found {len(metadata)} videos.")
    
    for entry in tqdm(metadata, desc="Extracting Features"):
        label = entry["label"]
        video_path = Path(entry["path"])
        
        save_dir = output_root / label
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f"{video_path.stem}.npy"
        
        if save_path.exists():
            continue
            
        try:
            frames = dcsass_loader.decode_video_opencv(
                str(video_path), 
                target_size=(224, 224),
                max_frames=1000
            )
            
            if len(frames) == 0:
                continue
                
            frames_arr = np.array(frames, dtype=np.float32)
            frames_preprocessed = tf.keras.applications.resnet_v2.preprocess_input(frames_arr)
            
            features = model.predict(frames_preprocessed, batch_size=32, verbose=0)
            
            np.save(save_path, features.astype(np.float16))
            
            if video_path.exists():
                video_path.unlink()
            
        except Exception as e:
            if "Disk quota exceeded" in str(e):
                print(f"CRITICAL ERROR: Disk Full processing {video_path}")
                if save_path.exists():
                    save_path.unlink()
                raise e
            print(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    extract_features()
