import os
import zipfile
from pathlib import Path

LABELS_TO_INCLUDE = ['Arrest', 'Fighting']
DATA_ROOT = Path('data/DCSASS Dataset')
OUTPUT_ZIP = 'subset_data.zip'


def zip_subset():
    """Create a zip file with selected label directories."""
    if not DATA_ROOT.exists():
        print(f"Error: {DATA_ROOT} does not exist.")
        return

    print(f"Creating {OUTPUT_ZIP} with labels: {LABELS_TO_INCLUDE}...")
    
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        labels_dir = DATA_ROOT / 'Labels'
        if labels_dir.exists():
            for file in labels_dir.glob('*'):
                zipf.write(file, arcname=str(file))
                print(f"Added metadata: {file.name}")
        
        for label in LABELS_TO_INCLUDE:
            label_dir = DATA_ROOT / label
            if not label_dir.exists():
                print(f"Warning: Label directory {label_dir} not found. Skipping.")
                continue
                
            print(f"Adding label: {label}...")
            for root, dirs, files in os.walk(label_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path
                    zipf.write(file_path, arcname=str(arcname))
                    
    print(f"Successfully created {OUTPUT_ZIP}")


if __name__ == "__main__":
    zip_subset()
