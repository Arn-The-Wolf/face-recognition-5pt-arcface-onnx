import urllib.request
import os
import shutil
import zipfile

# Official InsightFace release for buffalo_l (contains w600k_r50.onnx)
url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
target_dir = "models"
zip_file = os.path.join(target_dir, "buffalo_l.zip")
target_model_name = "w600k_r50.onnx"
final_dest = os.path.join(target_dir, "embedder_arcface.onnx")

os.makedirs(target_dir, exist_ok=True)

print(f"Downloading {url} to {zip_file}...")
try:
    with urllib.request.urlopen(url) as response, open(zip_file, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print("Download complete.")
    
    print(f"Extracting {target_model_name}...")
    with zipfile.ZipFile(zip_file, 'r') as zf:
        # Check if file is in root or subfolder
        found = False
        for name in zf.namelist():
            if name.endswith(target_model_name):
                print(f"Found {name} in zip.")
                with zf.open(name) as source, open(final_dest, 'wb') as target:
                    shutil.copyfileobj(source, target)
                found = True
                break
        
        if not found:
            raise RuntimeError(f"{target_model_name} not found in zip.")

    print(f"Model saved to {final_dest}")
    
    # Cleanup zip
    os.remove(zip_file)
    print("Cleanup done.")
    
except Exception as e:
    print(f"Process failed: {e}")
    exit(1)
