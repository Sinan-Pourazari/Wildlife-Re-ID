import os
from PIL import Image

# ==== CONFIG ====
INPUT_DIR = r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\Amur_Tigers\train"
OUTPUT_DIR = r"C:\Users\sinan\Projects\Wildlife-Re-ID\src\images\Amur_Tigers\train_resized"
SIZE = (256, 256)  # change if needed
# =================

def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

for root, dirs, files in os.walk(INPUT_DIR):
    for file in files:
        if not is_image_file(file):
            continue

        input_path = os.path.join(root, file)

        # recreate same folder structure
        rel_path = os.path.relpath(root, INPUT_DIR)
        output_folder = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, file)

        try:
            img = Image.open(input_path).convert("RGB")
            img = img.resize(SIZE, Image.BILINEAR)
            img.save(output_path, quality=95)

        except Exception as e:
            print(f"Failed: {input_path} | {e}")

print("Done.")