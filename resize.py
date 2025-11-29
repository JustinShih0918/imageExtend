import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor # 多核心神器

# 設定來源與目標
SRC_DIR = Path("data/train")
DST_DIR = Path("data/train_256")
TARGET_SIZE = (256, 256)

def process_single_image(file_path):
    try:
        # 建立對應的目標路徑
        rel_path = file_path.relative_to(SRC_DIR)
        save_path = DST_DIR / rel_path
        
        # 如果目標已存在，跳過
        if save_path.exists():
            return
            
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            # 使用 LANCZOS 或 BICUBIC 縮圖品質較好
            img = img.resize(TARGET_SIZE, Image.Resampling.BICUBIC)
            img.save(save_path, quality=95)
    except Exception:
        pass # 壞圖直接忽略

def main():
    # 1. 掃描所有檔案 (這步最快，幾秒鐘)
    print("Scanning files...")
    all_files = [p for p in SRC_DIR.rglob("*") 
                 if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
    print(f"Total images: {len(all_files)}")

    # 2. 開啟多核心處理
    # max_workers=None 會自動使用你 CPU 的最大核心數
    print("Starting multiprocessing resize...")
    with ProcessPoolExecutor() as executor:
        # 使用 tqdm 顯示進度條
        list(tqdm(executor.map(process_single_image, all_files), total=len(all_files)))

if __name__ == "__main__":
    main()