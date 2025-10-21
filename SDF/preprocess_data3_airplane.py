import sys
import os
import glob
import logging
import numpy as np
import trimesh
from tqdm import tqdm

# 确保可以找到 mesh_to_sdf 套件 (假设你的环境已配置正确)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接從 mesh_to_sdf 的核心模組導入函式
from mesh_to_sdf.mesh_to_sdf import sample_sdf_near_surface

def save_sdf_to_npz(filepath, points, sdf):
    """將採樣點和SDF值保存為 .npz 檔案"""
    pos_points = points[sdf >= 0]
    neg_points = points[sdf < 0]
    pos_sdf = sdf[sdf >= 0]
    neg_sdf = sdf[sdf < 0]

    np.savez(
        filepath, 
        pos=np.hstack([pos_points, pos_sdf[:, np.newaxis]]),
        neg=np.hstack([neg_points, neg_sdf[:, np.newaxis]])
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='DeepSdf - %(levelname)s - %(message)s')

    # --- 1. 【重要】在這裡調整參數以提高品質 ---
    SCAN_COUNT = 400
    SCAN_RESOLUTION = 1024
    SIGN_METHOD = 'depth'  # 'normal' 或 'depth'

    # --- 2. 設定你的輸入和輸出路徑 ---
    
    # **【请在此处替换为你的 .off 文件所在的文件夹路径】**
    # 例如: r'E:\MyProject\My10OffFiles'
    input_dir = r'F:\Code\train\modelnet_airplane' 
    
    # 輸出目錄，所有 .npz 檔案將保存在這裡
    output_dir = r'F:\Code\pointnet2\DeepSDF\Data2' 

    # --- 3. 查找所有匹配的 .off 檔案 ---
    # 使用 os.path.join 组合路径，确保跨操作系统兼容性
    input_pattern = os.path.join(input_dir, '*.off')
    
    print(f"正在从目录 '{input_dir}' 中搜索 .off 檔案...")
    # 使用 glob.glob 查找所有匹配 .off 扩展名的文件
    all_mesh_files = glob.glob(input_pattern)
    
    if not all_mesh_files:
        print(f"錯誤：在指定路徑 '{input_dir}' 下沒有找到任何 .off 檔案。請檢查路徑是否正確。")
        sys.exit(1)

    files_to_process = all_mesh_files # 处理所有找到的 .off 文件
    print(f"找到了 {len(files_to_process)} 个 .off 文件，将全部处理。")

    os.makedirs(output_dir, exist_ok=True)
    print(f"輸出目錄設定為: '{output_dir}'")
    print(f"使用參數: scan_count={SCAN_COUNT}, scan_resolution={SCAN_RESOLUTION}, sign_method='{SIGN_METHOD}'")

    # --- 4. 循環處理每一個檔案 ---
    print("\n開始批次處理...")
    for mesh_filepath in tqdm(files_to_process, desc="處理進度"):
        instance_name = os.path.splitext(os.path.basename(mesh_filepath))[0]
        output_npz_path = os.path.join(output_dir, instance_name + ".npz")
        
        # 检查是否需要跳过已处理的文件
        skip_processing = False 
        if skip_processing and os.path.exists(output_npz_path):
            continue
            
        try:
            # trimesh.load 可以直接处理 .off 文件
            mesh = trimesh.load(mesh_filepath) 
            
            # 直接呼叫核心函式，並傳入我們自訂的參數
            points, sdf = sample_sdf_near_surface(
                mesh, 
                number_of_points=500000, # 總採樣點數
                surface_point_method='scan', 
                sign_method=SIGN_METHOD,
                scan_count=SCAN_COUNT,
                scan_resolution=SCAN_RESOLUTION
            )
            
            # 保存為 .npz 檔案
            save_sdf_to_npz(output_npz_path, points, sdf)

        except Exception as e:
            print(f"\n處理檔案 {mesh_filepath} 時發生錯誤: {e}")

    print("\n批次處理完成！")