import os
import numpy as np
from PIL import Image, ImageChops  # 导入 ImageChops

def merge_masks_in_dir(base_path):
    for root, dirs, files in os.walk(base_path):
        if os.path.basename(root) == "masks":
            # 过滤图片
            mask_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not mask_files:
                continue
            
            print(f"Processing: {root}")
            merged_np = None
            
            for mask_file in mask_files:
                mask_path = os.path.join(root, mask_file)
                try:
                    # 使用 PIL 打开并转为 numpy 数组
                    img = Image.open(mask_path).convert('RGB')
                    img_np = np.array(img)
                    
                    if merged_np is None:
                        merged_np = img_np
                    else:
                        # np.maximum 会比较两个数组相同位置的像素值，取较大者
                        # 这对黑背景(0,0,0)的 mask 合并非常有效
                        merged_np = np.maximum(merged_np, img_np)
                        
                except Exception as e:
                    print(f"Error loading {mask_path}: {e}")

            if merged_np is not None:
                # 转回 PIL 图片
                parent_dir = os.path.dirname(root)
                save_path = os.path.join(parent_dir, "object_masks.png")
                
                result_img = Image.fromarray(merged_np.astype(np.uint8))
                result_img.save(save_path)
                print(f"Successfully saved: {save_path}")

if __name__ == "__main__":
    target_path = "/home/discover/sam3d_gs/interiorgs_results/"
    if os.path.exists(target_path):
        merge_masks_in_dir(target_path)
    else:
        print(f"Path not found: {target_path}")