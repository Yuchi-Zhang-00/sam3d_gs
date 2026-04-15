import os

def find_large_masks_folders(base_path, output_file):
    results = []
    
    # 递归遍历目录
    for root, dirs, files in os.walk(base_path):
        # 检查当前文件夹名是否为 "masks" (不区分大小写可以改为 root.lower())
        if os.path.basename(root) == "masks":
            # 计算当前文件夹下的文件数量（排除子文件夹）
            file_count = len([f for f in files if os.path.isfile(os.path.join(root, f))])
            
            # 如果文件数多于 5 个，记录路径和数量
            if file_count > 5:
                results.append(f"{root} {file_count}")
                print(f"Found: {root} ({file_count} files)")

    # 将结果写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
            
    print(f"\nScan complete. Total folders found: {len(results)}")
    print(f"Results saved to: {output_file}")

# 执行逻辑
if __name__ == "__main__":
    target_path = "/home/discover/sam3d_gs/data"
    save_path = "masks.txt"
    
    if os.path.exists(target_path):
        find_large_masks_folders(target_path, save_path)
    else:
        print(f"Error: Path {target_path} does not exist.")