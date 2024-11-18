import os

def rename_files_in_directory(base_directory):
    # 遍历指定目录下的所有子目录和文件
    for root, dirs, files in os.walk(base_directory):
        for filename in files:
            # 检查文件名是否符合模式 *_eval-*.json
            if '_eval2-' in filename and filename.endswith('.json'):
                # 构造新文件名
                new_filename = filename.replace('_eval2-', '_eval-')
                # 获取文件的完整路径
                old_file_path = os.path.join(root, filename)
                new_file_path = os.path.join(root, new_filename)
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f'Renamed {old_file_path} to {new_file_path}')

# 指定你的目录路径
results_directory = 'results'
rename_files_in_directory(results_directory)
