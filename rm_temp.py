import os
import shutil

def clear_temp_files(directory_path):
    """
    清空指定目录下的所有文件和子目录。
    
    参数:
    - directory_path (str): 要清空的目录路径。
    """
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"已删除文件: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

        for name in dirs:
            dir_path = os.path.join(root, name)
            try:
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"已删除目录: {dir_path}")
            except Exception as e:
                print(f"Error deleting {dir_path}: {e}")

if __name__ == "__main__":
    temp_path = "C:\\Users\\HYC11\\AppData\\Local\\Temp"
    clear_temp_files(temp_path)
