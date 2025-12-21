import os
import yaml
import shutil
from huggingface_hub import snapshot_download

CONFIG_FILE = 'models.yml'
DESTINATION_PATHS = {
    "generator": "./models/generator",
    "guardrail": "./models/guardrail",
}

def handle_backup(dest_path):
    """Xử lý việc sao lưu thư mục model hiện có."""
    backup_path = dest_path + "_bak"
    
    # 1. Xóa thư mục backup cũ nếu có
    if os.path.exists(backup_path):
        print(f"Đang xóa thư mục backup cũ tại: {backup_path}")
        try:
            shutil.rmtree(backup_path)
            print("Đã xóa xong thư mục backup.")
        except OSError as e:
            print(f"Lỗi khi xóa thư mục backup {backup_path}: {e}")
            return False # Báo hiệu thất bại

    # 2. Đổi tên thư mục model hiện tại thành backup
    if os.path.exists(dest_path):
        print(f"Đang sao lưu thư mục model hiện tại từ '{dest_path}' thành '{backup_path}'")
        try:
            shutil.move(dest_path, backup_path)
            print("Đã sao lưu xong.")
        except OSError as e:
            print(f"Lỗi khi sao lưu thư mục {dest_path}: {e}")
            return False # Báo hiệu thất bại
    
    return True # Báo hiệu thành công

def main():
    print("Bắt đầu quá trình thiết lập model...")

    if not os.path.exists(CONFIG_FILE):
        print(f"Lỗi: Không tìm thấy tệp cấu hình '{CONFIG_FILE}'.")
        return

    try:
        with open(CONFIG_FILE, 'r') as f:
            models_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Lỗi khi đọc tệp YAML: {e}")
        return

    if not models_config:
        print("Tệp cấu hình rỗng hoặc không hợp lệ.")
        return

    os.makedirs("./models", exist_ok=True)

    for model_key, dest_path in DESTINATION_PATHS.items():
        if model_key not in models_config:
            print(f"Không tìm thấy cấu hình cho '{model_key}' trong {CONFIG_FILE}. Bỏ qua.")
            continue

        model_info = models_config[model_key]
        source = model_info.get('source')

        print("-" * 40)
        print(f"Thiết lập model cho: {model_key.upper()}")

        if not handle_backup(dest_path):
            continue # Dừng lại nếu không thể xử lý backup

        try:
            if source == 'huggingface':
                repo_id = model_info.get('repo_id')
                if not repo_id:
                    print(f"Lỗi: 'repo_id' bị thiếu cho nguồn huggingface của '{model_key}'.")
                    continue
                print(f"Đang tải model '{repo_id}' từ Hugging Face về '{dest_path}'...")
                snapshot_download(repo_id=repo_id, local_dir=dest_path, local_dir_use_symlinks=False)
                print(f"Đã tải xong {repo_id}.")

            elif source == 'local':
                local_path = model_info.get('path')
                if not local_path or not os.path.isdir(local_path):
                    print(f"Lỗi: Đường dẫn '{local_path}' không hợp lệ hoặc không tồn tại cho '{model_key}'.")
                    continue
                print(f"Đang sao chép model từ '{local_path}' đến '{dest_path}'...")
                shutil.copytree(local_path, dest_path)
                print(f"Đã sao chép xong.")
            
            else:
                print(f"Lỗi: Nguồn '{source}' không được hỗ trợ cho '{model_key}'.")
                continue

        except Exception as e:
            print(f"ĐÃ XẢY RA LỖI trong quá trình thiết lập '{model_key}': {e}")
            print("Vui lòng kiểm tra lại. Cân nhắc phục hồi từ thư mục '_bak' nếu cần.")
            # Ví dụ phục hồi: shutil.move(dest_path + "_bak", dest_path)
    
    print("-" * 40)
    print("\nQuá trình thiết lập model hoàn tất.")

if __name__ == "__main__":
    main()
