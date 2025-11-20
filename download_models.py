import os
from huggingface_hub import snapshot_download

# Định nghĩa các model cần tải và đường dẫn lưu trữ cục bộ
MODELS_TO_DOWNLOAD = {
    "Qwen/Qwen2.5-0.5B": "./models/Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen3Guard-Gen-0.6B": "./models/Qwen/Qwen3Guard-Gen-0.6B",
}

os.makedirs("./models", exist_ok=True)

print("Bắt đầu tải các model...")

for repo_id, local_dir in MODELS_TO_DOWNLOAD.items():
    print(f"\nĐang tải model: {repo_id} về {local_dir}...")
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
        print(f"Đã tải xong {repo_id}.")
    except Exception as e:
        print(f"Lỗi khi tải {repo_id}: {e}")

print("\nQuá trình tải model hoàn tất.")
