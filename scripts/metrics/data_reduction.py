import os

def get_folder_size_in_gb(folder_path, end=""):
    total_size_bytes = 0
    
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            if f.endswith(end):
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size_bytes += os.path.getsize(fp)
    
    # Convert bytes to gigabytes (1 GB = 1,073,741,824 bytes)
    total_size_gb = total_size_bytes / (1024 ** 3)
    
    return total_size_gb


# Example usage
h264_path = '/home/agabriel/repos/video-stream-indexing/results/'
h264_folder_size_gb = get_folder_size_in_gb(h264_path, ".h264")


videos_path = '/home/agabriel/repos/video-stream-indexing/videos/'
videos_folder_size_gb = get_folder_size_in_gb(videos_path, ".mp4")

print(f"Retrieved data: {h264_folder_size_gb:.2f} GB")
print(f"Total data: {videos_folder_size_gb:.2f} GB")
print(f"Data reduction: {((videos_folder_size_gb-h264_folder_size_gb)/videos_folder_size_gb)*100:.2f}%")