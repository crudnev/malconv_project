import pefile
import os

benign_dir = '/home/cyril/malconv-evasion-project/datasets/benign'

for filename in sorted(os.listdir(benign_dir)):
    path = os.path.join(benign_dir, filename)
    try:
        pe = pefile.PE(path)
        overlay_offset = pe.get_overlay_data_start_offset()
        if overlay_offset:
            file_size = os.path.getsize(path)
            overlay_size = file_size - overlay_offset
            print(f"{filename:<30} overlay: {overlay_size:,} bytes")
        else:
            print(f"{filename:<30} no overlay")
    except Exception as e:
        print(f"{filename:<30} error: {e}")