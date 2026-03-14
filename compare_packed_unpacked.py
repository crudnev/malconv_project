import torch
import numpy as np
import os
import sys

sys.path.insert(0, '/home/cyril/malconv-evasion-project/MalConv2')
from MalConvGCT_nocat import MalConvGCT


def predict_file(model, file_path, device):
    limit = 2000000
    try:
        with open(file_path, 'rb') as f:
            byte_data = f.read(limit)
            data = np.frombuffer(byte_data, dtype=np.uint8).astype(np.int64) + 1
            if len(data) < limit:
                data = np.pad(data, (0, limit - len(data)), 'constant')

        tensor = torch.from_numpy(data).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            prediction = output[0] if isinstance(output, tuple) else output
            probs = torch.softmax(prediction, dim=1)
            score = probs[0, 1].item()

        return score

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return -1.0


if __name__ == "__main__":
    device = torch.device('cpu')

    model = MalConvGCT(channels=256, window_size=256, stride=64)

    checkpoint_path = '/home/cyril/malconv-evasion-project/models/pretrained/malconvGCT_nocat.checkpoint'

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ Successfully loaded weights from {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found!")
        exit()

    model.to(device)
    model.eval()

    packed_path   = '/home/cyril/malconv-evasion-project/datasets/malware/extracted/3_4.exe'
    unpacked_path = '/home/cyril/malconv-evasion-project/datasets/malware/chosen10/3_4.exe'

    print("\n--- Comparing packed vs unpacked: 3_4.exe ---\n")

    for label, path in [("PACKED  ", packed_path), ("UNPACKED", unpacked_path)]:
        if os.path.exists(path):
            score = predict_file(model, path, device)
            result = "DETECTED" if score >= 0.5 else "MISSED"
            print(f"  {label} | {result:<8} | Score: {score:.4f} | {path}")
        else:
            print(f"  {label} | ❌ File not found: {path}")