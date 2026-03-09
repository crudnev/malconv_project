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
            score = probs[0, 1].item()  # index 1 = malware probability

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

    target_dir = '/home/cyril/malconv-evasion-project/datasets/malware/extracted'

    if os.path.exists(target_dir):
        print(f"🔍 Scanning malware directory: {target_dir}\n")
        print(f"{'RESULT':<15} | {'FILENAME':<30} | {'SCORE'}")
        print("-" * 60)

        total = 0
        detected = 0

        for filename in sorted(os.listdir(target_dir)):
            file_path = os.path.join(target_dir, filename)
            if os.path.isfile(file_path):
                score = predict_file(model, file_path, device)
                total += 1
                if score >= 0.5:
                    result = "DETECTED ✓"
                    detected += 1
                else:
                    result = "MISSED ✗"
                print(f"{result:<15} | {filename[:30]:<30} | {score:.4f}")

        print("-" * 60)
        if total > 0:
            print(f"📊 Detected {detected}/{total} ({(detected/total)*100:.1f}% detection rate)")
    else:
        print(f"❌ Directory not found: {target_dir}")