import torch
import numpy as np
import os
import sys

# Step 1: Add the MalConv2 repo directory to path so we can import their class
# Make sure you've cloned the repo: git clone https://github.com/FutureComputing4AI/MalConv2
sys.path.insert(0, '/home/cyril/malconv-evasion-project/MalConv2')  # adjust path as needed

from MalConvGCT_nocat import MalConvGCT  # use THEIR class, not a custom one

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
                # If still 2 elements, take index 0 (malware score)
                if prediction.numel() > 1:
                    prediction = prediction[0, 0]

        # Output is a single sigmoid score: 1.0 = malware, 0.0 = benign
        return prediction.item()

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return -1.0

if __name__ == "__main__":
    device = torch.device('cpu')

    # Use THEIR exact architecture with the correct hyperparameters from the README
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

    target_dir = '/home/cyril/malconv-evasion-project/datasets/benign/'

    if os.path.exists(target_dir):
        print(f"🔍 Scanning: {target_dir}\n")
        print(f"{'RESULT':<15} | {'FILENAME':<30} | {'SCORE'}")
        print("-" * 60)

        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            if os.path.isfile(file_path):
                score = predict_file(model, file_path, device)
                result = "BENIGN" if score < 0.5 else "FALSE POSITIVE"
                print(f"{result:<15} | {filename[:30]:<30} | {score:.4f}")