import torch
import torch.nn as nn
import numpy as np
import os

# Define the MalConv Architecture (Optimized for Inference)
class MalConv(nn.Module):
    def __init__(self, input_length=2000000, window_size=512):
        super(MalConv, self).__init__()
        self.embed = nn.Embedding(257, 8)
        self.conv1 = nn.Conv1d(8, 128, kernel_size=window_size, stride=window_size, bias=True)
        self.conv2 = nn.Conv1d(8, 128, kernel_size=window_size, stride=window_size, bias=True)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        cnn_value = self.conv1(x) * torch.sigmoid(self.conv2(x))
        x = self.pooling(cnn_value).squeeze(-1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return self.sig(x)

def predict_file(model, file_path):
    # Standard MalConv processes the first 2MB
    limit = 2000000 
    with open(file_path, 'rb') as f:
        byte_data = f.read(limit)
        # Convert bytes to long integers and add 1 (to handle 0-255 + padding)
        data = np.frombuffer(byte_data, dtype=np.uint8).astype(np.int64) + 1
        if len(data) < limit:
            data = np.pad(data, (0, limit - len(data)), 'constant')
        
    tensor = torch.from_numpy(data).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor)
    return prediction.item()

if __name__ == "__main__":
    # 1. Setup device
    device = torch.device('cpu')
    
    # 2. Initialize Model
    model = MalConv().to(device)
    
    # 3. Load Weights (Ensure you download the checkpoint to your VM later)
    checkpoint_path = 'malconvGCT_nocat.checkpoint'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # MalConv2 checkpoints often nest the weights in 'model_state_dict'
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded weights from {checkpoint_path}")
    else:
        print(f"Warning: {checkpoint_path} not found. Running with random weights!")

    model.eval()

    # 4. Test on a directory (Change this path to match your VM folders)
    target_dir = './datasets/malware/extracted/'
    
    if os.path.exists(target_dir):
        print(f"Scanning directory: {target_dir}")
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            if os.path.isfile(file_path):
                score = predict_file(model, file_path)
                result = "MALICIOUS" if score > 0.5 else "BENIGN"
                print(f"[{result}] {filename} - Score: {score:.4f}")
    else:
        print(f"Directory {target_dir} not found. Check your paths!")