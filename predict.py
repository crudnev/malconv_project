import torch
import torch.nn as nn
import numpy as np
import os

# Define the MalConv Architecture (Optimized for Inference)
class MalConv(nn.Module):
    def __init__(self, input_length=2000000, window_size=512):
        super(MalConv, self).__init__()
        self.embed = nn.Embedding(257, 8)
        self.conv1 = nn.Conv1d(8, 256, kernel_size=window_size, stride=window_size, bias=True)
        self.conv2 = nn.Conv1d(8, 256, kernel_size=window_size, stride=window_size, bias=True)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc_1 = nn.Linear(256, 256)
        self.fc_2 = nn.Linear(256, 2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        cnn_value = self.conv1(x) * torch.sigmoid(self.conv2(x))
        x = self.pooling(cnn_value).squeeze(-1)

        x = x.view(x.size(0), -1)

        x = self.fc_1(x)
        x = self.fc_2(x)
        return self.sig(x)

def predict_file(model, file_path):
    limit = 2000000 
    try:
        with open(file_path, 'rb') as f:
            byte_data = f.read(limit)
            data = np.frombuffer(byte_data, dtype=np.uint8).astype(np.int64) + 1
            if len(data) < limit:
                data = np.pad(data, (0, limit - len(data)), 'constant')
            
        tensor = torch.from_numpy(data).unsqueeze(0)
        with torch.no_grad():
            prediction = model(tensor)
        return prediction.item()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0.0

if __name__ == "__main__":
    device = torch.device('cpu')
    model = MalConv().to(device)
    
    # --- UPDATED PATHS ---
    checkpoint_path = '/home/cyril/malconv-evasion-project/models/pretrained/malconvGCT_nocat.checkpoint'
    target_dir = '/home/cyril/malconv-evasion-project/datasets/benign/'
    
    # --- IMPROVED MALCONV2 LOADING LOGIC ---
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # This checks if it's a MalConv2 dictionary or a raw weight file
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # Remove keys that might conflict with the standard architecture
            state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
        print(f"✅ Successfully loaded weights from {checkpoint_path}")
    else:
        print(f"❌ Warning: {checkpoint_path} not found. Running with random weights!")

    model.eval()

    if os.path.exists(target_dir):
        print(f"🔍 Scanning benign directory: {target_dir}\n")
        print(f"{'RESULT':<12} | {'FILENAME':<30} | {'SCORE'}")
        print("-" * 60)
        
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            if os.path.isfile(file_path):
                score = predict_file(model, file_path)
                # Since we are scanning BENIGN files, we hope for "BENIGN" results
                result = "BENIGN" if score < 0.5 else "FALSE POSITIVE"
                print(f"{result:<12} | {filename[:30]:<30} | {score:.4f}")
    else:
        print(f"Directory {target_dir} not found. Check your paths!")