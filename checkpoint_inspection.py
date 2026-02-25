import torch

checkpoint = torch.load(
    '/home/cyril/malconv-evasion-project/models/pretrained/malconvGCT_nocat.checkpoint',
    map_location='cpu'
)

if isinstance(checkpoint, dict):
    print("Top-level keys:", list(checkpoint.keys()))
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    for k, v in state_dict.items():
        print(f"{k:50s} {str(v.shape)}")
else:
    print("Not a dict, type:", type(checkpoint))