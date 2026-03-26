import torch
import numpy as np
import os
import sys
import random

sys.path.insert(0, '/home/cyril/malconv-evasion-project/MalConv2')
from MalConvGCT_nocat import MalConvGCT

# ── paths ──────────────────────────────────────────────────────────────
MALWARE_DIR  = '/home/cyril/malconv-evasion-project/datasets/malware/chosen10'
OUTPUT_DIR   = '/home/cyril/malconv-evasion-project/datasets/malware/transformed_technique1'
RESULTS_DIR  = '/home/cyril/malconv-evasion-project/results'
CHECKPOINT   = '/home/cyril/malconv-evasion-project/models/pretrained/malconvGCT_nocat.checkpoint'
BENIGN_SOURCE = '/home/cyril/malconv-evasion-project/datasets/benign/notepad++.exe'

# ── append sizes to test ───────────────────────────────────────────────
APPEND_SIZES = [100_000, 500_000, 1_000_000]   # 100 KB, 500 KB, 1 MB


# ── MalConv helpers ────────────────────────────────────────────────────
def load_model(checkpoint_path, device):
    model = MalConvGCT(channels=256, window_size=256, stride=64)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    return model


def predict_file(model, file_path, device):
    limit = 2_000_000
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
            return probs[0, 1].item()
    except Exception as e:
        print(f"  ⚠ Score error for {file_path}: {e}")
        return -1.0


# ── appending logic ────────────────────────────────────────────────────
def get_notepadpp_bytes(num_bytes):
    """
    Sample num_bytes from a random offset inside notepad++.exe,
    skipping the first 1024 bytes to avoid the PE header.
    """
    size = os.path.getsize(BENIGN_SOURCE)
    if size < 1024 + num_bytes:
        # File smaller than requested: read what we can and tile it
        with open(BENIGN_SOURCE, 'rb') as f:
            f.seek(1024)
            chunk = f.read()
        tiled = (chunk * (num_bytes // len(chunk) + 1))[:num_bytes]
        return tiled

    max_offset = size - num_bytes
    offset = random.randint(1024, max_offset)
    with open(BENIGN_SOURCE, 'rb') as f:
        f.seek(offset)
        return f.read(num_bytes)


def append_bytes(malware_path, output_path, num_bytes):
    with open(malware_path, 'rb') as f:
        malware_bytes = f.read()
    benign_chunk = get_notepadpp_bytes(num_bytes)
    with open(output_path, 'wb') as f:
        f.write(malware_bytes + benign_chunk)


# ── main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device('cpu')

    if not os.path.exists(BENIGN_SOURCE):
        print(f"❌ Notepad++ not found at {BENIGN_SOURCE}")
        print("   Check the filename is exactly 'notepad++.exe' in your benign directory.")
        exit()

    print("✅ Loading MalConv model...")
    model = load_model(CHECKPOINT, device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    malware_files = [
        f for f in sorted(os.listdir(MALWARE_DIR))
        if os.path.isfile(os.path.join(MALWARE_DIR, f))
    ]

    size_labels = [f"{s // 1000}KB" for s in APPEND_SIZES]
    results = []   # (filename, {label: score})

    for filename in malware_files:
        malware_path = os.path.join(MALWARE_DIR, filename)
        print(f"\n{'─'*60}")
        print(f"  Sample: {filename}")

        size_scores = {}

        for num_bytes, label in zip(APPEND_SIZES, size_labels):
            out_name = f"{filename}.appended_{label}"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            append_bytes(malware_path, out_path, num_bytes)
            score = predict_file(model, out_path, device)
            evaded = "✓ EVADED" if score < 0.5 else "✗ still detected"
            print(f"  +{label:>7} appended  →  score: {score:.4f}  {evaded}")

            size_scores[label] = score

        results.append((filename, size_scores))

    # ── write results file ─────────────────────────────────────────────
    out_file = os.path.join(RESULTS_DIR, 'technique1_append_results.txt')

    with open(out_file, 'w') as f:
        f.write("Technique 1: Benign Byte Appending (source: notepad++.exe)\n")
        f.write("=" * 75 + "\n")
        header = f"{'FILENAME':<40} | " + " | ".join(f"{l:>8}" for l in size_labels)
        f.write(header + "\n")
        f.write("-" * 75 + "\n")
        for filename, size_scores in results:
            row = f"{filename[:40]:<40} | " + \
                  " | ".join(f"{size_scores.get(l, -1):>8.4f}" for l in size_labels)
            f.write(row + "\n")

    print(f"\n{'='*60}")
    print(f"✅ Transformed samples saved to {OUTPUT_DIR}")
    print(f"✅ Results saved to {out_file}")