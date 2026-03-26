import torch
import numpy as np
import os
import sys

sys.path.insert(0, '/home/cyril/malconv-evasion-project/MalConv2')
from MalConvGCT_nocat import MalConvGCT

# ── paths ──────────────────────────────────────────────────────────────
MALWARE_DIR   = '/home/cyril/malconv-evasion-project/datasets/malware/chosen10'
OUTPUT_DIR    = '/home/cyril/malconv-evasion-project/datasets/malware/transformed_technique1v2'
RESULTS_DIR   = '/home/cyril/malconv-evasion-project/results'
CHECKPOINT    = '/home/cyril/malconv-evasion-project/models/pretrained/malconvGCT_nocat.checkpoint'
BENIGN_SOURCE = '/home/cyril/malconv-evasion-project/datasets/benign/notepad++.exe'

# ── fixed settings ─────────────────────────────────────────────────────
APPEND_SIZE   = 1_000_000   # 1000 KB
FIXED_OFFSET  = 1024        # always sample from this offset in notepad++.exe


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
def get_notepadpp_bytes():
    """
    Always read APPEND_SIZE bytes from FIXED_OFFSET in notepad++.exe
    so results are fully reproducible across runs.
    """
    file_size = os.path.getsize(BENIGN_SOURCE)
    available = file_size - FIXED_OFFSET

    with open(BENIGN_SOURCE, 'rb') as f:
        f.seek(FIXED_OFFSET)
        chunk = f.read(min(APPEND_SIZE, available))

    # If notepad++.exe is smaller than needed, tile the chunk
    if len(chunk) < APPEND_SIZE:
        chunk = (chunk * (APPEND_SIZE // len(chunk) + 1))[:APPEND_SIZE]

    return chunk


def append_bytes(malware_path, output_path, benign_chunk):
    with open(malware_path, 'rb') as f:
        malware_bytes = f.read()
    with open(output_path, 'wb') as f:
        f.write(malware_bytes + benign_chunk)


# ── main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device('cpu')

    if not os.path.exists(BENIGN_SOURCE):
        print(f"❌ Notepad++ not found at {BENIGN_SOURCE}")
        exit()

    print("✅ Loading MalConv model...")
    model = load_model(CHECKPOINT, device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load the benign chunk once — same bytes used for every sample
    benign_chunk = get_notepadpp_bytes()
    print(f"✅ Loaded {len(benign_chunk):,} bytes from notepad++.exe at offset {FIXED_OFFSET}\n")

    malware_files = [
        f for f in sorted(os.listdir(MALWARE_DIR))
        if os.path.isfile(os.path.join(MALWARE_DIR, f))
    ]

    results = []   # (filename, score)

    for filename in malware_files:
        malware_path = os.path.join(MALWARE_DIR, filename)
        out_name = f"{filename}.appended_1000KB"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        append_bytes(malware_path, out_path, benign_chunk)
        score = predict_file(model, out_path, device)
        evaded = "✓ EVADED" if score < 0.5 else "✗ still detected"

        print(f"  {filename:<45} →  score: {score:.4f}  {evaded}")
        results.append((filename, score))

    # ── write results file ─────────────────────────────────────────────
    out_file = os.path.join(RESULTS_DIR, 'technique1v2_append_results.txt')

    with open(out_file, 'w') as f:
        f.write("Technique 1v2: Benign Byte Appending (source: notepad++.exe)\n")
        f.write(f"Append size: {APPEND_SIZE:,} bytes | Fixed offset: {FIXED_OFFSET}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'FILENAME':<45} | {'SCORE':>8} | RESULT\n")
        f.write("-" * 60 + "\n")
        for filename, score in results:
            result = "EVADED" if score < 0.5 else "DETECTED"
            f.write(f"{filename[:45]:<45} | {score:>8.4f} | {result}\n")

    print(f"\n{'='*60}")
    print(f"✅ Transformed samples saved to {OUTPUT_DIR}")
    print(f"✅ Results saved to {out_file}")