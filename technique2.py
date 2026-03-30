import torch
import numpy as np
import os
import sys
import pefile

sys.path.insert(0, '/home/cyril/malconv-evasion-project/MalConv2')
from MalConvGCT_nocat import MalConvGCT

# ── paths ──────────────────────────────────────────────────────────────
MALWARE_DIR   = '/home/cyril/malconv-evasion-project/datasets/malware/chosen10'
OUTPUT_DIR    = '/home/cyril/malconv-evasion-project/datasets/malware/transformed_technique2'
RESULTS_DIR   = '/home/cyril/malconv-evasion-project/results'
CHECKPOINT    = '/home/cyril/malconv-evasion-project/models/pretrained/malconvGCT_nocat.checkpoint'
OVERLAY_SOURCE = '/home/cyril/malconv-evasion-project/datasets/benign/anyburn_setup_x64.exe'


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


# ── overlay extraction ─────────────────────────────────────────────────
def extract_overlay(pe_path):
    """
    Use pefile to extract the overlay (bytes after the last PE section)
    from the given file. Returns raw bytes.
    """
    pe = pefile.PE(pe_path)
    overlay_offset = pe.get_overlay_data_start_offset()
    if overlay_offset is None:
        raise RuntimeError(f"No overlay found in {pe_path}")
    with open(pe_path, 'rb') as f:
        f.seek(overlay_offset)
        overlay_data = f.read()
    print(f"✅ Extracted overlay from {os.path.basename(pe_path)}: "
          f"{len(overlay_data):,} bytes at offset {overlay_offset}")
    return overlay_data


# ── injection logic ────────────────────────────────────────────────────
def inject_overlay(malware_path, output_path, overlay_data):
    with open(malware_path, 'rb') as f:
        malware_bytes = f.read()
    with open(output_path, 'wb') as f:
        f.write(malware_bytes + overlay_data)


# ── main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device('cpu')

    if not os.path.exists(OVERLAY_SOURCE):
        print(f"❌ Overlay source not found at {OVERLAY_SOURCE}")
        exit()

    print("✅ Loading MalConv model...")
    model = load_model(CHECKPOINT, device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Extract overlay once and reuse for all samples
    overlay_data = extract_overlay(OVERLAY_SOURCE)
    print()

    malware_files = [
        f for f in sorted(os.listdir(MALWARE_DIR))
        if os.path.isfile(os.path.join(MALWARE_DIR, f))
    ]

    results = []

    for filename in malware_files:
        malware_path = os.path.join(MALWARE_DIR, filename)
        out_name = f"{filename}.overlay_injected"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        inject_overlay(malware_path, out_path, overlay_data)
        score = predict_file(model, out_path, device)
        evaded = "✓ EVADED" if score < 0.5 else "✗ still detected"

        print(f"  {filename:<45} →  score: {score:.4f}  {evaded}")
        results.append((filename, score))

    # ── write results file ─────────────────────────────────────────────
    out_file = os.path.join(RESULTS_DIR, 'technique2_overlay_results.txt')

    with open(out_file, 'w') as f:
        f.write("Technique 2: Overlay Injection (source: anyburn_setup_x64.exe)\n")
        f.write(f"Overlay size: {len(overlay_data):,} bytes\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'FILENAME':<45} | {'SCORE':>8} | RESULT\n")
        f.write("-" * 60 + "\n")
        for filename, score in results:
            result = "EVADED" if score < 0.5 else "DETECTED"
            f.write(f"{filename[:45]:<45} | {score:>8.4f} | {result}\n")

    print(f"\n{'='*60}")
    print(f"✅ Transformed samples saved to {OUTPUT_DIR}")
    print(f"✅ Results saved to {out_file}")