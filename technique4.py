import torch
import numpy as np
import os
import sys
import pefile
import struct

sys.path.insert(0, '/home/cyril/malconv-evasion-project/MalConv2')
from MalConvGCT_nocat import MalConvGCT

# ── paths ──────────────────────────────────────────────────────────────
MALWARE_DIR  = '/home/cyril/malconv-evasion-project/datasets/malware/chosen10'
OUTPUT_DIR   = '/home/cyril/malconv-evasion-project/datasets/malware/transformed_technique4'
RESULTS_DIR  = '/home/cyril/malconv-evasion-project/results'
CHECKPOINT   = '/home/cyril/malconv-evasion-project/models/pretrained/malconvGCT_nocat.checkpoint'

# ── benign-looking section names to replace with ───────────────────────
# These are common section names seen in legitimate Windows software.
# Each original name gets mapped to a benign-looking alternative.
# Section names are exactly 8 bytes in the PE format — padded with nulls.
BENIGN_SECTION_NAMES = [
    '.text',    # code section
    '.data',    # initialized data
    '.rdata',   # read-only data
    '.rsrc',    # resources
    '.reloc',   # relocations
    '.pdata',   # exception data (common in x64)
    '.tls',     # thread local storage
    '.idata',   # import table
]


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


# ── section renaming logic ─────────────────────────────────────────────
def rename_sections(malware_path, output_path):
    """
    Rename all PE sections to benign-looking names commonly seen in
    legitimate Windows software. Section names are 8 bytes in the PE
    section table — we overwrite them directly in the raw binary.
    Windows does not use section names at load time so this is safe.
    """
    with open(malware_path, 'rb') as f:
        data = bytearray(f.read())

    try:
        pe = pefile.PE(data=bytes(data))
    except pefile.PEFormatError as e:
        print(f"  ⚠ pefile parse error: {e}")
        return False, []

    if not pe.sections:
        print(f"  ⚠ No sections found")
        return False, []

    rename_log = []

    for i, section in enumerate(pe.sections):
        old_name = section.Name.rstrip(b'\x00').decode('utf-8', errors='replace')

        # Pick a benign name — cycle through the list if more sections than names
        new_name_str = BENIGN_SECTION_NAMES[i % len(BENIGN_SECTION_NAMES)]

        # Section names are exactly 8 bytes, null-padded
        new_name_bytes = new_name_str.encode('utf-8').ljust(8, b'\x00')[:8]

        # Write the new name directly into the raw binary at the section's offset
        name_offset = section.get_file_offset()
        data[name_offset:name_offset + 8] = new_name_bytes

        rename_log.append((old_name, new_name_str))

    with open(output_path, 'wb') as f:
        f.write(data)

    return True, rename_log


# ── main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device('cpu')

    print("✅ Loading MalConv model...")
    model = load_model(CHECKPOINT, device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    malware_files = [
        f for f in sorted(os.listdir(MALWARE_DIR))
        if os.path.isfile(os.path.join(MALWARE_DIR, f))
    ]

    results = []

    print(f"🔍 Processing {len(malware_files)} samples...\n")

    for filename in malware_files:
        malware_path = os.path.join(MALWARE_DIR, filename)
        out_name = f"{filename}.sections_renamed"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        success, rename_log = rename_sections(malware_path, out_path)

        if success:
            score = predict_file(model, out_path, device)
            evaded = "✓ EVADED" if score < 0.5 else "✗ still detected"
            renames = ", ".join(f"{o}→{n}" for o, n in rename_log)
            print(f"  {filename:<45} →  score: {score:.4f}  {evaded}")
            print(f"    Renamed: {renames}")
            results.append((filename, score, renames, "OK"))
        else:
            print(f"  {filename:<45} →  SKIPPED (parse error)")
            results.append((filename, -1.0, "", "PARSE ERROR"))

    # ── write results file ─────────────────────────────────────────────
    out_file = os.path.join(RESULTS_DIR, 'technique4_section_rename_results.txt')

    with open(out_file, 'w') as f:
        f.write("Technique 4: PE Section Renaming\n")
        f.write("Replacement names: " + ", ".join(BENIGN_SECTION_NAMES) + "\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'FILENAME':<45} | {'SCORE':>8} | RESULT\n")
        f.write("-" * 65 + "\n")
        for filename, score, renames, status in results:
            if status == "OK":
                result = "EVADED" if score < 0.5 else "DETECTED"
                f.write(f"{filename[:45]:<45} | {score:>8.4f} | {result}\n")
                f.write(f"  Renamed: {renames}\n")
            else:
                f.write(f"{filename[:45]:<45} | {'N/A':>8} | {status}\n")

    print(f"\n{'='*60}")
    print(f"✅ Transformed samples saved to {OUTPUT_DIR}")
    print(f"✅ Results saved to {out_file}")