import torch
import numpy as np
import os
import sys
import pefile
import struct
import random

sys.path.insert(0, '/home/cyril/malconv-evasion-project/MalConv2')
from MalConvGCT_nocat import MalConvGCT

# ── paths ──────────────────────────────────────────────────────────────
MALWARE_DIR  = '/home/cyril/malconv-evasion-project/datasets/malware/chosen10'
OUTPUT_DIR   = '/home/cyril/malconv-evasion-project/datasets/malware/transformed_technique3'
RESULTS_DIR  = '/home/cyril/malconv-evasion-project/results'
CHECKPOINT   = '/home/cyril/malconv-evasion-project/models/pretrained/malconvGCT_nocat.checkpoint'

# Fixed seed for reproducibility
RANDOM_SEED  = 42


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


# ── header manipulation ────────────────────────────────────────────────
def manipulate_pe_header(malware_path, output_path, rng):
    """
    Modify safe PE header fields that Windows ignores at runtime:

    DOS Header:
      - e_oemid, e_oeminfo       : reserved, always ignored
      - e_res  (4 x uint16)      : reserved array, always zero normally
      - e_res2 (10 x uint16)     : reserved array, always zero normally

    COFF File Header:
      - TimeDateStamp            : compile time, not checked at runtime
      - PointerToSymbolTable     : deprecated, always 0 in modern PE
      - NumberOfSymbols          : deprecated, always 0 in modern PE

    Optional Header:
      - MajorLinkerVersion       : linker metadata, not used at runtime
      - MinorLinkerVersion       : linker metadata, not used at runtime
      - MajorImageVersion        : app-defined, never checked
      - MinorImageVersion        : app-defined, never checked
      - CheckSum                 : only verified for drivers/system DLLs
    """
    with open(malware_path, 'rb') as f:
        data = bytearray(f.read())

    try:
        pe = pefile.PE(data=bytes(data))
    except pefile.PEFormatError as e:
        print(f"  ⚠ pefile parse error for {os.path.basename(malware_path)}: {e}")
        return False

    # ── DOS Header fields ──────────────────────────────────────────────
    # e_oemid at offset 0x24 (2 bytes)
    struct.pack_into('<H', data, 0x24, rng.randint(0, 0xFFFF))
    # e_oeminfo at offset 0x26 (2 bytes)
    struct.pack_into('<H', data, 0x26, rng.randint(0, 0xFFFF))
    # e_res at offset 0x28: 4 x uint16 (8 bytes)
    for i in range(4):
        struct.pack_into('<H', data, 0x28 + i * 2, rng.randint(0, 0xFFFF))
    # e_res2 at offset 0x34: 10 x uint16 (20 bytes)
    for i in range(10):
        struct.pack_into('<H', data, 0x34 + i * 2, rng.randint(0, 0xFFFF))

    # ── COFF File Header fields ────────────────────────────────────────
    # PE signature is at offset pe.DOS_HEADER.e_lfanew
    # COFF header starts 4 bytes after (after "PE\0\0" signature)
    coff_offset = pe.DOS_HEADER.e_lfanew + 4

    # TimeDateStamp at COFF offset +4 (4 bytes)
    struct.pack_into('<I', data, coff_offset + 4, rng.randint(0, 0xFFFFFFFF))
    # PointerToSymbolTable at COFF offset +8 (4 bytes)
    struct.pack_into('<I', data, coff_offset + 8, 0)
    # NumberOfSymbols at COFF offset +12 (4 bytes)
    struct.pack_into('<I', data, coff_offset + 12, 0)

    # ── Optional Header fields ─────────────────────────────────────────
    # Optional header starts at COFF offset +20
    opt_offset = coff_offset + 20

    # MajorLinkerVersion at opt offset +2 (1 byte)
    data[opt_offset + 2] = rng.randint(0, 0xFF)
    # MinorLinkerVersion at opt offset +3 (1 byte)
    data[opt_offset + 3] = rng.randint(0, 0xFF)
    # MajorImageVersion at opt offset +48 (2 bytes)
    struct.pack_into('<H', data, opt_offset + 48, rng.randint(0, 0xFFFF))
    # MinorImageVersion at opt offset +50 (2 bytes)
    struct.pack_into('<H', data, opt_offset + 50, rng.randint(0, 0xFFFF))
    # CheckSum at opt offset +64 (4 bytes)
    struct.pack_into('<I', data, opt_offset + 64, rng.randint(0, 0xFFFFFFFF))

    with open(output_path, 'wb') as f:
        f.write(data)

    return True


# ── main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device('cpu')
    rng = random.Random(RANDOM_SEED)

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
        out_name = f"{filename}.header_mangled"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        success = manipulate_pe_header(malware_path, out_path, rng)

        if success:
            score = predict_file(model, out_path, device)
            evaded = "✓ EVADED" if score < 0.5 else "✗ still detected"
            print(f"  {filename:<45} →  score: {score:.4f}  {evaded}")
            results.append((filename, score, "OK"))
        else:
            print(f"  {filename:<45} →  SKIPPED (parse error)")
            results.append((filename, -1.0, "PARSE ERROR"))

    # ── write results file ─────────────────────────────────────────────
    out_file = os.path.join(RESULTS_DIR, 'technique3_header_results.txt')

    with open(out_file, 'w') as f:
        f.write("Technique 3: PE Header Manipulation\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write("Fields modified: e_oemid, e_oeminfo, e_res, e_res2, TimeDateStamp,\n")
        f.write("                 PointerToSymbolTable, NumberOfSymbols,\n")
        f.write("                 MajorLinkerVersion, MinorLinkerVersion,\n")
        f.write("                 MajorImageVersion, MinorImageVersion, CheckSum\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'FILENAME':<45} | {'SCORE':>8} | RESULT\n")
        f.write("-" * 65 + "\n")
        for filename, score, status in results:
            if status == "OK":
                result = "EVADED" if score < 0.5 else "DETECTED"
                f.write(f"{filename[:45]:<45} | {score:>8.4f} | {result}\n")
            else:
                f.write(f"{filename[:45]:<45} | {'N/A':>8} | {status}\n")

    print(f"\n{'='*60}")
    print(f"✅ Transformed samples saved to {OUTPUT_DIR}")
    print(f"✅ Results saved to {out_file}")