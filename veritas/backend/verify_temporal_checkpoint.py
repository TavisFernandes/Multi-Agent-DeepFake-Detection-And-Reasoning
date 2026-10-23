"""
verify_temporal_checkpoint.py
Quick one-shot inspection of the temporal_agent_checkpoint.pth.
Run with: C:\Python314\python.exe -X utf8 verify_temporal_checkpoint.py
"""
import sys, io
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

CKPT = Path(__file__).resolve().parent / "models" / "temporal_agent_checkpoint.pth"

print("=" * 60)
print("TEMPORAL CHECKPOINT VERIFICATION")
print("=" * 60)
print(f"Path   : {CKPT}")
print(f"Exists : {CKPT.exists()}")
if not CKPT.exists():
    print("ABORT: file not found"); sys.exit(1)

print(f"Size   : {CKPT.stat().st_size / 1e6:.1f} MB")

# Load
ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
print(f"\nTop-level keys: {list(ckpt.keys())}")
for k in ("epoch", "model_name", "num_classes", "class_names", "num_frames",
          "frame_size", "metrics"):
    print(f"  {k}: {ckpt.get(k, '<missing>')}")

threshold = ckpt.get("threshold", ckpt.get("best_threshold", None))
print(f"  threshold: {threshold}")

# Verify state_dict key
if "model_state_dict" not in ckpt:
    print("\nFAIL: 'model_state_dict' key missing")
    print("Available:", list(ckpt.keys()))
    sys.exit(1)
print("\n[PASS] 'model_state_dict' key present")

# Build model and load
model = r3d_18(weights=None)
model.fc = nn.Linear(512, 1)
result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
print(f"[PASS] load_state_dict strict=True  missing={result.missing_keys}  unexpected={result.unexpected_keys}")

# Classifier check
print(f"\nmodel.fc = {model.fc}")
assert model.fc.in_features == 512 and model.fc.out_features == 1, "FC mismatch"
print("[PASS] fc = Linear(512, 1)")

# Dummy inference
model.eval()
dummy = torch.zeros(1, 3, 16, 112, 112)
with torch.no_grad():
    logit = model(dummy)
    prob  = torch.sigmoid(logit.squeeze()).item()
print(f"\nDummy inference (zeros):  logit={logit.item():.4f}  prob={prob:.4f}")
assert 0.0 <= prob <= 1.0
print("[PASS] output is valid probability")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print("=" * 60)
