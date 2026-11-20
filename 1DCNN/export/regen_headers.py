"""Regenerate every <tag>/*.h model header from the .tflite files and verify the
bytes round-trip.  Usage: .venv-export/bin/python 1DCNN/export/regen_headers.py P1 P2"""
import re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "HasanML" / "paper3"))
from src.export import generate_c_header
for tag in sys.argv[1:]:
    for p in sorted((HERE / tag).glob("*.tflite")):
        b = p.read_bytes()
        name = p.stem.lower()
        h = HERE / tag / f"{name}.h"
        generate_c_header(b, name, h)
        body = h.read_text().split("{", 1)[1].split("}", 1)[0]
        got = bytes(int(x, 16) for x in re.findall(r"0x([0-9a-f]{2})", body))
        assert got == b, p
        print("ok", h.name, len(b))
