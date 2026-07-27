"""Arachne Scholar -- Hardware Probe dinamico (GPU / VRAM / OCR tier).

Modulo condiviso tra:
  - dashboard/app.py   -> endpoint GET /api/system/hardware
  - src/ingest_pdf.py  -> switch dinamico del pre-processor PDF->Markdown

Rileva GPU attiva, VRAM totale/usata/libera e classifica la macchina in un
"tier" OCR hardware-aware (soglie da specifica progettuale):

  TIER 1  VRAM totale > 12 GB  -> GLM-OCR completo su GPU (layout + OCR)
  TIER 2  3 GB <= VRAM <= 12GB -> GLM-OCR ibrido: layout su CPU, OCR su GPU
  TIER 3  nessuna GPU o < 3 GB -> fallback PyMuPDF + sanitizzazione regex

Nessuna eccezione esce da probe_hardware(): in caso di dubbio si degrada
sempre a TIER 3 (il percorso classico deve restare sempre disponibile).
"""
import os
import shutil
import subprocess
from datetime import datetime, timezone

TIER_GB_FULL_GPU = 12   # sopra -> TIER 1
TIER_GB_MIN_OCR = 3     # sotto -> TIER 3

TIER_LABELS = {
    1: "TIER 1 - GLM-OCR full-GPU (layout + OCR su GPU)",
    2: "TIER 2 - GLM-OCR ibrido (layout CPU, OCR GPU)",
    3: "TIER 3 - PyMuPDF + sanitizzazione regex",
}

TIER_LAYOUT_DEVICE = {1: "cuda:0", 2: "cpu", 3: None}


def compute_tier(vram_total_mb, gpu_present):
    """Classificazione statica per capacita' totale (specifica utente).
    La VRAM libera istantanea e' esposta a parte per l'OOM-safety a runtime."""
    if not gpu_present or not vram_total_mb:
        return 3
    if vram_total_mb > TIER_GB_FULL_GPU * 1024:
        return 1
    if vram_total_mb >= TIER_GB_MIN_OCR * 1024:
        return 2
    return 3


def _probe_nvidia_smi():
    """Fonte primaria: nvidia-smi via subprocess (non richiede torch)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        line = out.stdout.strip().splitlines()[0]
        name, total, used, free = [p.strip() for p in line.split(",")[:4]]
        return {
            "gpu_name": name,
            "vram_total_mb": int(float(total)),
            "vram_used_mb": int(float(used)),
            "vram_free_mb": int(float(free)),
            "probe_source": "nvidia-smi",
        }
    except Exception:
        return None


def _probe_torch():
    """Fallback: torch.cuda (mem_get_info -> free/total reali)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info(0)
        total_mb, free_mb = int(total_b // 1048576), int(free_b // 1048576)
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_mb": total_mb,
            "vram_used_mb": total_mb - free_mb,
            "vram_free_mb": free_mb,
            "probe_source": "torch",
        }
    except Exception:
        return None


def glmocr_available():
    """True se l'SDK GLM-OCR e' raggiungibile: env GLMOCR_BIN, settings.json
    (glmocr_bin), binario nel PATH o modulo importabile. Il binario puo'
    vivere in un venv dedicato (pattern produzione Tier 2)."""
    if os.environ.get("GLMOCR_BIN") and os.path.exists(os.environ["GLMOCR_BIN"]):
        return True
    try:
        import json as _json
        settings_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "settings.json")
        bin_from_settings = _json.load(open(settings_path)).get("glmocr_bin")
        if bin_from_settings and os.path.exists(bin_from_settings):
            return True
    except Exception:
        pass
    if shutil.which("glmocr"):
        return True
    try:
        import importlib.util
        return importlib.util.find_spec("glmocr") is not None
    except Exception:
        return False


def probe_hardware():
    """Ritorna un dict autosufficiente e MAI eccezionale."""
    gpu = _probe_nvidia_smi() or _probe_torch()
    present = gpu is not None
    total = gpu["vram_total_mb"] if gpu else 0
    tier = compute_tier(total, present)
    info = {
        "gpu_present": present,
        "gpu_name": gpu["gpu_name"] if gpu else None,
        "vram_total_mb": total if gpu else None,
        "vram_used_mb": gpu["vram_used_mb"] if gpu else None,
        "vram_free_mb": gpu["vram_free_mb"] if gpu else None,
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "layout_device": TIER_LAYOUT_DEVICE[tier],
        "probe_source": gpu["probe_source"] if gpu else "none",
        "glmocr_available": glmocr_available(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(probe_hardware(), indent=2, ensure_ascii=False))
