"""Arachne Scholar -- Hardware Probe dinamico (GPU / VRAM / OCR via Ollama).

Modulo condiviso tra:
  - dashboard/app.py   -> endpoint GET /api/system/hardware
  - src/ingest_pdf.py  -> riga [hardware] nei log di ingestione

Rileva GPU attiva, VRAM totale/usata/libera e classifica la macchina in un
"tier" informativo per la HUD:

  TIER 1  VRAM totale > 12 GB
  TIER 2  3 GB <= VRAM <= 12GB
  TIER 3  nessuna GPU o < 3 GB -> percorso classico PyMuPDF consigliato

L'OCR vero NON dipende piu' dal tier ne' da SDK locali: e' una chiamata HTTP
diretta a Ollama (localhost in produzione, http://ollama:11434 in Docker).
La disponibilita' OCR = server Ollama raggiungibile + un modello glm-ocr
presente nei tag. Il probe e' MAI eccezionale: in dubbio, ocr_available=False
e il percorso classico resta sempre disponibile.
"""
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone

TIER_GB_FULL_GPU = 12   # sopra -> TIER 1
TIER_GB_MIN_OCR = 3     # sotto -> TIER 3

TIER_LABELS = {
    1: "TIER 1 - OCR diretto Ollama (GPU >12GB)",
    2: "TIER 2 - OCR diretto Ollama (GPU compatta)",
    3: "TIER 3 - PyMuPDF + sanitizzazione regex",
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OCR_MODEL = "glm-ocr-16k"


def compute_tier(vram_total_mb, gpu_present):
    """Classificazione statica per capacita' totale (informativa per HUD)."""
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


def _load_settings():
    try:
        return json.load(open(SETTINGS_FILE))
    except Exception:
        return {}


def ollama_ocr_probe():
    """Verifica (mai eccezionale) che Ollama sia raggiungibile e abbia un
    modello glm-ocr. Ritorna dict con ocr_available/ollama_url/ocr_model.
    Priorita' configurazione: env OLLAMA_BASE_URL / OLLAMA_OCR_MODEL >
    settings.json (ollama_base_url / ollama_model) > default."""
    settings = _load_settings()
    base_url = (os.environ.get("OLLAMA_BASE_URL")
                or settings.get("ollama_base_url")
                or DEFAULT_OLLAMA_URL).rstrip("/")
    wanted = (os.environ.get("OLLAMA_OCR_MODEL")
              or settings.get("ollama_model") or DEFAULT_OCR_MODEL)
    info = {"ocr_available": False, "ollama_url": base_url, "ocr_model": None}
    try:
        req = urllib.request.Request(base_url + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as r:
            tags = json.loads(r.read().decode("utf-8"))
        names = [m.get("name", "") for m in tags.get("models", [])]
    except Exception:
        return info
    candidates = [wanted]
    if ":" not in wanted:
        candidates.append(wanted + ":latest")
    candidates += ["glm-ocr-16k", "glm-ocr:latest", "glm-ocr"]
    for c in candidates:
        if c in names:
            info["ocr_available"] = True
            info["ocr_model"] = c
            break
    return info


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
        "probe_source": gpu["probe_source"] if gpu else "none",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        info.update(ollama_ocr_probe())
    except Exception:
        info.update({"ocr_available": False, "ollama_url": None,
                     "ocr_model": None})
    return info


if __name__ == "__main__":
    print(json.dumps(probe_hardware(), indent=2, ensure_ascii=False))
