# Fattibilità — Pre-processor PDF→Markdown multimodale (GLM-OCR)

Data: 2026-07-27 · Stato: **FASE 1-3 implementate e testate** · Autore: Hermes

## 1. Obiettivo

L'estrazione testuale PyMuPDF produce artefatti (column bleeding, sillabazioni
a fine riga, note a piè di pagina fuse nel corpo) che inquinano il parser SVO
deterministico di spaCy. Si valuta e si predispone l'integrazione di
**GLM-OCR** (zai-org) come pre-processor hardware-aware. **Il parser spaCy/SVO
non è stato toccato**: il lavoro riguarda solo il confine PDF→Markdown.

## 2. Fatti verificati su GLM-OCR (fonti: repo zai-org/GLM-OCR, PyPI glmocr 0.1.5, ollama.com)

| Fatto | Valore | Implicazione |
|---|---|---|
| Modello | ~0.9B (CogViT encoder + GLM-0.5B decoder) | minuscolo per gli standard VLM |
| Pipeline | due stadi: layout **PP-DocLayout-V3** → riconoscimento parallelo | layout e OCR sono separabili: lo switch Tier 2 è nativo |
| VRAM FP16 | ~2-4 GB | entra nella Tesla P4 (8GB) anche con Ollama/gemma3 residente |
| Flag layout CPU | **esiste davvero**: CLI `--layout-device cpu`, API `GlmOcr(layout_device="cpu")` | la specifica Tier 2 dell'utente è confermata upstream |
| Backend self-hosted | vLLM ≥0.19, SGLang ≥0.5.10, **Ollama**, mlx-vlm | vLLM/SGLang richiedono CC ≥ 7.x → **incompatibili con Tesla P4 (sm_61 Pascal)**; Ollama (llama.cpp) supporta sm_61 |
| Ollama | `ollama pull glm-ocr:latest` (ufficiale, tag q8_0 disponibili); consigliato endpoint nativo `/api/generate` con `api_mode: ollama_generate` | rotta consigliata per la P4 |
| `pip install glmocr` (base) | pillow, numpy, requests, pydantic, PyYAML, portalocker, dotenv, tqdm, **pymupdf≥1.24** | coesiste con il venv Arachne (pymupdf 1.28 già presente) |
| extra `[layout]`/`[selfhosted]` | **torch≥2.10, transformers≥5.3, torchvision≥0.25** | ⚠️ CONFLITTO con venv prod (torch 2.5.1+cu121, transformers 4.53.2). Inoltre torch 2.10 abbandona le GPU Pascal |

## 3. Verdetto di coesistenza

**MAI installare `glmocr[layout]`/`[selfhosted]` nel venv Arachne**
(`/tmp/arachne_gpu_venv`): romperebbe i pin torch/transformers dello stack
spaCy e torch 2.10 non supporterebbe comunque la P4 (sm_61).

Architettura raccomandata (Tier 2, la nostra P4):

```
┌──────────────────────── 192.168.1.89 ────────────────────────┐
│ Ollama server (già attivo)  ◄── glm-ocr:latest (~2GB VRAM)   │
│      ▲ /api/generate                    (accanto a gemma3)   │
│ glmocr_venv dedicato (py3.12, torch CPU + glmocr[layout])    │
│      ▲ CLI: glmocr parse file.pdf --layout-device cpu        │
│ arachne venv (INVARIATO) — src/ingest_pdf.py invoca la CLI   │
│      via subprocess (settings.glmocr_bin / GLMOCR_BIN)       │
└───────────────────────────────────────────────────────────────┘
```

- Layout (PP-DocLayout-V3) gira su **CPU** nel venv dedicato: torch CPU non
  ha vincoli di compute capability.
- Il VLM OCR gira su **GPU** dentro Ollama: llama.cpp supporta Pascal.
- `CUDA_VISIBLE_DEVICES=""` (force_cpu) non influenza il backend Ollama
  (processo server separato) — documentato come comportamento voluto.
- Nota VRAM: gemma3 (~3-5GB) + glm-ocr (~2GB) coesistono in 7.5GB, ma sotto
  carico valutare `OLLAMA_KEEP_ALIVE` breve per scaricare gemma3 durante
  l'ingestione.

## 4. Switch dinamico implementato (FASE 2)

`src/ingest_pdf.py` (riscrittura full-block, firma pubblica invariata):

| Tier | Condizione (VRAM totale) | Percorso |
|---|---|---|
| 1 | > 12 GB | GLM-OCR, `--layout-device cuda:0` |
| 2 | 3-12 GB (P4) | GLM-OCR, `--layout-device cpu` |
| 3 | < 3 GB / no GPU | PyMuPDF + `sanitize_markdown()` |

- Se il tier è 1/2 ma il binario glmocr manca/timeout/rc≠0 → **fallback
  automatico per-PDF** al percorso classico (mai bloccare la pipeline).
- Override: `settings.json → ocr_mode` (`auto`|`glm`|`classic`),
  `glmocr_bin`, `glmocr_config`, `glmocr_timeout`; env `ARACHNE_OCR_MODE`,
  `GLMOCR_BIN`, `GLMOCR_CONFIG`.
- Output GLM-OCR: pulizia **light** (niente de-sillabazione: romperebbe le
  tabelle Markdown). Output classico: pulizia **heavy**.

## 5. Sanitizzazione regex Tier 3 (`sanitize_markdown`)

| Regola | Pattern riparato | Falsi positivi residui |
|---|---|---|
| Legature | ﬁ ﬂ ﬀ ﬃ ﬄ ﬅ → fi fl ff ffi ffl st | nessuno noto |
| Soft hyphen | U+00AD rimosso | nessuno |
| De-sillabazione | `informa-\nzione` → `informazione` (solo minuscole) | composti legittimi a fine riga ("well-\nknown" → "wellknown") — accettato |
| Apici [n] | `[12]`, `[1,2]`, `[3-5]` attaccati a parola/punto | citazioni [n] legittime vengono rimosse (voluto: inquinano SVO) |
| Apici numerici | `parola12 La`, `frase.3 Il` | rari identificatori alfanumerici (`alfa1 Beta`) |
| Numeri pagina | righe con solo cifre/num. romani | righe dati isolate a solo numero |
| Header/footer | stessa riga <80 char su ≥30% pagine (min 3) | righe brei ripetute legittime |

**Limite strutturale onesto**: il column bleeding (righe di due colonne
interallacciate) NON è riparabile via regex in modo affidabile — serve il
layout detector (Tier 1/2). Tier 3 mitiga ma non risolve: è un fallback,
non un sostituto.

## 6. FASE 1 — Diagnostica e UI (implementata)

- `src/hardware_probe.py`: probe `nvidia-smi` (subprocess, timeout 5s) →
  fallback `torch.cuda.mem_get_info` → `none`. Mai eccezionale.
- `GET /api/system/hardware`: gpu, VRAM totale/usata/libera, tier, label,
  layout_device, probe_source, glmocr_available, force_cpu. **Mai 5xx**
  (degrada a tier 3 con campo `error`).
- Top-bar `index.html`: pill `#vram-pill` accanto al badge GPU, polling 5s,
  `VRAM x.x / y.y GB · Tn`, tooltip con tier esteso, soglie colore
  amber >75% / red >90%. Nascosta su macchine senza GPU.

## 7. Test eseguiti (reali, non self-report)

- 34/34 assert verdi: soglie tier (24/13/12/8/3/2GB, no-GPU), sanitizer su
  6 classi di artefatti, conversione classica su PDF sintetico reale,
  fallback glm→classico, endpoint 200 + regressione `/api/system-check`.
- `node --check` sul JS estratto: OK; tag `<script>` bilanciati.
- Probe reale su 192.168.1.89: `Tesla P4, 7680MB, tier 2, layout cpu,
  probe nvidia-smi` — esattamente il Tier 2 previsto.

## 8. Deploy del backend GLM-OCR (ESEGUITO 2026-07-27)

Stato reale su 192.168.1.89:

- `ollama pull glm-ocr:latest` → OK (2.2GB, container Docker `ollama`)
- venv `/tmp/glmocr_venv` (py3.12, torch 2.13.0+cpu, glmocr 0.1.5,
  transformers 5.14.1) — creato con `python3 -m venv --without-pip` +
  get-pip.py (host senza python3.12-venv / ensurepip)
- `glmocr_config.yaml` nel repo root prod: backend `ollama_generate` @
  localhost:11434, layout `PaddlePaddle/PP-DocLayoutV3_safetensors`
- `settings.json` prod: `ocr_mode=auto`, `glmocr_bin`, `glmocr_config`,
  `glmocr_timeout=7200`
- Endpoint live: `glmocr_available: true` su http://192.168.1.89:8001

### Pitfall trovati in produzione (reali, non teorici)

1. **Il config custom SOSTITUISCE i default SDK**: senza
   `page_loader.task_prompt_mapping` nel nostro yaml, il prompt OCR era
   vuoto → output = soli marker markdown ("# ", "## ", "$$") senza testo.
   Fix: ridichiarare text/table/formula ("Text Recognition:" ecc.).
2. **ensurepip assente sull'host**: `python3 -m venv` fallisce →
   `--without-pip` + `get-pip.py`.
3. **Timeout OCR**: un paper denso di 15 pag. (tabelle+formule) richiede
   **>30 min** su P4 (ogni regione tabellare genera 2000-3000 token a
   ~75 tok/s, slot Ollama serializzato). Default `glmocr_timeout` alzato
   1800 → **7200s**.
4. **ensurepip/timing a parte**, il primo run E2E (run #4) ha dimostrato:
   slicer anti-OOM OK (22 slice, trf gpu=True, zero Axis mismatch) e
   fallback PyMuPDF funzionante (sanitize: 75 numeri pagina, 252
   header/footer rimossi).

### Misurazioni reali (Tesla P4, glm-ocr via Ollama)

| Metrica | Valore |
|---|---|
| Throughput VLM | ~70-82 tok/s |
| VRAM totale con glm-ocr residente | 3119 MiB / 7680 MiB |
| Tempo OCR paper 15 pag. (denso) | 30-45 min |
| Tempo OCR pagina singola (test diretto) | ~20-30s |
