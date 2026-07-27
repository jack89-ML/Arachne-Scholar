# Arachne Scholar 🕷️

Local-first Knowledge Graph Engine for Academic Literature.
PDF accademici → **OCR multimodale via Ollama (Zero-SDK)** → Markdown → grafo SVO
(spaCy) → metriche SNA (betweenness, Louvain, constraint di Burt) → HUD WebGL.

> Stato: **v0.2.0 — architettura "Zero SDK"** (2026-07-27).
> Validata in produzione su Tesla P4: 15/15 pagine OCR senza fallback,
> ~13 min per paper, grafo al 95%+ del benchmark testo-nativo
> (462 nodi / 1601 archi su "Attention Is All You Need").

---

## 1. Architettura a 2 container

Nessun SDK esterno, nessun layout detector, nessun venv dedicato all'OCR.
Il backend parla con Ollama via **HTTP diretto** (urllib, stdlib Python).

```
┌─────────────────────────────┐      HTTP (DNS interno)      ┌──────────────────────────┐
│  backend (arachne-backend)  │  ──────────────────────────► │  ollama (arachne-ollama) │
│  python:3.11-slim           │   POST /api/generate         │  ollama/ollama:latest    │
│  FastAPI + HUD WebGL        │   prompt "Text Recognition:" │  modello: glm-ocr        │
│  spaCy SVO (EN/IT/ES)       │   immagine pagina (base64)   │  GPU passthrough CUDA    │
│  PyMuPDF raster in-memory   │ ◄──────────────────────────  │  (RTX / Tesla / A2000)   │
│  CPU (o GPU opzionale)      │   markdown della pagina      │                          │
└─────────────────────────────┘                              └──────────────────────────┘
```

Flusso ingestione: PyMuPDF rasterizza ogni pagina in PNG **in memoria**
(200 dpi, niente poppler) → una chiamata `POST /api/generate` per pagina con il
prompt ufficiale `Text Recognition:` → markdown concatenato → `data/converted_md/`.
Fallback a due livelli, mai bloccante: pagina fallita → testo nativo di quella
pagina; Ollama irraggiungibile → intero PDF al percorso classico PyMuPDF.

---

## 2. Quickstart Docker (Linux + Windows 11 / WSL2)

Prerequisiti: Docker con **NVIDIA Container Toolkit** (Linux) oppure
**Docker Desktop + WSL2** (Windows 11, la GPU passa via WSL2).

```bash
git clone https://github.com/jack89-ML/Arachne-Scholar.git
cd Arachne-Scholar

# 1) Avvia solo Ollama e scarica il modello OCR (una tantum, ~2 GB)
docker compose up -d ollama
docker exec arachne-ollama ollama pull glm-ocr

# 2) Build e avvio dello stack
docker compose up -d --build

# 3) Dashboard
#    http://localhost:8000
```

I dati vivono in due volumi: `arachne_data` (PDF, markdown, grafi, db) e
`ollama_models` (i modelli scaricati). Senza GPU: rimuovi il blocco
`deploy.resources` dal servizio `ollama` nel compose — l'OCR gira su CPU
(lento ma funzionante).

---

## 3. Build backend con GPU (opzionale)

Di default il backend è **CPU-only** (torch wheel CPU): è la scelta giusta nel
99% dei casi, perché l'OCR gira sul container Ollama e spaCy-trf su CPU regge
paper interi. Se vuoi spaCy su GPU **dentro** il backend:

```bash
docker compose build --build-arg INSTALL_GPU=true
docker compose up -d
```

⚠️ Avvertenza: con `INSTALL_GPU=true` l'immagine scarica la build CUDA
standard di PyTorch (~2 GB in più) e richiede il NVIDIA Container Toolkit
funzionante. Su GPU datacenter vecchie (sm_61 come la Tesla P4) le build
torch recenti **non sono compatibili**: restare su CPU o pinnare torch 2.5.1+cu121.

---

## 4. Il trucco del contesto: `OLLAMA_NUM_CTX=16384` ⚠️ FONDAMENTALE

GLM-OCR converte una pagina intera in **~6.000 token visivi** più l'output
markdown (fino a 8k token). Con il contesto di default di Ollama (4096) la
finestra non basta: il *context-shift* tronca e corrompe l'output
(**sezioni vuote, paragrafi duplicati, loop degenerativi** — verificato in
produzione).

Due livelli di difesa, entrambi già attivi:

1. **Per richiesta** (sempre attivo): il backend forza
   `options: {num_ctx: 16384, num_predict: 8192}` in ogni chiamata
   `/api/generate`. Funziona con qualsiasi tag `glm-ocr`.
2. **Lato server** (consigliato): la variabile `OLLAMA_NUM_CTX=16384` è già
   nel `docker-compose.yml` (servizio `ollama`). Per un'Ollama **locale**
   (bare-metal, fuori da Docker):

   ```bash
   # Linux (systemd):  sudo systemctl edit ollama
   [Service]
   Environment="OLLAMA_NUM_CTX=16384"

   # oppure avvio manuale
   OLLAMA_NUM_CTX=16384 ollama serve
   ```

VRAM necessaria: modello ~3 GB + KV cache 16k (<1 GB con flash-attention).
Una GPU da 8 GB basta; su 16 GB (RTX A2000) si è larghi.

---

## 5. Il fix `LD_LIBRARY_PATH` (run bare-metal / venv GPU)

Se esegui il backend **fuori da Docker** con un venv CUDA (torch+cupy dalle
wheel `nvidia-*-cu12`), spaCy può **cadere silenziosamente su CPU**
(`[warn] GPU non disponibile (CuPy is not installed)`, poi `gpu=False`)
perché le shared library nvidia non sono nel linker path. Fix:

```bash
export LD_LIBRARY_PATH="$(ls -d /percorso/venv/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH}"
uvicorn dashboard.app:app --host 0.0.0.0 --port 8001
```

Verifica (con lo stesso env):

```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"   # deve stampare >= 1
```

Lo script di riferimento per la produzione è `scripts/restart_prod.sh`
(include già l'export). Nei container ufficiali il problema non si pone.

---

## 6. Setup bare-metal (senza Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Modello NLP transformer (EN, qualità max) — richiede torch:
pip install torch --index-url https://download.pytorch.org/whl/cpu   # o build CUDA
pip install -e '.[gpu]'
python -m spacy download en_core_web_trf
# Fallback leggeri IT/ES:
python -m spacy download it_core_news_lg es_core_news_lg

# Ollama locale con il modello OCR
ollama pull glm-ocr            # vedi sez. 4 per OLLAMA_NUM_CTX

uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
```

Config runtime in `settings.json` (repo root, non tracciato):

```json
{
  "nlp_model": "trf",
  "force_cpu": false,
  "ocr_mode": "auto",
  "ollama_base_url": "http://localhost:11434",
  "ollama_model": "glm-ocr-16k",
  "ocr_page_timeout": 600,
  "ocr_dpi": 200
}
```

`ocr_mode`: `auto` (OCR se Ollama risponde, altrimenti classico) | `ollama`
(forza OCR) | `classic` (solo PyMuPDF). Override possibili anche via env
(`OLLAMA_BASE_URL`, `OLLAMA_OCR_MODEL`, `ARACHNE_OCR_MODE`). Il modello viene
auto-risolto sui tag presenti: `glm-ocr-16k` → `glm-ocr:latest` → `glm-ocr`.
Per creare la variante 16k esplicita (opzionale):

```bash
curl -X POST http://localhost:11434/api/create -H 'Content-Type: application/json' -d \
  '{"model":"glm-ocr-16k","from":"glm-ocr:latest","parameters":{"num_ctx":16384},"stream":false}'
```

---

## 7. Workflow operativo

1. **Upload**: dalla home della dashboard seleziona i PDF (il workspace viene
   azzerato; lo storico dei run in `graph_out/runs/` è immutabile e preservato).
2. **Pipeline live**: il terminale della UI streama i log in tempo reale —
   `[OCR] Pagina X/Y completata`, chunk `[SVO] k/N`, metriche SNA.
3. **Esplora**: HUD WebGL (Sigma.js + ForceAtlas2), dimensione nodo =
   betweenness, colore = community Louvain.
4. **Export**: GEXF/GraphML per Gephi dall'Export Hub, scope `all|hub`.

Struttura:

```
src/ingest_pdf.py     PDF -> Markdown: raster PyMuPDF -> Ollama diretto (Zero-SDK)
src/extract_svo.py    MD -> graph.json (spaCy EN/IT/ES, safety slicer 1800 char)
src/sna_metrics.py    graph -> metriche (betweenness, Louvain, constraint)
src/hardware_probe.py probe GPU/VRAM + disponibilita' OCR (mai eccezionale)
dashboard/            FastAPI + HUD single-file (Sigma.js/Graphology)
scripts/restart_prod.sh  restart bare-metal con LD_LIBRARY_PATH (sez. 5)
docs/                 report storici (FEASIBILITY_GLM_OCR.md = era SDK, superato)
```

## 💻 Hardware consigliato

- **OCR (container ollama)**: GPU NVIDIA ≥ 8 GB VRAM (Tesla P4 ok, RTX/A2000
  ideale). ~30-75 s/pagina su P4; proporzionalmente più veloce su RTX.
- **Backend**: qualsiasi CPU moderna; 8 GB RAM comodi per spaCy-trf.
- **CPU fallback totale**: senza GPU tutto funziona (OCR lento su CPU,
  spaCy passa automaticamente ai modelli lg).
- **Agentic Use**: punta il tuo CLI agent (Hermes, Claude Code, ecc.) alla
  directory `graph_out/`: JSON pensato per minimizzare i token LLM (GraphRAG)
  massimizzando l'inferenza causale.
