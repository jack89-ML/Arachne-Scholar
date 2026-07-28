# Advanced Setup — Arachne Scholar

Guide sistemistiche avanzate per chi NON usa il percorso Docker standard.
Se cerchi il quickstart, torna al [README.md](../README.md).

Contenuti:
1. [Setup Bare-Metal (senza Docker)](#1-setup-bare-metal-senza-docker)
2. [Il trucco del contesto: OLLAMA_NUM_CTX=16384](#2-il-trucco-del-contesto-ollama_num_ctx16384)
3. [Il fix LD_LIBRARY_PATH (SpaCy su GPU bare-metal)](#3-il-fix-ld_library_path-spacy-su-gpu-bare-metal)
4. [Reference: settings.json](#4-reference-settingsjson)

---

## 1. Setup Bare-Metal (senza Docker)

Tutto lo stack su un'unica macchina: backend FastAPI + spaCy + Ollama locale.

```bash
git clone https://github.com/jack89-ML/Arachne-Scholar.git
cd Arachne-Scholar

python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Modello NLP transformer (EN, qualita' max) — richiede torch:
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
# ...oppure la build CUDA della tua piattaforma (vedi avvertenza sotto)
pip install -e '.[gpu]'
python -m spacy download en_core_web_trf
# Fallback leggeri IT/ES (consigliati):
python -m spacy download it_core_news_lg es_core_news_lg

# Ollama locale con il modello OCR (vedi sez. 2 per OLLAMA_NUM_CTX!)
ollama pull glm-ocr

uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
```

⚠️ **Avvertenza torch su GPU datacenter vecchie (sm_61, es. Tesla P4):** le
build torch recenti non supportano sm_61. Pinnare `torch==2.5.1+cu121` e
`cupy-cuda12x`, oppure far girare spaCy su CPU (automatico se la GPU non è
utilizzabile). La RTX A2000 (sm_86) e le RTX consumer non hanno questo limite.

Python 3.14: `spacy-transformers` potrebbe non avere wheel precompilate
(spacy-alignments). Installare il core e usare i modelli `lg`, oppure restare
su Python 3.11/3.12 per il modello `trf`.

---

## 2. Il trucco del contesto: `OLLAMA_NUM_CTX=16384`

**Perché è fondamentale.** GLM-OCR converte una pagina intera in **~6.000
token visivi** più l'output markdown (fino a 8k token). Con il contesto di
default di Ollama (**4096**) la finestra non basta: il *context-shift* tronca
e corrompe l'output. Sintomi reali osservati in produzione: sezioni vuote,
paragrafi duplicati, loop degenerativi ("The Law Law Law...").

**Due livelli di difesa (entrambi consigliati):**

1. **Per richiesta — già attivo, non toccare.** Il backend forza
   `options: {num_ctx: 16384, num_predict: 8192}` in ogni chiamata
   `/api/generate`. Copre qualsiasi tag `glm-ocr`, anche senza configurazione
   lato server.
2. **Lato server — consigliato.** Imposta `OLLAMA_NUM_CTX=16384` sul processo
   `ollama serve`:

**Docker Compose:** già presente nel servizio `ollama` del
`docker-compose.yml` (niente da fare).

**systemd (Linux, installazione nativa):**

```bash
sudo systemctl edit ollama
# incolla:
[Service]
Environment="OLLAMA_NUM_CTX=16384"

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**Avvio manuale (qualsiasi OS):**

```bash
OLLAMA_NUM_CTX=16384 ollama serve
```

**Windows (nativo, no WSL2):** imposta la variabile d'ambiente utente
`OLLAMA_NUM_CTX=16384` (Impostazioni → Variabili d'ambiente) e riavvia Ollama
dalla tray.

**Verifica del contesto attivo:** mentre l'OCR lavora, il runner deve mostrare
il flag `-c 16384`:

```bash
ps -eo cmd | grep '[l]lama-server' | grep -o '\-c [0-9]*'
```

**VRAM necessaria:** modello ~3 GB + KV cache 16k (<1 GB con flash-attention).
8 GB bastano, 16 GB (RTX A2000) è largo.

**Variante di modello esplicita (opzionale):** invece dell'env, puoi creare
un tag con il contesto pinnato:

```bash
curl -X POST http://localhost:11434/api/create -H 'Content-Type: application/json' -d \
  '{"model":"glm-ocr-16k","from":"glm-ocr:latest","parameters":{"num_ctx":16384},"stream":false}'
```

Nota: il campo `modelfile` nel JSON di `/api/create` è rifiutato da alcune
versioni di Ollama — usa `from` + `parameters` come sopra.

---

## 3. Il fix `LD_LIBRARY_PATH` (SpaCy su GPU bare-metal)

Se il backend gira **fuori da Docker** con un venv CUDA (torch+cupy dalle
wheel `nvidia-*-cu12`), spaCy può **cadere silenziosamente su CPU**:

```
[warn] GPU non disponibile (CuPy is not installed), fallback CPU
[setup] lang=en model=en_core_web_trf gpu=False
```

Causa: le shared library nvidia dentro `site-packages` non sono nel linker
path. Fix (da fare NELLO STESSO shell/env che avvia uvicorn):

```bash
export LD_LIBRARY_PATH="$(ls -d /percorso/venv/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH}"
uvicorn dashboard.app:app --host 0.0.0.0 --port 8001
```

Verifica (sempre con lo stesso env):

```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"   # deve stampare >= 1
```

Lo script di riferimento è [`scripts/restart_prod.sh`](../scripts/restart_prod.sh):
fa l'export prima del `nohup`. Se scrivi un tuo unit systemd o uno script di
avvio, copia quella riga. Nei container ufficiali il problema non si pone.

---

## 4. Reference: settings.json

File di configurazione runtime in repo root (NON tracciato in git). Il backend
lo legge all'avvio di ogni pipeline; ogni chiave ha un default sensato e può
essere omessa.

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

| Chiave | Default | Effetto |
|---|---|---|
| `nlp_model` | `auto` | `auto` = modelli `lg` (leggeri, niente torch, CPU ok); `trf` = transformer `en_core_web_trf` (richiede torch+spacy-transformers, GPU opzionale; su IT/ES ricade su `lg` perché non esistono pipeline trf ufficiali; se il trf non è installato, fallback automatico su `lg` con warning nei log) |
| `force_cpu` | `false` | `true` → esporta `CUDA_VISIBLE_DEVICES=""` ai subprocess (SpaCy CPU) |
| `ocr_mode` | `auto` | `auto` = OCR se Ollama risponde, altrimenti PyMuPDF; `ollama` = forza OCR; `classic` = solo PyMuPDF |
| `ollama_base_url` | `http://localhost:11434` | endpoint Ollama (in Docker: `http://ollama:11434` via env) |
| `ollama_model` | `glm-ocr-16k` | auto-risolto sui tag: `glm-ocr-16k` → `glm-ocr:latest` → `glm-ocr` |
| `ocr_page_timeout` | `600` | secondi max per singola pagina OCR |
| `ocr_dpi` | `200` | risoluzione raster PyMuPDF |

Override equivalenti via ambiente: `OLLAMA_BASE_URL`, `OLLAMA_OCR_MODEL`,
`ARACHNE_OCR_MODE` (valori: `auto|ollama|classic`), `ARACHNE_NLP_MODEL`
(valori: `auto|trf`).
