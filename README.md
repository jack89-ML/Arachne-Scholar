# Arachne Scholar 🕷️

**Local-first Knowledge Graph Engine for Academic Literature** — dai PDF alla mappa epistemica, tutto in locale.

![Version](https://img.shields.io/badge/version-0.2.0-a855f7) ![Docker](https://img.shields.io/badge/docker-compose%20ready-22d3ee) ![Python](https://img.shields.io/badge/python-%E2%89%A53.10-4ade80) ![OCR](https://img.shields.io/badge/OCR-GLM--OCR%20via%20Ollama-fbbf24)

---

## 🎯 Obiettivo del progetto

Arachne trasforma una collezione di paper accademici in un **grafo di conoscenza
interrogabile**: estrae concetti, autori e relazioni SVO (Soggetto-Verbo-Oggetto)
e calcola metriche di rete (betweenness, community, structural holes) per
mostrare *come* le idee si connettono — non solo *di cosa* parlano.
Nessun cloud, nessun servizio esterno: tutto gira sulla tua macchina.

## ⚙️ Cosa fa e come funziona

Architettura a **2 container**, zero SDK esterni: il backend parla con Ollama
via HTTP diretto.

```
┌─────────────────────────────┐      HTTP (DNS interno)      ┌──────────────────────────┐
│  backend (arachne-backend)  │  ──────────────────────────► │  ollama (arachne-ollama) │
│  FastAPI + HUD WebGL        │   POST /api/generate         │  modello: glm-ocr        │
│  spaCy SVO (EN/IT/ES)       │   pagina PNG + prompt        │  GPU passthrough CUDA    │
│  PyMuPDF raster in-memory   │ ◄──────────────────────────  │                          │
└─────────────────────────────┘   markdown della pagina      └──────────────────────────┘
```

Il flusso: **PyMuPDF** rasterizza ogni pagina PDF in memoria → **Ollama**
(GLM-OCR) la trascrisce in Markdown → **spaCy** estrae entità e triple SVO →
**NetworkX** calcola le metriche SNA → esplori tutto nella **HUD WebGL**
(Sigma.js + ForceAtlas2) o esporti in Gephi (GEXF/GraphML).

## 💻 Requisiti Hardware

- **GPU NVIDIA ≥ 8 GB VRAM** raccomandata per l'OCR (Tesla P4 ok, RTX / A2000 ideale)
- 8 GB RAM per il backend (modello NLP transformer)
- **Senza GPU** tutto funziona comunque: OCR su CPU (lento) o fallback testuale PyMuPDF
- Docker + NVIDIA Container Toolkit (Linux) o Docker Desktop + WSL2 (Windows 11)

## 🚀 Come si installa (Quickstart Docker)

```bash
git clone https://github.com/jack89-ML/Arachne-Scholar.git
cd Arachne-Scholar

# 1) Avvia Ollama e scarica il modello OCR (una tantum, ~2 GB)
docker compose up -d ollama
docker exec arachne-ollama ollama pull glm-ocr

# 2) Build e avvio dello stack
docker compose up -d --build

# 3) Apri la dashboard
#    http://localhost:8000
```

Backend con accelerazione GPU per spaCy (opzionale, immagine più pesante):

```bash
docker compose build --build-arg INSTALL_GPU=true
docker compose up -d
```

> 🔧 Setup bare-metal, tuning del contesto OCR (`OLLAMA_NUM_CTX=16384`) e fix
> `LD_LIBRARY_PATH`: vedi **[docs/ADVANCED_SETUP.md](docs/ADVANCED_SETUP.md)**.

## 🕹️ Come si usa

1. **Upload PDF** — dalla home seleziona i paper (EN/IT/ES) e premi
   *Avvia Estrazione* ⚡ (il workspace si azzera, lo storico dei run è preservato).
2. **Attendi l'elaborazione** — il terminale live mostra ogni pagina OCR
   (`[OCR] Pagina X/Y completata`) e i chunk NLP in tempo reale.
3. **Esplora la HUD** — grafo interattivo (dimensione = betweenness,
   colore = community), poi esporta in Gephi dall'Export Hub.

## 📸 Screenshots

![Dashboard Screenshot](docs/assets/dashboard.png)
*(aggiungi qui uno screenshot della HUD — salvalo in `docs/assets/dashboard.png`)*
