# Arachne Scholar 🕷️
### Deterministic Knowledge Graph Engine for Agentic RAG

![Version](https://img.shields.io/badge/version-0.2.0-a855f7) ![Docker](https://img.shields.io/badge/docker-compose%20ready-22d3ee) ![Python](https://img.shields.io/badge/python-%E2%89%A53.10-4ade80) ![OCR](https://img.shields.io/badge/OCR-GLM--OCR%20via%20Ollama-fbbf24)

---

## 🎯 L'Obiettivo: Knowledge Base a Costo Zero, su Hardware Consumer

Costruire una pipeline **100% locale e gratuita**: nessuna chiamata ad API
esterne a pagamento, nessun token meter, nessun dato che lascia la macchina.
Il sistema è progettato per girare fluentemente su **hardware consumer a basso
costo e basso consumo** — una GPU da 8/12 GB di VRAM è sufficiente per
l'intero workflow, dalla pagina rasterizzata al grafo finale.

Arachne trasforma complessi PDF accademici in una **Knowledge Base solida e
versionabile** (`graph.json`): nodi tipizzati (concetti, autori, istituzioni),
archi relazionali, metriche di rete già calcolate. Un artefatto pronto per
l'**Agentic RAG**: perfettamente navigabile da interfacce come Graphify e dai
moderni CLI Agent, che possono ragionare sulla struttura causale del corpus
invece che su sacchetti di chunk testuali.

## 🧭 Il Manifesto Deterministico: SpaCy, non LLM

Per l'estrazione delle relazioni SVO (Subject-Verb-Object) **non usiamo un
LLM, e la ragione è epistemologica prima ancora che tecnica.**

Gli LLM sono macchine **probabilistiche**: campionano da distribuzioni,
cambiano risposta al variare del seed e del contesto, e sono strutturalmente
inclini all'allucinazione. Un grafo di conoscenza estratto da un LLM è un
artefatto *non replicabile*: rieseguilo domani e otterrai nodi e relazioni
diversi. Per la ricerca accademica questo è inaccettabile — un dato che non si
riproduce non è un dato, è un'aneddotica.

Arachne separa nettamente **percezione** e **ragionamento**:

- **L'OCR visivo (GLM-OCR) ha un solo compito meccanico**: *leggere* i pixel
  della pagina e trascriverli in testo. Nient'altro.
- **L'estrazione logica è affidata ai trasformatori NLP di SpaCy**: parsing
  sintattico a dipendenze, POS tagging e dependency traversal sono calcolo
  **matematico, geometrico e sintattico** del linguaggio naturale — non
  campionamento.

Lo stesso PDF produce **sempre lo stesso grafo**: stessa entità, stessa
relazione, stessa metrica. Il risultato è una **mappa causale del pensiero
umano** — deterministica, esatta e rigorosamente replicabile — su cui si può
fare scienza: confronti longitudinali, studi di structural holes, revisione
paritaria dei risultati.

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
