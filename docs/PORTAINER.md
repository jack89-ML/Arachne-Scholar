# Deploy con Portainer — Arachne Scholar

Guida per il deploy dello stack via Portainer (Stacks → Add stack), con e
senza GPU. Per il setup bare-metal vedi [ADVANCED_SETUP.md](ADVANCED_SETUP.md).

---

## ⚡ La risposta breve: "basta lo YAML in Portainer per la GPU?"

**Sì, a tre condizioni** — se una manca, il container parte ma spaCy resta su
CPU (con `[warn] GPU non disponibile` nei log):

1. **L'host Docker ha il driver NVIDIA + NVIDIA Container Toolkit.**
   Verifica sull'host: `nvidia-smi` deve rispondere, e
   `docker info | grep -i nvidia` deve mostrare `Runtimes: ... nvidia`.
   Senza il toolkit, il blocco `deploy.devices` viene ignorato o il deploy
   fallisce.
2. **Il build del backend deve avere `INSTALL_GPU=true`** — è dentro gli YAML
   qui sotto come build arg. Senza, l'immagine non contiene torch/cupy/trf.
3. **Il servizio backend deve riservare la GPU** nel compose (blocco
   `deploy.resources.reservations.devices`, presente negli YAML GPU).

> ℹ️ L'accelerazione GPU dell'**OCR** NON dipende dallo stack del backend:
> la usa Ollama sulla macchina dove gira (es. il server `SERVER-HOST` con la
> sua installazione nativa). Se l'OCR remoto è già su GPU, non devi fare nulla.

---

## Scenario A — Solo backend in container, OCR su Ollama remoto (LAN)

Layout attuale consigliato: Ollama resta quello nativo su `SERVER-HOST`
(già con GPU e `glm-ocr`), Portainer deploya **solo il backend** su un host
qualsiasi della LAN. Incolla in Portainer → Stacks:

```yaml
services:
  backend:
    build:
      context: https://github.com/jack89-ML/Arachne-Scholar.git#main
    image: arachne-scholar:latest
    container_name: arachne-backend
    ports:
      - "8000:8000"
    volumes:
      - arachne_data:/app/data
    environment:
      - OLLAMA_BASE_URL=http://SERVER-HOST:11434
      # - ARACHNE_NLP_MODEL=auto   # auto (lg, CPU) | trf (richiede GPU: vedi Scenario B)
    restart: unless-stopped

volumes:
  arachne_data:
```

Verifica post-deploy: `http://<host>:8000/api/system/hardware` deve mostrare
`"ocr_available": true` e `"ollama_url": "http://SERVER-HOST:11434"`.

---

## Scenario B — Stack completo sulla macchina GPU (tutto in uno)

Per l'host CON la GPU: Ollama + backend in container, spaCy trf su CUDA.
Se sull'host gira già un Ollama **nativo** (porta 11434 occupata), fermalo
prima (`sudo systemctl stop ollama`) oppure togli il blocco `ports` del
servizio ollama — il backend lo raggiunge comunque via DNS interno, ma due
Ollama sulla stessa GPU si contendono la VRAM.

```yaml
services:
  ollama:
    image: ollama/ollama:0.32.5
    container_name: arachne-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    environment:
      - OLLAMA_FLASH_ATTENTION=1
      - OLLAMA_NUM_CTX=16384
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "ollama", "ls"]
      interval: 5s
      timeout: 5s
      retries: 12
      start_period: 10s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  ollama-init:
    image: ollama/ollama:0.32.5
    container_name: arachne-ollama-init
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      - OLLAMA_HOST=http://ollama:11434
    entrypoint: ["sh", "-c", "ollama pull glm-ocr"]
    restart: "no"

  backend:
    build:
      context: https://github.com/jack89-ML/Arachne-Scholar.git#main
      args:
        INSTALL_GPU: "true"
    image: arachne-scholar:latest
    container_name: arachne-backend
    ports:
      - "8000:8000"
    volumes:
      - arachne_data:/app/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - ARACHNE_NLP_MODEL=trf
    depends_on:
      ollama:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama_models:
  arachne_data:
```

Verifica GPU nel log del backend (Portainer → Containers → logs): durante
l'estrazione SVO deve comparire `[setup] lang=en model=en_core_web_trf
mode=trf gpu=True`.

---

## Note

- **Portainer da Git**: negli YAML sopra il `build.context` punta alla repo
  GitHub — Portainer la clona e builda da sé. In alternativa: Stacks → Git
  repository → URL repo, compose path `docker-compose.yml` (e per la GPU
  aggiungi `docker-compose.gpu.yml` come *additional compose file* oppure
  incolla lo YAML dello Scenario B).
- **Persistenza**: grafi, DB e PDF vivono nel volume `arachne_data`; i modelli
  OCR nel volume `ollama_models`. Un re-deploy non perde nulla.
- **Equivale da CLI** (senza Portainer), nella repo clonata:
  ```bash
  docker compose up -d --build                                            # CPU/lg
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build   # GPU/trf
  ```
- **Rete**: lo stack espone le porte su tutte le interfacce — pensato per LAN
  privata, niente autenticazione. Non pubblicarlo su Internet.
