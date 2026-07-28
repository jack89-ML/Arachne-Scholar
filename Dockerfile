# Arachne Scholar — Backend (FastAPI + SpaCy SVO + ingest OCR diretto via Ollama)
# L'OCR NON vive in questa immagine: il backend parla HTTP al container ollama
# (vedi docker-compose.yml). Niente SDK glmocr, niente layout detector.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard

RUN pip install --upgrade pip \
    && pip install .

# Modelli NLP "lg" (default nlp_model=auto): leggeri, niente torch, CPU ok.
# Sono il percorso standard: l'OCR pesante gira sul container ollama, non qui.
# Versioni PINNATE (compatibilita' spacy 3.8.x): stesso modello = stesso grafo.
RUN python -m spacy download en_core_web_lg-3.8.0 \
    && python -m spacy download it_core_news_lg-3.8.0 \
    && python -m spacy download es_core_news_lg-3.8.0

# OPZIONALE — modello transformer EN (nlp_model=trf, qualita' max):
# richiede torch + spacy-transformers. Immagine molto piu' pesante.
#   docker compose build --build-arg INSTALL_GPU=true
# (torch da PyPI = build CUDA su Linux; per torch CPU-only installare a mano
#  da https://download.pytorch.org/whl/cpu prima di spacy-transformers)
ARG INSTALL_GPU=false
RUN if [ "$INSTALL_GPU" = "true" ]; then \
        pip install torch spacy-transformers \
        && python -m spacy download en_core_web_trf-3.8.0; \
    fi

EXPOSE 8000

# Dati persistenti (PDF, markdown, grafi, arachne.db)
VOLUME ["/app/data"]

CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
