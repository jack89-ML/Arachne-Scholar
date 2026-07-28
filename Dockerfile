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
# PINNING via URL wheel DIRETTO: la lookup di compatibilita' di
# `spacy download nome-x.y.z` e' fragile (fallita in produzione); l'URL
# GitHub Releases e' l'artefatto esatto e versionato.
ARG SPACY_MODELS_BASE=https://github.com/explosion/spacy-models/releases/download
RUN pip install \
    ${SPACY_MODELS_BASE}/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl \
    ${SPACY_MODELS_BASE}/it_core_news_lg-3.8.0/it_core_news_lg-3.8.0-py3-none-any.whl \
    ${SPACY_MODELS_BASE}/es_core_news_lg-3.8.0/es_core_news_lg-3.8.0-py3-none-any.whl

# OPZIONALE — modello transformer EN (nlp_model=trf, qualita' max) + GPU CUDA:
# torch (build CUDA dai wheel PyPI su Linux), spacy-transformers e cupy
# pinnati. Richiede runtime nvidia sul host (vedi docker-compose.gpu.yml).
#   docker compose build --build-arg INSTALL_GPU=true
ARG INSTALL_GPU=false
RUN if [ "$INSTALL_GPU" = "true" ]; then \
        pip install torch spacy-transformers==1.4.0 cupy-cuda12x==14.1.1 \
        && pip install ${SPACY_MODELS_BASE}/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl; \
    fi

EXPOSE 8000

# Dati persistenti (PDF, markdown, grafi, arachne.db)
VOLUME ["/app/data"]

CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
