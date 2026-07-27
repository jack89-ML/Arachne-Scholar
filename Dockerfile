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

# SpaCy trf (EN) richiede spacy-transformers -> torch. Di default torch CPU:
# immagine leggera e portabile (l'OCR gira sul container ollama, non qui).
# Per GPU nel backend (non necessaria): docker build --build-arg INSTALL_GPU=true .
ARG INSTALL_GPU=false
RUN if [ "$INSTALL_GPU" = "true" ]; then \
        pip install torch spacy-transformers; \
    else \
        pip install torch --index-url https://download.pytorch.org/whl/cpu \
        && pip install spacy-transformers; \
    fi

# Modelli NLP precaricati: EN transformer (default), IT/ES large.
RUN python -m spacy download en_core_web_trf \
    && python -m spacy download it_core_news_lg \
    && python -m spacy download es_core_news_lg

EXPOSE 8000

# Dati persistenti (PDF, markdown, grafi, arachne.db)
VOLUME ["/app/data"]

CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
