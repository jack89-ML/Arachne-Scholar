# Arachne Scholar — Immagine di produzione (CPU di default, GPU opzionale)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dipendenze core (py3.11: wheel precompilate per tutto lo stack, niente build Rust)
COPY pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard

RUN pip install --upgrade pip \
    && pip install . pymupdf python-multipart

# Extra GPU (opzionale): pip install .[gpu]  — richiede toolkit CUDA sull'host.
# Abilitare con: docker build --build-arg INSTALL_GPU=true .
ARG INSTALL_GPU=false
RUN if [ "$INSTALL_GPU" = "true" ]; then \
        pip install '.[gpu]' || echo '[warn] extra GPU non installato, continuo in CPU'; \
    fi

# Modello NLP inglese precaricato (transformer, default); IT/ES scaricati on-demand
RUN python -m spacy download en_core_web_trf

EXPOSE 8000

# Dati persistenti (PDF, markdown, grafi, arachne.db)
VOLUME ["/app/data"]

CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
