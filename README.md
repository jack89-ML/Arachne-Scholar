# Arachne Scholar 🕷️

Local-first Knowledge Graph Engine for Academic Literature via SVO Dependency Parsing.

## Architecture

```text
Arachne-Scholar/
├── data/                  # Raw PDFs and Markdown texts (git ignored)
├── graph_out/             # Output graph.json (git ignored)
├── src/                   # Core engine
│   ├── __init__.py
│   ├── extract_svo.py     # spaCy & SVO extraction pipeline
│   ├── clean_graph.py     # Anti-noise post-processing filter
│   └── sna_metrics.py     # Topological analysis & Structural Holes
├── dashboard/             # Interactive plancia di comando
│   ├── app.py             # FastAPI backend with Micro-RAG endpoints
│   └── static/
│       └── index.html     # vis-network frontend
├── .gitignore
├── pyproject.toml         # Dependencies and metadata
└── README.md
```
