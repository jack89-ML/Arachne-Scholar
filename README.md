# Arachne Scholar 🕷️

Local-first Knowledge Graph Engine for Academic Literature via SVO Dependency Parsing. 
Arachne transforms raw academic PDFs into a queryable topological graph, exposing Structural Holes and Epistemic Connections.

## 🚀 The Workflow
1. **Ingest**: Drop your academic papers in `data/raw_pdfs/`.
2. **Convert**: Run `python src/ingest_pdf.py` to extract text to Markdown.
3. **Parse**: Run `python src/extract_svo.py` to run the spaCy transformer and extract SVO (Subject-Verb-Object) networks.
4. **Analyze**: Run `python src/sna_metrics.py` to calculate Betweenness Centrality and Structural Holes (Burt's constraint).
5. **Explore**: Launch the FastAPI dashboard `uvicorn app:app --app-dir dashboard` to visualize the network.

## 💻 Hardware & Agent Integration
This pipeline is heavily optimized for **Local GPU Execution**.
- **GPU Requirements**: Nvidia GPU with at least 8GB VRAM (e.g., Tesla P4, RTX series) is highly recommended for the `en_core_web_trf` spaCy model.
- **CPU Fallback**: The engine will gracefully fall back to the lighter `en_core_web_lg` model if no GPU is detected.
- **Agentic Use**: Point your local CLI agent (Hermes, Claude Code, etc.) to the `graph_out/` directory. The JSON structure is designed to minimize LLM token usage (GraphRAG) while maximizing causal inference accuracy.

*A Dockerfile and docker-compose setup for one-click deployment are on the roadmap.*
