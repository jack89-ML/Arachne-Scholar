#!/usr/bin/env python3
"""
Arachne Scholar — NLP Knowledge Graph Builder (spaCy GPU) v2
Estrazione entità via NER + noun chunks con lemmatizzazione e filtri anti-rumore.
Archi via Dependency Parsing (SVO + relazioni logiche) con etichette verbali reali.

Uso:
    python3 build_local_graph.py [markdown_dir] [output_dir]
"""
import os, sys, json, glob, re, time
from collections import Counter

import spacy

# --- GPU SETUP ----------------------------------------------------------------
try:
    spacy.require_gpu()
    GPU_ACTIVE = True
except Exception as e:
    GPU_ACTIVE = False
    print(f"[warn] GPU non disponibile ({e}), fallback CPU", file=sys.stderr)

MODEL_NAME = "en_core_web_trf"
try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    MODEL_NAME = "en_core_web_lg"
    nlp = spacy.load(MODEL_NAME)

nlp.max_length = 6_000_000
print(f"[setup] model={MODEL_NAME} gpu={GPU_ACTIVE}", file=sys.stderr)

# --- STOPWORDS & FILTRI --------------------------------------------------------
GENERIC_HEADS = {
    "theory", "model", "concept", "method", "approach", "analysis", "study",
    "result", "results", "data", "paper", "chapter", "book", "table",
    "figure", "section", "process", "system", "work", "research", "field",
    "case", "example", "question", "problem", "issue", "fact", "reason",
    "way", "part", "kind", "type", "form", "level", "point", "time",
    "year", "years", "number", "people", "group", "groups", "thing",
    "things", "lot", "use", "role", "effect", "effects", "impact",
    "difference", "differences", "change", "changes", "relationship",
    "relationships", "variable", "variables", "factor", "factors",
    "aspect", "aspects", "element", "elements", "feature", "features",
    "context", "contexts", "dimension", "dimensions", "mechanism",
    "mechanisms", "structure", "structures", "pattern", "patterns",
    "outcome", "outcomes", "finding", "findings", "evidence",
    "literature", "review", "introduction", "conclusion", "discussion",
    "summary", "overview", "background", "framework", "perspective",
}

DETERMINERS = {"the", "a", "an", "this", "these", "that", "those", "some",
               "any", "each", "other", "another", "such", "all", "both",
               "several", "many", "most", "more", "few", "little", "much",
               "no", "not", "very", "quite", "rather"}

POSSESSIVES = {"his", "her", "their", "its", "our", "my", "your", "whose"}

JUNK_TOKENS = {"br", "p", "pp", "ed", "eds", "vol", "no", "cf", "ibid",
               "etc", "ie", "eg", "al", "et", "ff", "fn", "n", "nd"}

# Pronomi/question-words che il parser cattura come chunk -> nodi junk (fix v2.1)
JUNK_ENTITIES = {
    "which", "that", "who", "whom", "whose", "what", "whatever", "whichever",
    "they", "them", "their", "theirs", "he", "him", "his", "she", "her",
    "hers", "it", "its", "itself", "we", "us", "our", "ours", "i", "me",
    "my", "mine", "you", "your", "yours", "one", "ones", "someone",
    "something", "anything", "everything", "nothing", "anyone", "everyone",
    "nobody", "somebody", "everybody", "anybody", "this", "these", "those",
    "there", "here", "where", "when", "why", "how", "all", "both", "each",
    "either", "neither", "none", "few", "many", "much", "most", "more",
    "several", "others", "another", "such", "same", "own", "else",
    "whenever", "wherever", "however",
}

WEAK_VERBS = {"be", "have", "do", "say", "get", "make", "take", "see",
              "know", "think", "go", "come", "give", "tell", "call", "seem",
              "appear", "become", "remain", "stay", "begin", "start", "end",
              "put", "set", "let", "keep", "help", "try", "need", "want",
              "like", "feel", "look", "sound", "mean", "believe", "note",
              "mention", "refer", "cite", "quote", "state", "report",
              "according", "following", "based", "using", "used"}

RELATION_VERBS = {
    "analyze", "analyse", "define", "explain", "describe", "examine", "study",
    "investigate", "explore", "develop", "propose", "introduce", "apply",
    "employ", "test", "measure", "compare", "contrast", "evaluate",
    "assess", "demonstrate", "show", "reveal", "find", "identify", "argue",
    "claim", "suggest", "support", "challenge", "critique", "criticize",
    "extend", "build", "construct", "predict", "influence", "affect",
    "shape", "determine", "cause", "produce", "generate", "create", "form",
    "constitute", "comprise", "include", "contain", "involve", "require",
    "relate", "connect", "link", "associate", "correlate", "depend",
    "derive", "emerge", "arise", "lead", "contribute", "mediate",
    "moderate", "underlie", "ground", "frame", "conceptualize",
    "theorize", "formalize", "operationalize", "validate", "replicate",
    "confirm", "refute", "reject", "assume", "hypothesize", "posit",
    "address", "focus", "concentrate", "center", "deal", "treat", "cover",
    "discuss", "consider", "regard", "view", "interpret", "understand",
    "transform", "change", "alter", "modify", "structure", "organize",
    "integrate", "combine", "merge", "unify", "separate", "distinguish",
    "differentiate", "classify", "categorize", "group", "cluster",
}

NER_TYPE_MAP = {
    "PERSON": "author",
    "ORG": "institution",
    "GPE": "institution",
    "WORK_OF_ART": "data",
    "EVENT": "concept",
    "LAW": "concept",
    "PRODUCT": "concept",
    "NORP": "concept",
}

TYPE_KEYWORDS = [
    ("theory", {"theory", "theories", "ism", "paradigm", "doctrine",
                "thesis", "hypothesis", "axiom"}),
    ("methodology", {"method", "methods", "methodology", "analysis",
                     "technique", "design", "experiment", "survey",
                     "measurement", "estimation", "simulation", "regression",
                     "sampling", "ethnography", "interview", "questionnaire"}),
    ("data", {"data", "dataset", "statistics", "sample", "observations",
              "corpus", "archive", "database"}),
]


def normalize(text):
    """Normalizza: lowercase, rimuove caratteri non-alfabetici, collassa spazi."""
    t = text.lower()
    t = re.sub(r"[_*#`~\[\](){}<>]", " ", t)  # markdown artifacts
    t = re.sub(r"[^a-z0-9\s\-']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def node_id_from(text):
    nid = re.sub(r"[^a-z0-9\s-]", "", text.lower())
    nid = re.sub(r"[\s_]+", "-", nid).strip("-")
    return nid[:80]


def clean_chunk_label(chunk):
    """
    Pulisce il label del chunk: rimuove determinanti, possessivi,
    tokens junk, e normalizza la forma.
    """
    tokens = []
    for tok in chunk:
        low = tok.text.lower()
        if low in DETERMINERS or low in POSSESSIVES or low in JUNK_TOKENS:
            continue
        if tok.is_punct or tok.is_space:
            continue
        # rimuovi caratteri markdown residui
        txt = re.sub(r"<\s*br\s*/?\s*>", " ", tok.text, flags=re.IGNORECASE)
        txt = re.sub(r"[_*#`~\[\](){}<>]", "", txt)
        if not txt or not re.search(r"[a-zA-Z]", txt):
            continue
        tokens.append(txt.strip())
    label = " ".join(tokens).strip()
    # collassa spazi e trattini
    label = re.sub(r"\s+-\s+", "-", label)
    label = re.sub(r"\s+", " ", label).strip(" -")
    return label


def chunk_lemma(chunk):
    """Lemma del root del chunk per dedup morfologico."""
    root = chunk.root
    if root.lemma_ and len(root.lemma_) > 1:
        return normalize(root.lemma_)
    return normalize(root.text)


def is_valid_entity(label, lemma_head):
    """Filtro anti-rumore rigoroso."""
    if not label or len(label) < 4 or len(label) > 90:
        return False
    norm = normalize(label)
    if not norm or len(norm) < 4:
        return False
    # pronomi/question-words (fix v2.1: evita nodi tipo 'which', 'they')
    if norm in JUNK_ENTITIES or lemma_head in JUNK_ENTITIES:
        return False
    # artifact markdown: pipe, underscore, asterischi in etichetta
    if label.startswith(("|", "_", "#", "*", "`")) or "|" in label:
        return False
    # head generico (es. "mechanism" da solo) — scarta
    if lemma_head in GENERIC_HEADS and len(norm.split()) == 1:
        return False
    # norm generico completo
    if norm in GENERIC_HEADS:
        return False
    # tutto junk/numeri
    if not re.search(r"[a-z]", norm):
        return False
    if re.fullmatch(r"[\d\s.,%\-']+", norm):
        return False
    # singola parola troppo comune
    words = norm.split()
    if len(words) == 1 and words[0] in GENERIC_HEADS:
        return False
    # inizia con numero
    if words and words[0].isdigit():
        return False
    return True


def classify(label, lemma_head):
    lt = f"{normalize(label)} {lemma_head}"
    for ntype, keywords in TYPE_KEYWORDS:
        for kw in keywords:
            if re.search(rf"\b{kw}\b", lt):
                return ntype
    return "concept"


def extract_entities(doc, entities):
    """Estrae entità da doc spaCy e le aggiunge al dict condiviso."""
    # 1) NER
    for ent in doc.ents:
        label = clean_chunk_label(ent)
        norm = normalize(label)
        if not is_valid_entity(label, norm):
            continue
        ntype = NER_TYPE_MAP.get(ent.label_, "concept")
        nid = node_id_from(norm)
        if nid and nid not in entities:
            entities[nid] = {
                "id": nid,
                "label": label[:80],
                "type": ntype,
                "description": f"{ntype} (NER:{ent.label_})",
            }

    # 2) Noun chunks
    for chunk in doc.noun_chunks:
        label = clean_chunk_label(chunk)
        lem = chunk_lemma(chunk)
        if not is_valid_entity(label, lem):
            continue
        norm = normalize(label)
        nid = node_id_from(norm)
        if not nid or nid in entities:
            continue
        ntype = classify(label, lem)
        entities[nid] = {
            "id": nid,
            "label": label[:80],
            "type": ntype,
            "description": f"{ntype} (chunk, lemma:{lem})",
        }
    return entities


def nid_for_span(token, entities):
    """
    Dato un token, risale al chunk che lo contiene nella frase e
    restituisce il nid entità corrispondente.
    """
    sent = token.sent
    for chunk in sent.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            label = clean_chunk_label(chunk)
            norm = normalize(label)
            nid = node_id_from(norm)
            if nid in entities:
                return nid
    # fallback: il token stesso come nome proprio
    if token.pos_ == "PROPN":
        norm = normalize(token.text)
        nid = node_id_from(norm)
        if nid in entities:
            return nid
    return None


def extract_svo_edges(doc, entities):
    """Dependency parsing: triple (soggetto, verbo, oggetto) con etichette verbali."""
    edges = Counter()
    for sent in doc.sents:
        for tok in sent:
            if tok.pos_ != "VERB":
                continue
            lemma = tok.lemma_.lower()
            if lemma in WEAK_VERBS or lemma not in RELATION_VERBS:
                continue

            subj_nid, obj_nids = None, []

            for child in tok.children:
                dep = child.dep_
                if dep in ("nsubj", "nsubjpass", "agent", "expl"):
                    nid = nid_for_span(child, entities)
                    if nid:
                        subj_nid = nid
                elif dep in ("dobj", "attr", "acomp", "oprd", "dative"):
                    nid = nid_for_span(child, entities)
                    if nid:
                        obj_nids.append(nid)
                elif dep == "prep":
                    for gc in child.children:
                        if gc.dep_ in ("pobj", "pcomp"):
                            nid = nid_for_span(gc, entities)
                            if nid:
                                obj_nids.append(nid)
                elif dep in ("xcomp", "ccomp", "advcl"):
                    # verbo subordinato: collega soggetto all'oggetto della subordinata
                    for gc in child.children:
                        if gc.dep_ in ("dobj", "attr"):
                            nid = nid_for_span(gc, entities)
                            if nid:
                                obj_nids.append(nid)

            if subj_nid:
                for tgt in obj_nids:
                    if tgt != subj_nid:
                        edges[(subj_nid, tgt, lemma)] += 1
    return edges


def extract_cochunk_edges(doc, entities, max_per_sent=5):
    """Co-occorrenza intra-frase come relazione logica debole."""
    edges = Counter()
    for sent in doc.sents:
        nids, seen = [], set()
        for chunk in sent.noun_chunks:
            label = clean_chunk_label(chunk)
            norm = normalize(label)
            nid = node_id_from(norm)
            if nid in entities and nid not in seen:
                seen.add(nid)
                nids.append(nid)
        if len(nids) < 2:
            continue
        nids = nids[:max_per_sent]
        for i in range(len(nids)):
            for j in range(i + 1, len(nids)):
                a, b = sorted([nids[i], nids[j]])
                edges[(a, b, "co_occurs")] += 1
    return edges


def chunk_text(text, max_chars=200_000):
    if len(text) <= max_chars:
        return [text]
    chunks, current = [], ""
    for para in text.split("\n\n"):
        if len(current) + len(para) > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current:
        chunks.append(current)
    return chunks


def process_file(path, nlp):
    with open(path, encoding="utf-8") as f:
        text = f.read()

    entities, svo_edges, co_edges = {}, Counter(), Counter()
    pieces = chunk_text(text)
    t0 = time.time()

    for i, piece in enumerate(pieces):
        for doc in nlp.pipe([piece], batch_size=8):
            entities = extract_entities(doc, entities)
            svo_edges.update(extract_svo_edges(doc, entities))
            co_edges.update(extract_cochunk_edges(doc, entities))
        elapsed = time.time() - t0
        print(f"    [{i+1}/{len(pieces)}] {len(entities)} entità, "
              f"{len(svo_edges)} SVO, {elapsed:.0f}s", file=sys.stderr)

    return entities, svo_edges, co_edges


def main():
    md_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/scholar_engine/converted_md")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
        "~/scholar_engine/graph_out")
    os.makedirs(out_dir, exist_ok=True)

    md_files = sorted(glob.glob(os.path.join(md_dir, "*.md")))
    if not md_files:
        print("Nessun markdown trovato.", file=sys.stderr)
        sys.exit(1)

    all_entities, all_svo, all_co = {}, Counter(), Counter()

    for fp in md_files:
        print(f"\n=== {os.path.basename(fp)[:70]} ===", file=sys.stderr)
        ents, svo, co = process_file(fp, nlp)
        all_entities.update(ents)
        all_svo.update(svo)
        all_co.update(co)

    nodes = list(all_entities.values())

    # Archi SVO: soglia >=1, etichetta verbale reale
    links, seen = [], set()
    for (src, tgt, rel), count in all_svo.most_common():
        if (src, tgt, rel) in seen:
            continue
        seen.add((src, tgt, rel))
        links.append({
            "source": src, "target": tgt, "relation": rel,
            "weight": min(count, 5),
            "confidence": "extracted" if count >= 2 else "inferred",
        })

    # co_occur: soglia >=5 (forte riduzione rumore), peso 1-2
    for (src, tgt, rel), count in all_co.most_common():
        if count < 5:
            break
        if (src, tgt, rel) in seen:
            continue
        seen.add((src, tgt, rel))
        links.append({
            "source": src, "target": tgt, "relation": rel,
            "weight": 1 if count < 10 else 2,
            "confidence": "inferred",
        })

    graph = {
        "nodes": nodes,
        "links": links,
        "edges": links,
        "meta": {"model": MODEL_NAME, "gpu": GPU_ACTIVE,
                 "engine": "spacy trf: NER + noun chunks + dependency parsing"},
    }

    out_path = os.path.join(out_dir, "graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    # --- Metriche ---
    type_counts = Counter(n["type"] for n in nodes)
    rel_counts = Counter(l["relation"] for l in links)
    conn = Counter()
    for l in links:
        conn[l["source"]] += 1
        conn[l["target"]] += 1
    label_map = {n["id"]: n["label"] for n in nodes}

    print("\n" + "=" * 62)
    print("  ARACHNE SCHOLAR — NLP GRAPH BUILD v2 COMPLETATO")
    print("=" * 62)
    print(f"  Nodi:   {len(nodes)}")
    for t, c in type_counts.most_common():
        print(f"    - {t:14} {c}")
    print(f"  Archi:  {len(links)}")
    print("  Relazioni principali:")
    for r, c in rel_counts.most_common(15):
        print(f"    - {r:16} {c}")
    print("  Nodi più connessi:")
    for nid, c in conn.most_common(10):
        print(f"    - {label_map.get(nid, nid)[:45]:45} {c}")
    print(f"  Output: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
    print("=" * 62)


if __name__ == "__main__":
    main()
