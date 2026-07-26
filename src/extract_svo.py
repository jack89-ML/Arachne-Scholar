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

# --- LINGUA & MODELLO (v1.2 multilingua EN/IT/ES) ------------------------------
_lang_arg = sys.argv[3].lower() if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else "en"
LANG_MODELS = {
    "en": "en_core_web_trf",
    "it": "it_core_news_lg",
    "es": "es_core_news_lg",
}
LANG = _lang_arg if _lang_arg in LANG_MODELS else "en"
MODEL_NAME = LANG_MODELS[LANG]

# --- GPU SETUP ----------------------------------------------------------------
try:
    spacy.require_gpu()
    GPU_ACTIVE = True
except Exception as e:
    GPU_ACTIVE = False
    print(f"[warn] GPU non disponibile ({e}), fallback CPU", file=sys.stderr)

# --- AUTO-DOWNLOAD & LOAD ------------------------------------------------------
try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    print(f"[setup] modello {MODEL_NAME} mancante, download automatico...", file=sys.stderr)
    from spacy.cli import download as spacy_download
    spacy_download(MODEL_NAME)
    nlp = spacy.load(MODEL_NAME)

nlp.max_length = 6_000_000
print(f"[setup] lang={LANG} model={MODEL_NAME} gpu={GPU_ACTIVE}", file=sys.stderr)

# --- STOPWORDS & FILTRI --------------------------------------------------------
GENERIC_HEADS_DICT = {
    "en": {
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
},
    "it": {
        "teoria", "modello", "concetto", "metodo", "approccio", "analisi",
        "studio", "risultato", "risultati", "dati", "dato", "articolo",
        "capitolo", "libro", "tabella", "figura", "sezione", "processo",
        "sistema", "lavoro", "ricerca", "campo", "caso", "esempio",
        "domanda", "problema", "questione", "fatto", "ragione", "modo",
        "parte", "tipo", "forma", "livello", "punto", "tempo", "anno",
        "anni", "numero", "persone", "gruppo", "gruppi", "cosa", "cose",
        "uso", "ruolo", "effetto", "effetti", "impatto", "differenza",
        "differenze", "cambiamento", "cambiamenti", "relazione",
        "relazioni", "variabile", "variabili", "fattore", "fattori",
        "aspetto", "aspetti", "elemento", "elementi", "caratteristica",
        "caratteristiche", "contesto", "contesti", "dimensione",
        "dimensioni", "meccanismo", "meccanismi", "struttura", "strutture",
        "esito", "esiti", "evidenza", "letteratura", "rassegna",
        "introduzione", "conclusione", "discussione", "riassunto",
        "panoramica", "quadro", "prospettiva",
    },
    "es": {
        "teoría", "modelo", "concepto", "método", "enfoque", "análisis",
        "estudio", "resultado", "resultados", "datos", "dato", "artículo",
        "capítulo", "libro", "tabla", "figura", "sección", "proceso",
        "sistema", "trabajo", "investigación", "campo", "caso", "ejemplo",
        "pregunta", "problema", "cuestión", "hecho", "razón", "manera",
        "parte", "tipo", "forma", "nivel", "punto", "tiempo", "año",
        "años", "número", "personas", "grupo", "grupos", "cosa", "cosas",
        "uso", "rol", "papel", "efecto", "efectos", "impacto",
        "diferencia", "diferencias", "cambio", "cambios", "relación",
        "relaciones", "variable", "variables", "factor", "factores",
        "aspecto", "aspectos", "elemento", "elementos", "característica",
        "características", "contexto", "contextos", "dimensión",
        "dimensiones", "mecanismo", "mecanismos", "estructura",
        "estructuras", "patrón", "patrones", "evidencia", "literatura",
        "revisión", "introducción", "conclusión", "discusión", "resumen",
        "panorama", "marco", "perspectiva",
    },
}
GENERIC_HEADS = GENERIC_HEADS_DICT.get(LANG, GENERIC_HEADS_DICT["en"])

DETERMINERS = {"the", "a", "an", "this", "these", "that", "those", "some",
               "any", "each", "other", "another", "such", "all", "both",
               "several", "many", "most", "more", "few", "little", "much",
               "no", "not", "very", "quite", "rather"}

POSSESSIVES = {"his", "her", "their", "its", "our", "my", "your", "whose"}

JUNK_TOKENS = {"br", "p", "pp", "ed", "eds", "vol", "no", "cf", "ibid",
               "etc", "ie", "eg", "al", "et", "ff", "fn", "n", "nd"}

# Pronomi/question-words che il parser cattura come chunk -> nodi junk (fix v2.1)
JUNK_ENTITIES_DICT = {
    "en": {
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
},
    "it": {
        "che", "cui", "chi", "quale", "quali", "cosa", "ciò", "quello",
        "quella", "questo", "questa", "questi", "queste", "loro", "essi",
        "esse", "egli", "ella", "esso", "essa", "noi", "voi", "io", "me",
        "mio", "mia", "miei", "mie", "tuo", "tua", "suoi", "sue", "suo",
        "sua", "nostro", "nostra", "vostro", "vostra", "qualcuno",
        "qualcosa", "tutto", "niente", "nulla", "chiunque", "ognuno",
        "tutti", "nessuno", "alcuno", "ciascuno", "entrambi", "entrambe",
        "ogni", "alcuni", "alcune", "molti", "molte", "molto", "molta",
        "più", "meno", "parecchi", "parecchie", "diversi", "diverse",
        "altri", "altre", "altro", "altra", "tale", "tali", "stesso",
        "stessa", "medesimo", "qui", "qua", "lì", "là", "dove", "quando",
        "perché", "come", "ovunque", "dovunque", "comunque", "qualunque",
        "qualsiasi", "ne", "ci", "vi", "si",
    },
    "es": {
        "que", "cual", "cuál", "cuales", "quién", "quienes", "qué",
        "este", "esta", "esto", "estos", "estas", "ese", "esa", "eso",
        "esos", "esas", "aquel", "aquella", "aquello", "ellos", "ellas",
        "él", "ella", "nosotros", "nosotras", "vosotros", "yo", "mí",
        "me", "mío", "mía", "tuyo", "tuya", "suyo", "suya", "su", "sus",
        "nuestro", "nuestra", "vuestro", "alguien", "algo", "todo",
        "nada", "nadie", "cualquiera", "quienquiera", "todos", "todas",
        "cada", "ambos", "ambas", "algunos", "algunas", "muchos",
        "muchas", "mucho", "mucha", "más", "menos", "varios", "varias",
        "otros", "otras", "otro", "otra", "tal", "tales", "mismo",
        "misma", "aquí", "ahí", "allí", "donde", "dónde", "cuando",
        "cuándo", "cómo", "como", "dondequiera",
    },
}
JUNK_ENTITIES = JUNK_ENTITIES_DICT.get(LANG, JUNK_ENTITIES_DICT["en"])

WEAK_VERBS_DICT = {
    "en": {"be", "have", "do", "say", "get", "make", "take", "see",
              "know", "think", "go", "come", "give", "tell", "call", "seem",
              "appear", "become", "remain", "stay", "begin", "start", "end",
              "put", "set", "let", "keep", "help", "try", "need", "want",
              "like", "feel", "look", "sound", "mean", "believe", "note",
              "mention", "refer", "cite", "quote", "state", "report",
              "according", "following", "based", "using", "used"},
    "it": {
        "essere", "avere", "fare", "dire", "ottenere", "prendere",
        "vedere", "sapere", "conoscere", "pensare", "andare", "venire",
        "dare", "raccontare", "chiamare", "sembrare", "apparire",
        "diventare", "rimanere", "restare", "iniziare", "cominciare",
        "finire", "mettere", "porre", "lasciare", "tenere", "aiutare",
        "provare", "tentare", "bisognare", "volere", "dovere", "potere",
        "piacere", "sentire", "guardare", "suonare", "significare",
        "credere", "notare", "menzionare", "riferire", "citare",
        "dichiarare", "riportare", "affermare", "seguire", "basare",
        "usare",
    },
    "es": {
        "ser", "estar", "haber", "hacer", "decir", "obtener", "conseguir",
        "tomar", "ver", "saber", "conocer", "pensar", "ir", "venir",
        "dar", "contar", "llamar", "parecer", "aparecer", "volverse",
        "quedarse", "permanecer", "empezar", "comenzar", "terminar",
        "poner", "dejar", "mantener", "ayudar", "intentar", "necesitar",
        "querer", "deber", "poder", "gustar", "sentir", "mirar", "sonar",
        "significar", "creer", "notar", "mencionar", "referir", "citar",
        "declarar", "informar", "afirmar", "seguir", "basar", "usar",
    },
}
WEAK_VERBS = WEAK_VERBS_DICT.get(LANG, WEAK_VERBS_DICT["en"])

RELATION_VERBS_DICT = {
    "en": {
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
},
    "it": {
        "analizzare", "definire", "spiegare", "descrivere", "esaminare",
        "studiare", "investigare", "esplorare", "sviluppare", "proporre",
        "introdurre", "applicare", "impiegare", "testare", "verificare",
        "misurare", "confrontare", "valutare", "dimostrare", "mostrare",
        "rivelare", "trovare", "identificare", "sostenere", "argomentare",
        "suggerire", "supportare", "sfidare", "criticare", "estendere",
        "costruire", "prevedere", "influenzare", "incidere",
        "condizionare", "determinare", "causare", "produrre", "generare",
        "creare", "formare", "costituire", "comprendere", "includere",
        "contenere", "coinvolgere", "richiedere", "correlare",
        "collegare", "associare", "dipendere", "derivare", "emergere",
        "sorgere", "condurre", "portare", "contribuire", "mediare",
        "moderare", "sottendere", "inquadrare", "concettualizzare",
        "teorizzare", "formalizzare", "operazionalizzare", "validare",
        "replicare", "confermare", "confutare", "respingere", "assumere",
        "ipotizzare", "postulare", "affrontare", "focalizzare",
        "concentrare", "centrare", "trattare", "coprire", "discutere",
        "considerare", "riguardare", "interpretare", "capire",
        "trasformare", "cambiare", "alterare", "modificare",
        "strutturare", "organizzare", "integrare", "combinare", "unire",
        "unificare", "separare", "distinguere", "differenziare",
        "classificare", "categorizzare", "raggruppare",
    },
    "es": {
        "analizar", "definir", "explicar", "describir", "examinar",
        "estudiar", "investigar", "explorar", "desarrollar", "proponer",
        "introducir", "aplicar", "emplear", "probar", "medir",
        "comparar", "contrastar", "evaluar", "demostrar", "mostrar",
        "revelar", "encontrar", "identificar", "argumentar", "afirmar",
        "sugerir", "apoyar", "sostener", "desafiar", "criticar",
        "extender", "construir", "predecir", "influir", "influenciar",
        "afectar", "condicionar", "determinar", "causar", "producir",
        "generar", "crear", "formar", "constituir", "comprender",
        "incluir", "contener", "involucrar", "requerir", "relacionar",
        "conectar", "vincular", "asociar", "correlacionar", "depender",
        "derivar", "emerger", "surgir", "conducir", "llevar",
        "contribuir", "mediar", "moderar", "subyacer", "enmarcar",
        "conceptualizar", "teorizar", "formalizar", "operacionalizar",
        "validar", "replicar", "confirmar", "refutar", "rechazar",
        "asumir", "hipotetizar", "postular", "abordar", "enfocar",
        "concentrar", "centrar", "tratar", "cubrir", "discutir",
        "considerar", "interpretar", "entender", "transformar",
        "cambiar", "alterar", "modificar", "estructurar", "organizar",
        "integrar", "combinar", "fusionar", "unificar", "separar",
        "distinguir", "diferenciar", "clasificar", "categorizar",
        "agrupar",
    },
}
RELATION_VERBS = RELATION_VERBS_DICT.get(LANG, RELATION_VERBS_DICT["en"])

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
    """(v1.6) Normalizzazione canonicizzata: lowercased, strip di spaziature
    PDF, rimozione artefatti markdown, suffissi deboli (-era, -related...) e
    collasso di whitespace. La forma prodotta e' il riferimento unico per
    tutti i node_id: due varianti morfologiche che canonizzano alla stessa
    stringa generano un solo nodo (es. 'Trump-era' -> 'trump')."""
    t = text.strip()
    # rimuovi caratteri markdown/detriti di conversione PDF
    t = re.sub(r"[_*#`~\\\[\](){}<>]", " ", t)
    t = re.sub(r"[^a-z0-9\s\-']", " ", t.lower())
    t = re.sub(r"\s+", " ", t).strip()
    # (FIX 2-3) Suffix-stripping canonico: normalizza -era, -based, ecc.
    # alla root per fondere nodi frammentati. Ordine: suffissi piu' lunghi
    # prima per evitare matching parziali.
    for suffix in ["-oriented", "-induced", "-related", "-driven",
                    "-specific", "-esque", "-style", "-based", "-like",
                    "-era"]:
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            t = t[:-len(suffix)]
            break  # un solo suffix match per label
    return t.strip(" -'")


def node_id_from(text):
    nid = re.sub(r"[^a-z0-9\s-]", "", text.lower())
    nid = re.sub(r"[\s_]+", "-", nid).strip("-")
    return nid[:80]


def clean_chunk_label(chunk):
    """
    Pulisce il label del chunk: rimuove determinanti, possessivi,
    tokens junk, e normalizza la forma. (v1.6) Strip aggressivo
    di whitespace/spaziature PDF da inizio/fine label.
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
        txt = re.sub(r"[_*#`~\\[\\](){}<>]", "", txt)
        if not txt or not re.search(r"[a-zA-Z]", txt):
            continue
        tokens.append(txt.strip())
    label = " ".join(tokens).strip()
    # collassa spazi e trattini; aggiuntivo strip contro residui di conversione
    label = re.sub(r"\s+-\s+", "-", label)
    label = re.sub(r"\s+", " ", label).strip("\"'` \t\n\r\f\v-–—")
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


def extract_entities(doc, entities, freq=None):
    """Estrae entità da doc spaCy e le aggiunge al dict condiviso.
    Se freq (Counter) e' passato, conta le occorrenze come
    'n. di frasi che menzionano l'entità' (una per frase, evitando il
    doppio conteggio NER+noun-chunk sulla stessa menzione)."""
    counted = set()

    def _bump(sent_start, nid):
        if freq is None or not nid:
            return
        key = (sent_start, nid)
        if key not in counted:
            counted.add(key)
            freq[nid] += 1

    # 1) NER
    for ent in doc.ents:
        label = clean_chunk_label(ent)
        norm = normalize(label)
        if not is_valid_entity(label, norm):
            continue
        ntype = NER_TYPE_MAP.get(ent.label_, "concept")
        nid = node_id_from(norm)
        _bump(ent.sent.start, nid)
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
        if not nid:
            continue
        _bump(chunk.sent.start, nid)
        if nid in entities:
            continue
        ntype = classify(label, lem)
        entities[nid] = {
            "id": nid,
            "label": label[:80],
            "type": ntype,
            "description": f"{ntype} (chunk, lemma:{lem})",
        }
    return entities


STRIP_CHARS = " \t\n\r-–—_‐‑‒'\"`’‘“”.,;:!?()[]{}<>|/\\*#@~^="


def sanitize_label(text):
    """(SANITIZE v1.5) Pulizia label: strip di punteggiatura orfana, trattini,
    apici e caratteri spuri all'inizio/fine; collassa spazi multipli."""
    t = text.strip(STRIP_CHARS)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_keepable_label(label):
    """Tieni label >= 3 caratteri; sotto i 2 solo se acronimo maiuscolo (AI, EU...)."""
    if not label:
        return False
    if len(label) >= 3:
        return True
    return label.isupper() and label.replace(" ", "").isalpha()


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
                if dep in ("nsubj", "nsubjpass", "nsubj:pass", "csubj", "csubjpass", "agent", "expl"):
                    nid = nid_for_span(child, entities)
                    if nid:
                        subj_nid = nid
                elif dep in ("dobj", "obj", "iobj", "obl", "attr", "acomp", "oprd", "dative"):
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
                        if gc.dep_ in ("dobj", "obj", "attr"):
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


def extract_cowindow_edges(doc, entities, window=5, max_per_window=12):
    """(A) Co-occorrenza a finestra scorrevole di `window` frasi.
    Approssima la finestra-a-paragrafo del vecchio build_local_graph.py,
    mantenendo il matching NLP (stessa pulizia chunk di extract_cochunk_edges).
    max_per_window argina l'esplosione combinatoria sulle frasi dense."""
    edges = Counter()
    sent_nids = []
    for sent in doc.sents:
        nids, seen = [], set()
        for chunk in sent.noun_chunks:
            label = clean_chunk_label(chunk)
            norm = normalize(label)
            nid = node_id_from(norm)
            if nid in entities and nid not in seen:
                seen.add(nid)
                nids.append(nid)
        sent_nids.append(nids)
    for i in range(len(sent_nids)):
        pool, seen_pool = [], set()
        for j in range(i, min(i + window, len(sent_nids))):
            for nid in sent_nids[j]:
                if nid not in seen_pool:
                    seen_pool.add(nid)
                    pool.append(nid)
        pool = pool[:max_per_window]
        for ai in range(len(pool)):
            for bi in range(ai + 1, len(pool)):
                a, b = sorted([pool[ai], pool[bi]])
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


def process_file(path, nlp, window_size=5):
    with open(path, encoding="utf-8") as f:
        text = f.read()

    entities, svo_edges, co_edges = {}, Counter(), Counter()
    freq = Counter()
    pieces = chunk_text(text)
    t0 = time.time()

    for i, piece in enumerate(pieces):
        for doc in nlp.pipe([piece], batch_size=32):
            entities = extract_entities(doc, entities, freq)
            svo_edges.update(extract_svo_edges(doc, entities))
            co_edges.update(extract_cowindow_edges(doc, entities, window=window_size))
        elapsed = time.time() - t0
        print(f"    [{i+1}/{len(pieces)}] {len(entities)} entità, "
              f"{len(svo_edges)} SVO, {elapsed:.0f}s", file=sys.stderr)

    return entities, svo_edges, co_edges, freq


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Arachne Scholar — SVO graph extractor (multilingua)")
    parser.add_argument("md_dir", nargs="?", default=os.path.expanduser("~/scholar_engine/converted_md"))
    parser.add_argument("out_dir", nargs="?", default=os.path.expanduser("~/scholar_engine/graph_out"))
    parser.add_argument("lang", nargs="?", default=LANG)
    parser.add_argument("--co-threshold", type=int, default=2,
                        help="min co-occorrenze per arco co_occurs (default 2; era 5)")
    parser.add_argument("--min-freq", type=int, default=2,
                        help="frequenza corpus minima per tenere un nodo senza legami (default 2)")
    parser.add_argument("--window-size", type=int, default=5,
                        help="ampiezza finestra co-occorrenza in frasi (default 5)")
    args = parser.parse_args()
    md_dir, out_dir = args.md_dir, args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    md_files = sorted(glob.glob(os.path.join(md_dir, "*.md")))
    if not md_files:
        print("Nessun markdown trovato.", file=sys.stderr)
        sys.exit(1)

    all_entities, all_svo, all_co, all_freq = {}, Counter(), Counter(), Counter()

    for fp in md_files:
        print(f"\n=== {os.path.basename(fp)[:70]} ===", file=sys.stderr)
        ents, svo, co, fq = process_file(fp, nlp, window_size=args.window_size)
        all_entities.update(ents)
        all_svo.update(svo)
        all_co.update(co)
        all_freq.update(fq)

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

    # (B) co_occur: soglia co_threshold (default 2, come il vecchio prototipo),
    # peso calibrato min(c,5), confidence graduata extracted>=3 / inferred=2
    for (src, tgt, rel), count in all_co.most_common():
        if count < args.co_threshold:
            break
        if (src, tgt, rel) in seen:
            continue
        seen.add((src, tgt, rel))
        links.append({
            "source": src, "target": tgt, "relation": rel,
            "weight": min(count, 5),
            "confidence": "extracted" if count >= 3 else "inferred",
        })

    # (C) FILTRO HAPAX: scarta nodi menzionati in meno di min_freq frasi E zero legami.
    # Taglia la coda lunga di entita' citate una sola volta (i nodi isolati).
    connected = set()
    for l in links:
        connected.add(l["source"])
        connected.add(l["target"])
    before = len(all_entities)
    kept = {nid: n for nid, n in all_entities.items()
            if all_freq.get(nid, 0) >= args.min_freq or nid in connected}
    removed = before - len(kept)
    if removed > 0:
        links = [l for l in links if l["source"] in kept and l["target"] in kept]
    nodes = list(kept.values())
    print(f"  [hapax-filter] rimossi {removed}/{before} nodi "
          f"(freq<{args.min_freq}, zero legami)", file=sys.stderr)

    # (SANITIZE v1.5) Filtro finale: pulizia label nodi e relazioni,
    # scarto entita' spurie (<2 char non acronimi) e archi orfani.
    pre_sanitize = len(nodes)
    clean_nodes = []
    for n in nodes:
        lbl = sanitize_label(n["label"])
        if not is_keepable_label(lbl):
            continue
        n["label"] = lbl
        clean_nodes.append(n)
    kept_ids = {n["id"] for n in clean_nodes}
    links = [l for l in links if l["source"] in kept_ids and l["target"] in kept_ids]
    for l in links:
        l["relation"] = sanitize_label(l["relation"])
    nodes = clean_nodes
    if pre_sanitize - len(nodes):
        print(f"  [sanitize] rimossi {pre_sanitize - len(nodes)} nodi spuri "
              f"(label non valide)", file=sys.stderr)

    graph = {
        "nodes": nodes,
        "links": links,
        "edges": links,
        "meta": {"model": MODEL_NAME, "gpu": GPU_ACTIVE, "lang": LANG,
                 "co_threshold": args.co_threshold, "min_freq": args.min_freq,
                 "window_size": args.window_size, "hapax_removed": removed,
                 "engine": "spacy trf: NER + noun chunks + dep parsing + sliding-window co-occurrence"},
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
