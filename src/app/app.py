# -*- coding: utf-8 -*-
"""
App locale per l'anonimizzazione reversibile di documenti con il modello PII.

Flusso d'uso:
  1) ANONIMIZZA  - incolli testo o carichi un PDF; il modello + una rete regex/checksum
     trovano le PII. Ogni entita' riceve un ID univoco e reversibile: [FULLNAME_1],
     [IBAN_1], ... (valori uguali condividono lo stesso ID).
  2) COPIA       - copi il testo anonimizzato e lo incolli in ChatGPT/altro LLM.
  3) RIPRISTINA  - incolli la risposta dell'LLM (che contiene i placeholder) e l'app
     rimette i valori veri usando il dizionario locale.

Tutto in locale: il testo e il dizionario {placeholder -> valore} non lasciano la macchina.

Il modello e' affiancato da una rete REGEX + CHECKSUM (EMAIL, TELEFONO, IBAN, CF, PIVA,
carta di credito, importi, targhe, URL, IP). Le entita' validate matematicamente (IBAN/CF/
PIVA/carta) hanno priorita' sul modello in caso di sovrapposizione.

Il dizionario di ripristino si puo' DISATTIVARE (switch nell'UI, --no-mapping, PII_MAPPING=0):
l'anonimizzazione diventa definitiva, nessuna chiave placeholder->valore viene costruita.

Endpoint HTTP:
  GET  /health    liveness/readiness senza inference (200 = modello caricato, 503 = no)
  POST /analyze   {"text": ...} oppure multipart con un file (.pdf/.md/.txt); campi
                  opzionali "exclude_tags" e "include_mapping" per override per-richiesta
  POST /pdf       stesso input di /analyze -> scarica il PDF ANONIMIZZATO (redazione
                  vera del PDF caricato, oppure PDF ricostruito dal testo)
  POST /preview   multipart con un .pdf -> lo tiene in memoria per l'ANTEPRIMA a video e
                  ritorna {doc_id, n_pages, text}
  POST /pdf/preview  come /pdf, ma invece del binario ritorna {doc_id, n_pages, ...}:
                  il PDF anonimizzato resta in memoria e si guarda pagina per pagina
  GET  /doc/<id>/page/<n>.png   pagina renderizzata (anteprima a video)
  GET  /doc/<id>/file.pdf       download del documento tenuto in memoria
  GET  /settings  legenda dei 23 tag, tag esclusi, stato del dizionario reversibile,
                  stato dell'OCR (motore installato, switch automatico)
  POST /settings  {"excluded_tags": [...], "mapping_enabled": bool, "ocr_enabled": bool}
                  (/tags e' un alias storico degli stessi due endpoint)
  GET  /config, POST /config, GET /port-check   host/porta del server

OCR: quando il PDF caricato NON ha layer testuale (una scansione), l'app lo
riconosce da sola e passa la pagina al motore OCR (Tesseract o RapidOCR/ONNX,
vedi src/app/ocr.py): il testo riconosciuto entra nell'analisi come qualsiasi
altro testo, e il PDF viene redatto in "burn-in" (le caselle vengono annerite
DENTRO i pixel, cosi' la PII non resta recuperabile nel file). Lo switch "OCR
automatico" lo disattiva; senza alcun motore installato l'app funziona come
prima e dice cosa installare.

Avvio:  python app.py   ->   http://127.0.0.1:5005
Configurazione host/porta (precedenza): CLI --host/--port > env PII_HOST/PII_PORT >
  config.json (vedi server_config.py) > default 127.0.0.1:5005
Preferenze di anonimizzazione (precedenza): campo nella richiesta > CLI --exclude-tags/
  --no-mapping > env PII_EXCLUDE_TAGS/PII_MAPPING > prefs.json > default
"""

import bisect
import hashlib
import os
import re
import secrets
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import fitz  # PyMuPDF
import torch
from flask import (Flask, jsonify, render_template_string, request,
                   send_from_directory)

import ocr
import pdf_export
import pdf_text
import server_config
# Rete REGEX + CHECKSUM: modulo a parte, senza dipendenze dal modello. I nomi
# restano importabili da qui (`app.detect_regex`) per non rompere chi li usa.
from detectors import (DETECTORS, SOFT_REGEX_LABELS, cf_ok,  # noqa: F401
                       complete_time, detect_iban, detect_regex, iban_ok, luhn_ok,
                       piva_ok)
from transformers import pipeline


def _resource_path(rel):
    """Percorso risorsa valido sia in sviluppo sia dentro l'exe PyInstaller."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


# MODELLO usato dall'app. Accetta tre forme, provate in quest'ordine:
#   "1.2.0"  -> models/rizzo-pii-0.3B-v1.2.0/   (versione di un run di training)
#   "main"   -> models/rizzo-pii-0.3B-main/     (revision scaricata da HF, senza numero)
#   percorso -> usato tal quale (assoluto o relativo alla root della repo)
# Metti None per usare AUTOMATICAMENTE l'ultima versione `-v*` disponibile.
APP_MODEL_VERSION = "1.5.0"

# Dentro l'exe il modello e' impacchettato come "pii_model" (vedi build.spec).
# In sviluppo: pin sopra -> auto-ultima versione -> vecchio non versionato -> legacy.
# Override puntuale a runtime: env PII_MODEL_DIR.
if getattr(sys, "_MEIPASS", None):
    MODEL_DIR = _resource_path("pii_model")
elif os.environ.get("PII_MODEL_DIR"):
    MODEL_DIR = os.environ["PII_MODEL_DIR"]
else:
    import re
    _models = Path(__file__).resolve().parents[2] / "models"
    _cands = []
    if APP_MODEL_VERSION:
        _cands = [_models / f"rizzo-pii-0.3B-v{APP_MODEL_VERSION}",
                  _models / f"rizzo-pii-0.3B-{APP_MODEL_VERSION}"]
        if os.sep in APP_MODEL_VERSION or "/" in APP_MODEL_VERSION:   # e' un percorso
            _cands += [Path(APP_MODEL_VERSION), _models.parent / APP_MODEL_VERSION]
    _pinned = next((p for p in _cands if p.is_dir()), None)
    _versioned = [p for p in _models.glob("rizzo-pii-0.3B-v*") if p.is_dir()]
    if _pinned:
        MODEL_DIR = str(_pinned)
    elif _versioned:
        MODEL_DIR = str(max(_versioned, key=lambda p: tuple(
            int(x) for x in re.search(r"-v([0-9][0-9.]*)$", p.name).group(1).split("."))))
    else:
        _prod = _models / "rizzo-pii-0.3B"
        MODEL_DIR = str(_prod if _prod.exists() else _models / "pii_model_legacy")

# Se la cartella non c'e', dirlo QUI e con il comando giusto: senza questo controllo si
# arriva a from_pretrained() con un path inventato (pii_model_legacy, ultimo fallback) e
# l'errore parla di una cartella che l'utente non ha mai sentito nominare.
if not getattr(sys, "_MEIPASS", None) and not os.path.isdir(MODEL_DIR):
    _v = APP_MODEL_VERSION or "1.5.0"
    print(f"ERRORE: modello non trovato in {MODEL_DIR}\n"
          f"Scaricalo con (la revision e la cartella devono combaciare):\n"
          f"  hf download rizzoaiacademy/rizzo-pii-0.3B --revision v{_v} "
          f"--local-dir models/rizzo-pii-0.3B-v{_v}\n"
          f"Oppure indica una cartella tua con la variabile PII_MODEL_DIR.",
          file=sys.stderr)
    sys.exit(2)

ASSETS_DIR = _resource_path("assets")   # mascotte / icone (servite su /assets/<file>)
APP_VERSION = "2.0.0"                    # versione mostrata nell'UI (allineata a tauri.conf.json)
MAX_WORDS = 120      # parole per chunk (~180 subword, sotto i 512 del training)
OVERLAP = 20         # parole di sovrapposizione tra chunk consecutivi

# --------------------------------------------------------------------------- #
# Caricamento modello (una sola volta all'avvio)
# --------------------------------------------------------------------------- #
device = 0 if torch.cuda.is_available() else -1
print(f"Carico il modello da {MODEL_DIR} su {'GPU' if device == 0 else 'CPU'}...")
try:
    nlp = pipeline(
        "token-classification",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        aggregation_strategy="simple",
        device=device,
    )
except Exception as e:
    # La cartella del modello puo' ESISTERE ed essere incompleta: e' quello che resta
    # quando il download viene interrotto, oppure quando i symlink del cache HF (che
    # puntano a ../../blobs/) restano appesi perche' la cartella blobs/ e' stata
    # cancellata. transformers, in quel caso, non dice "manca config.json" ma
    # "Unrecognized model ... Should have a `model_type` key in its config.json": un
    # messaggio che sembra accusare il FORMATO del modello e invece segnala un file
    # che non c'e'. Qui si distingue la cartella incompleta dall'errore vero e proprio.
    _v = APP_MODEL_VERSION or "1.5.0"
    if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        print(f"ERRORE: modello incompleto in {MODEL_DIR}: config.json manca o non e' "
              f"leggibile (download interrotto, oppure symlink rimasti appesi: i "
              f"symlink del cache HF puntano a ../../blobs/, se blobs/ e' sparito il "
              f"modello non c'e' piu').\n"
              f"Riscaria il modello per intero:\n"
              f"  rm -rf {MODEL_DIR}\n"
              f"  hf download rizzoaiacademy/rizzo-pii-0.3B --revision v{_v} "
              f"--local-dir models/rizzo-pii-0.3B-v{_v}\n"
              f"Oppure indica una cartella tua con la variabile PII_MODEL_DIR.",
              file=sys.stderr)
    else:
        print(f"ERRORE: caricamento del modello fallito da {MODEL_DIR}: {e}\n"
              f"Verifica che la cartella contenga config.json, tokenizer.json e "
              f"model.safetensors.", file=sys.stderr)
    sys.exit(2)
print("Modello pronto.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Estensioni accettate dall'upload (il PDF passa da PyMuPDF, il resto e' testo puro).
TEXT_EXTS = {".md", ".markdown", ".txt", ".text"}

# --------------------------------------------------------------------------- #
# Anteprima a video dei PDF (documento caricato a sinistra, anonimizzato a destra)
#
# I PDF vengono renderizzati QUI, con PyMuPDF, e serviti come PNG per pagina: non
# serve un viewer PDF nel browser (WebView2/WKWebView dentro Tauri non lo hanno in
# modo affidabile) ne' una libreria JS esterna (l'app e' offline, niente CDN).
#
# I documenti restano in memoria (mai su disco: valgono quanto il documento stesso)
# in una LRU piccola, e muoiono con il processo. Il render di una pagina e' pigro e
# poi tenuto in cache: aprire un PDF di 200 pagine non costa 200 render.
# --------------------------------------------------------------------------- #
PREVIEW_DPI = 110          # ~1150 px di larghezza su un A4: leggibile senza pesare
MAX_DOCS = 6               # documenti tenuti in memoria (LRU, i piu' vecchi cadono)

_DOCS = OrderedDict()      # doc_id -> {"pdf": bytes, "pages": {n: png}, "n_pages", "name"}
_DOCS_LOCK = threading.Lock()


def _safe_name(name, default="documento.pdf"):
    """Nome file pulito per Content-Disposition (ASCII, niente separatori)."""
    base = os.path.basename(name or "").strip()
    base = re.sub(r"[^A-Za-z0-9._\- ]+", "_", base).strip(" ._")
    return base or default


def _store_doc(data, name="documento.pdf"):
    """Mette un PDF nello store dell'anteprima. Ritorna (doc_id, n_pages)."""
    with fitz.open(stream=data, filetype="pdf") as doc:
        n_pages = doc.page_count
    doc_id = secrets.token_urlsafe(12)
    with _DOCS_LOCK:
        _DOCS[doc_id] = {"pdf": data, "pages": {}, "n_pages": n_pages,
                         "name": _safe_name(name)}
        while len(_DOCS) > MAX_DOCS:
            _DOCS.popitem(last=False)
    return doc_id, n_pages


def _get_doc(doc_id):
    with _DOCS_LOCK:
        d = _DOCS.get(doc_id)
        if d is not None:
            _DOCS.move_to_end(doc_id)          # LRU: l'uso lo tiene in vita
    return d


def _page_png(d, n, dpi=PREVIEW_DPI):
    """PNG della pagina n (0-based) del documento, con cache. None se fuori range."""
    if not (0 <= n < d["n_pages"]):
        return None
    key = (n, dpi)
    with _DOCS_LOCK:
        png = d["pages"].get(key)
    if png is None:
        with fitz.open(stream=d["pdf"], filetype="pdf") as doc:
            png = doc.load_page(n).get_pixmap(dpi=dpi).tobytes("png")
        with _DOCS_LOCK:
            d["pages"][key] = png
    return png


# --------------------------------------------------------------------------- #
# Legenda dei tag + tag esclusi dall'anonimizzazione
#
# I 22 tag del modello (docs/TASSONOMIA_TAG.md) + URL e IPADDR, che sono solo-regex:
# il modello non e' stato addestrato su di essi, li trova la rete regex.
# L'utente puo' DESELEZIONARE un tag: le entita' di quel tipo vengono rilevate ma non
# sostituite (restano in chiaro). Serve a chi deve confrontare gli importi (AMOUNT) o
# tenere eta'/sesso in un caso clinico.
# --------------------------------------------------------------------------- #
TAGS = [
    ("FULLNAME", "Nome di persona (anche ruoli legali: giudice, avvocato, parti, teste)",
     "Person name (legal roles included: judge, lawyer, parties, witness)", "Mario Rossi"),
    ("AGE", "Età", "Age", "45 anni"),
    ("GENDER", "Sesso / genere", "Sex / gender", "Femmina"),
    ("DATE", "Data di calendario", "Calendar date", "12/06/1985"),
    ("TIME", "Ora", "Time of day", "ore 15:30"),
    ("STREET", "Via / piazza / corso", "Street / square", "Via Garibaldi"),
    ("BUILDINGNUM", "Numero civico", "Building number", "24"),
    ("ZIPCODE", "CAP", "ZIP / postal code", "00185"),
    ("CITY", "Città", "City", "Milano"),
    ("PROVINCE", "Sigla della provincia", "Province code", "MI"),
    ("EMAIL", "Email, PEC inclusa", "Email, certified mail included", "m.rossi@studio.it"),
    ("TELEPHONENUM", "Numero di telefono", "Phone number", "+39 333 1234567"),
    ("CF", "Codice fiscale (checksum verificato)", "Italian tax code (checksum verified)",
     "RSSMRA85H12F205Y"),
    ("PIVA", "Partita IVA (checksum verificato)", "VAT number (checksum verified)", "12345678903"),
    ("ID_DOC", "Numero di documento d'identità (carta, passaporto, patente)",
     "Identity document number (ID card, passport, driving licence)", "CA12345AB"),
    ("IBAN", "IBAN / numero di conto (checksum verificato)",
     "IBAN / account number (checksum verified)", "IT60X0542811101000000123456"),
    ("CREDITCARDNUMBER", "Numero di carta di credito (Luhn verificato)",
     "Credit card number (Luhn verified)", "4111 1111 1111 1111"),
    ("AMOUNT", "Importo in denaro", "Money amount", "€ 12.500,00"),
    ("TARGA", "Targa di veicolo", "Vehicle plate", "AB 123 CD"),
    ("ORG", "Ragione sociale privata: società, studio legale, banca",
     "Private organization: company, law firm, bank", "Edilnord S.r.l."),
    ("DOCID", "Codice di un atto: ruolo generale, protocollo, repertorio, sentenza",
     "Document code: case number, protocol, repertoire, judgment", "1234/2024"),
    ("CATASTO", "Dati catastali: foglio, particella, subalterno",
     "Land registry data: sheet, parcel, subordinate", "Foglio 12, part. 345, sub. 6"),
    ("URL", "Indirizzo web (rilevato solo dalla rete regex, non dal modello)",
     "Web address (regex net only, not from the model)", "https://www.studiorossi.it"),
    ("IPADDR", "Indirizzo IP, anche subnet CIDR o range (rilevato solo dalla rete regex)",
     "IP address, including CIDR subnet or range (regex net only)", "192.168.1.1"),
]
TAG_NAMES = [t[0] for t in TAGS]

# Preferenze di default del server (env > prefs.json). Gli argomenti CLI le sovrascrivono
# all'avvio; ogni singola richiesta puo' comunque passare le proprie.
#
# MAPPING_ENABLED = False -> l'anonimizzazione e' DEFINITIVA: nessun dizionario
# placeholder->valore viene costruito, restituito o salvato, e la risposta non contiene
# piu' il testo originale delle entita' (nemmeno dentro `segments`). Serve a chi non
# vuole che una chiave di ripristino esista, da nessuna parte.
_prefs = server_config.load_prefs()
EXCLUDED_TAGS = _prefs["excluded_tags"]
MAPPING_ENABLED = _prefs["mapping_enabled"]
OCR_ENABLED = _prefs.get("ocr_enabled", True)   # OCR automatico sulle scansioni
_PREFS_LOCK = threading.Lock()   # protegge la lettura/scrittura ATOMICA della coppia sopra


def _get_prefs():
    """Snapshot coerente di (excluded_tags, mapping_enabled): mai una POST /settings a meta'."""
    with _PREFS_LOCK:
        return EXCLUDED_TAGS, MAPPING_ENABLED


def _get_ocr_enabled():
    """Stato dello switch "OCR automatico" (con lo stesso lock delle altre preferenze)."""
    with _PREFS_LOCK:
        return OCR_ENABLED


# --------------------------------------------------------------------------- #
# Chunking word-safe + inferenza del modello su tutto il documento
# --------------------------------------------------------------------------- #
def chunk_text(text, max_words=MAX_WORDS, overlap=OVERLAP):
    """Ritorna [(sottostringa, offset_char_globale), ...] senza tagliare parole."""
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, max_words - overlap)
    while i < len(words):
        block = words[i:i + max_words]
        start, end = block[0].start(), block[-1].end()
        chunks.append((text[start:end], start))      # slice esatto -> offset diretti
        if i + max_words >= len(words):
            break
        i += step
    return chunks


def detect_model(text):
    """Entita' trovate dal modello mmBERT su tutti i chunk, su offset globali."""
    chunks = chunk_text(text)
    ents = []
    if chunks:
        results = nlp([c for c, _ in chunks])
        if isinstance(results, dict):                 # singolo chunk -> normalizza
            results = [results]
        for (_, off), res in zip(chunks, results):
            for e in res:
                ents.append({
                    "label": e["entity_group"],
                    "start": int(e["start"]) + off,
                    "end": int(e["end"]) + off,
                    "score": float(e["score"]),
                    "validated": False,
                    "source": "modello",
                })
    return ents, len(chunks)


# --------------------------------------------------------------------------- #
# Fusione modello + regex, ID reversibili, testo anonimizzato
# --------------------------------------------------------------------------- #
def _is_word(ch):
    """Carattere interno a una parola (lettere accentate e cifre incluse)."""
    return ch.isalnum() or ch == "_"


def _merge(cands, text):
    """Greedy senza overlap. Priorita': checksum-valido > regex (non soft) > score
    > lunghezza.
    La rete regex copre campi a forma molto specifica: per quegli span e' piu' affidabile
    del modello (evita la frammentazione di CF/IBAN/carta in piu' pezzi). Fanno eccezione
    i tag in SOFT_REGEX_LABELS, dove la forma non ha un checksum a confermarla."""
    order = sorted(
        cands,
        key=lambda e: (1 if e["validated"] else 0,
                       1 if (e["source"] == "regex"
                             and e["label"] not in SOFT_REGEX_LABELS) else 0,
                       e["score"], e["end"] - e["start"]),
        reverse=True,
    )
    kept = []
    for e in order:
        # kept resta ordinata per start e senza sovrapposizioni: allora un candidato puo'
        # accavallarsi solo con i due vicini, che la ricerca binaria trova subito. Il
        # confronto con TUTTA kept era O(n^2): su un documento con 40.000 entita' (una chat
        # esportata di qualche MB) erano 111 s spesi qui, dopo l'inferenza.
        i = bisect.bisect_right(kept, e["start"], key=lambda k: k["start"])
        if (i and kept[i - 1]["end"] > e["start"]) or \
           (i < len(kept) and kept[i]["start"] < e["end"]):
            continue
        kept.insert(i, e)
    # niente spazi inglobati nei placeholder (il modello a volte include lo spazio iniziale)
    for e in kept:
        while e["start"] < e["end"] and text[e["start"]].isspace():
            e["start"] += 1
        while e["end"] > e["start"] and text[e["end"] - 1].isspace():
            e["end"] -= 1
    kept = [e for e in kept if e["end"] > e["start"]]

    # Allineamento ai confini di parola. Il modello etichetta i sotto-token e a volte ne
    # copre solo una parte ("No" di "Novara"): sostituendo la span cosi' com'e' resterebbe
    # "[CITY_1]vara", cioe' un valore ancora ricostruibile. Se una span taglia una parola
    # a meta', la si estende fino a coprirla. Nel dubbio si maschera un carattere in piu':
    # per un anonimizzatore l'errore per eccesso e' l'unico accettabile.
    for e in kept:
        while (e["start"] > 0
               and _is_word(text[e["start"] - 1]) and _is_word(text[e["start"]])):
            e["start"] -= 1
        while (e["end"] < len(text)
               and _is_word(text[e["end"]]) and _is_word(text[e["end"] - 1])):
            e["end"] += 1

    # L'estensione puo' rendere due span sovrapposte o adiacenti: si fondono, altrimenti
    # una stessa parola verrebbe sostituita due volte ("[CATASTO_1][CATASTO_2]").
    kept.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    merged = []
    for e in kept:
        if merged and e["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
            continue
        if merged and e["start"] == merged[-1]["end"] and e["label"] == merged[-1]["label"]:
            merged[-1]["end"] = e["end"]
            continue
        merged.append(e)
    return merged


def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).casefold()


def analyze(text, excluded=None, mapping_enabled=True):
    """excluded = tag da NON anonimizzare: le entita' di quel tipo vengono scartate
    prima della fusione, quindi il valore resta in chiaro nel testo di output.

    mapping_enabled=False -> anonimizzazione DEFINITIVA: nessun dizionario
    placeholder->valore, e i segmenti-entita' non riportano il testo originale
    (che altrimenti permetterebbe di ricostruire il dizionario dalla risposta).
    La numerazione dei placeholder resta: dice che due occorrenze sono lo stesso
    soggetto, ma da sola non fa risalire al valore."""
    excluded = set(excluded or ())
    model_ents, n_chunks = detect_model(text)
    # il modello taglia i minuti delle ore: la regex dell'ora completa SOLO le sue span TIME
    complete_time(model_ents, text)
    cands = model_ents + detect_regex(text)
    if excluded:
        cands = [e for e in cands if e["label"] not in excluded]
    kept = _merge(cands, text)

    # ID reversibili: stesso (label, valore-normalizzato) -> stesso placeholder.
    counters, seen, mapping = {}, {}, {}
    for e in kept:
        val = text[e["start"]:e["end"]]
        key = (e["label"], _norm(val))
        if key in seen:
            e["ph"] = seen[key]
        else:
            counters[e["label"]] = counters.get(e["label"], 0) + 1
            ph = f"[{e['label']}_{counters[e['label']]}]"
            seen[key] = ph
            if mapping_enabled:
                mapping[ph] = val
            e["ph"] = ph

    # segmenti per la preview + testo anonimizzato + statistiche
    segments, anon, by_label, by_source, pos = [], [], {}, {}, 0
    for e in kept:
        if e["start"] > pos:
            segments.append({"t": text[pos:e["start"]]})
            anon.append(text[pos:e["start"]])
        seg = {
            "label": e["label"],
            "ph": e["ph"],
            "src": e["source"],
            "validated": e["validated"],
        }
        if mapping_enabled:                       # senza dizionario niente valore originale
            seg["t"] = text[e["start"]:e["end"]]
        segments.append(seg)
        anon.append(e["ph"])
        by_label[e["label"]] = by_label.get(e["label"], 0) + 1
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1
        pos = e["end"]
    if pos < len(text):
        segments.append({"t": text[pos:]})
        anon.append(text[pos:])

    return {
        "segments": segments,
        "anonymized_text": "".join(anon),
        "mapping": mapping,
        "mapping_enabled": mapping_enabled,
        "n_chunks": n_chunks,
        "n_chars": len(text),
        "n_entities": len(kept),
        "n_unique": len(seen),
        "by_label": dict(sorted(by_label.items(), key=lambda x: -x[1])),
        "by_source": by_source,
        "excluded_tags": sorted(excluded),
        "_raw_entities": cands,  # entità prima della fusione (per fusione OCR)
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
def _page():
    return PAGE.replace("__VERSION__", APP_VERSION)


@app.route("/")
def index():
    return _page()


@app.route("/assets/<path:fn>")
def assets(fn):
    if os.path.isfile(os.path.join(ASSETS_DIR, fn)):
        return send_from_directory(ASSETS_DIR, fn)
    return ("", 404)


@app.route("/favicon.ico")
def favicon():
    if os.path.isfile(os.path.join(ASSETS_DIR, "mascot_shield.png")):
        return send_from_directory(ASSETS_DIR, "mascot_shield.png")
    return ("", 204)


@app.errorhandler(404)
def not_found(_e):
    return _page()


@app.route("/health")
@app.route("/healthz")
def health():
    """Liveness/readiness SENZA inference: sonda economica per orchestratori e sidecar.
    200 = modello caricato e pronto; 503 = server su ma modello non disponibile."""
    ready = nlp is not None
    excl, mapping = _get_prefs()
    body = {
        "status": "ok" if ready else "loading",
        "model_loaded": ready,
        "model": os.path.basename(str(MODEL_DIR).rstrip("/\\")),
        "model_version": APP_MODEL_VERSION,
        "app_version": APP_VERSION,
        "device": "cuda" if device == 0 else "cpu",
        "tags": len(TAG_NAMES),
        "excluded_tags": excl,
        "mapping_enabled": mapping,
        "ocr_enabled": _get_ocr_enabled(),
        "ocr": ocr.engine_status(),
    }
    return jsonify(body), (200 if ready else 503)


def _is_pdf(name, data):
    return os.path.splitext((name or "").lower())[1] == ".pdf" or data[:5] == b"%PDF-"


class OcrFailed(ValueError):
    """L'OCR e' servito ma non e' stato possibile farlo.

    ValueError perche' le rotte gia' lo trattano come "file non leggibile"
    (400 con messaggio); `code` dice al client il motivo preciso, cosi' l'UI
    puo' distinguere "motore non installato" da "OCR disattivato" da "errore".
    """

    def __init__(self, msg, code="ocr_failed"):
        super().__init__(msg)
        self.code = code


# Cache degli OCR: /preview, /analyze e /pdf/preview ricevono lo STESSO file e
# l'OCR e' la parte piu' lenta di tutta la pipeline. La cache e' in memoria (mai
# su disco, come i documenti dell'anteprima) e tiene poche voci: la chiave e'
# l'hash dei byte del PDF, quindi lo stesso file non viene riconosciuto due volte.
_OCR_CACHE = OrderedDict()
_OCR_CACHE_MAX = 3
_OCR_CACHE_LOCK = threading.Lock()


def _ocr_cache_get(key):
    with _OCR_CACHE_LOCK:
        hit = _OCR_CACHE.get(key)
        if hit is not None:
            _OCR_CACHE.move_to_end(key)
    return hit


def _ocr_cache_put(key, value):
    with _OCR_CACHE_LOCK:
        _OCR_CACHE[key] = value
        _OCR_CACHE.move_to_end(key)
        while len(_OCR_CACHE) > _OCR_CACHE_MAX:
            _OCR_CACHE.popitem(last=False)


def _extract_document_text(name, data):
    """Testo del documento caricato + i metadati dell'OCR.

    Ritorna (text, meta):
      text -> il testo da analizzare (layer testuale, con le pagine scansionate
              sostituite dal testo riconosciuto dall'OCR);
      meta -> {"ocr": {...}} se l'OCR e' stato usato, {} altrimenti. La chiave
              interna "_ocr_result" contiene il risultato grezzo dell'OCR (con i
              box delle parole), serve alla redazione "burn-in" e NON viene
              mandata al client.

    Per i PDF non c'e' niente da cambiare: se il documento ha il layer testuale
    si comporta esattamente come prima. L'OCR scatta solo se una pagina non ha
    testo utilizzabile (una scansione).
    """
    if not _is_pdf(name, data):
        return _text_from_bytes(name, data), {}

    key = hashlib.sha256(data).hexdigest()
    hit = _ocr_cache_get(key)
    if hit is not None:
        return hit["text"], hit["meta"]

    with fitz.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise PdfLocked("PDF protetto da password: rimuovi la protezione e riprova.")
        scans = ocr.scanned_pages(doc)
        if not scans:
            # nessuna scansione: il layer testuale basta (comportamento di sempre)
            text = pdf_text.collect_readable_text(doc)
            meta = {}
        else:
            pages = ocr.page_texts(doc)
            extras = pdf_text.collect_extras(doc)

    if scans:
        if not _get_ocr_enabled():
            raise OcrFailed(
                "Questo PDF e' una scansione (il testo e' dentro un'immagine) e "
                "l'OCR automatico e' disattivato: riattivalo nelle impostazioni "
                "oppure incolla il testo a mano.",
                code="ocr_disabled")
        try:
            res = ocr.ocr_pdf(data, pages=scans)
        except ocr.OcrUnavailable as e:
            raise OcrFailed(str(e), code="ocr_unavailable") from e
        except ocr.OcrError as e:
            raise OcrFailed(f"OCR fallito: {e}", code="ocr_failed") from e
        for p in res["pages"]:
            pages[p["page"]] = p["text"]
        text = "\n".join(pages)
        if extras:
            text = text + "\n" + extras
        meta = {
            "ocr": {
                "engine": res["engine"],
                "dpi": res["dpi"],
                "lang": res["lang"],
                "pages": len(res["pages"]),
                "page_numbers": [p["page"] + 1 for p in res["pages"]],
                "seconds": res["seconds"],
                "chars": len(res["text"]),
            },
            "_ocr_result": res,
        }
    _ocr_cache_put(key, {"text": text, "meta": meta})
    return text, meta


def _public_meta(meta):
    """Meta senza le chiavi interne (quelle che iniziano con "_")."""
    return {k: v for k, v in (meta or {}).items() if not k.startswith("_")}


def _text_from_bytes(name, data):
    """Testo dai bytes di un upload: PDF via PyMuPDF, .md/.txt come testo puro.

    Per i PDF non basta `page.get_text()`: nei moduli AcroForm il valore sta
    nei widget (issue #85). `pdf_text.collect_readable_text` allinea
    l'estrazione a quanto `pdf_export` gia' ispezionava per i residui.
    """
    name = (name or "").lower()
    ext = os.path.splitext(name)[1]
    if _is_pdf(name, data):
        with fitz.open(stream=data, filetype="pdf") as doc:
            # senza, l'utente leggeva l'errore grezzo di PyMuPDF: "document closed or
            # encrypted". Un PDF con la sola password del proprietario si apre: needs_pass
            # e' falso e si prosegue.
            if doc.needs_pass:
                raise PdfLocked("PDF protetto da password: rimuovi la protezione e riprova.")
            # Non solo page.get_text(): nei PDF fillable il valore sta nei
            # widget AcroForm (issue #85). Stesso contratto di pdf_export.
            return pdf_text.collect_readable_text(doc)
    if ext in TEXT_EXTS or not ext:
        # utf-16 solo col BOM: senza, il codec accetta qualunque sequenza di lunghezza pari
        # e si mangia i .txt a 8 bit. cp1252 prima di latin-1: e' quello che scrivono Word
        # e il Blocco note italiani, e i due differiscono su euro e virgolette tipografiche.
        primo = ("utf-16",) if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ()
        for enc in primo + ("utf-8-sig", "cp1252", "latin-1"):
            try:
                # Via i byte nulli. Un utf-16 SENZA BOM finisce per forza su una codifica a
                # 8 bit e il testo esce "M\0a\0r\0i\0o": a schermo non si vede niente (in
                # HTML il NUL e' invisibile), quindi l'utente legge il documento intero,
                # non trova segnaposti e conclude che non ci fosse nulla da anonimizzare -
                # mentre i rilevatori, che lavorano sul testo vero, non trovano niente.
                # In un utf-16 latino il byte alto e' proprio quello nullo: tolto, il testo
                # torna esatto e le PII tornano trovabili. Riconoscere l'utf-16 contando i
                # byte nulli, invece, sbaglia in entrambe le direzioni.
                return data.decode(enc).replace("\x00", "")
            except UnicodeDecodeError:
                continue
    raise ValueError(
        f"Formato non supportato: {ext or name or 'sconosciuto'}. "
        "Accetto .pdf, .md, .txt oppure testo incollato."
    )


class PdfLocked(ValueError):
    """PDF con password di apertura: /preview lo segnala con `code`, cosi' l'UI lo
    dice nella sua lingua invece di "Anteprima del PDF non disponibile"."""


def _extract_upload(fs):
    """Testo da un file caricato (consuma lo stream)."""
    return _text_from_bytes(fs.filename, fs.read())


def _uploaded_file():
    """Il file dell'upload, se c'e'. "pdf" e' il nome storico del campo (l'UI lo
    usa ancora), "file" e' l'alias nuovo."""
    return next((request.files[k] for k in ("pdf", "file")
                 if k in request.files and request.files[k].filename), None)


@app.route("/analyze", methods=["POST"])
def analyze_route():
    up = _uploaded_file()
    meta = {}
    if up is not None:
        try:
            text, meta = _extract_document_text(up.filename, up.read())
        except OcrFailed as e:
            return jsonify({"error": str(e), "code": e.code}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:                       # PDF corrotto / protetto
            return jsonify({"error": f"Impossibile leggere il file: {e}"}), 400
        raw_excl = request.form.get("exclude_tags")
        raw_map = request.form.get("include_mapping")
    else:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        raw_excl = payload.get("exclude_tags")
        raw_map = payload.get("include_mapping")

    text = (text or "").strip()
    if not text:
        return jsonify({"error": "Nessun testo da analizzare."}), 400

    # override per-richiesta; senza override vale la configurazione del server
    # (snapshot atomico: una POST /settings concorrente non deve far leggere
    # un tag escluso aggiornato insieme a un mapping_enabled non ancora aggiornato, o viceversa)
    default_excl, default_map = _get_prefs()
    excl = server_config.parse_tag_list(raw_excl) if raw_excl is not None else default_excl
    keep_map = (server_config.parse_bool(raw_map, default_map)
                if raw_map is not None else default_map)
    out = analyze(text, excl, keep_map)
    # Il testo originale torna solo col dizionario attivo (l'UI lo rimette nella casella
    # dopo un upload). In modalita' definitiva la risposta non deve contenere nessun
    # valore in chiaro: la chiave manca del tutto, come il valore nei segmenti (#118).
    if keep_map:
        out["source_text"] = text
    # com'e' andata l'estrazione: se il documento era una scansione, l'UI lo dice
    out.update(_public_meta(meta))
    return jsonify(out)


# --------------------------------------------------------------------------- #
# PDF anonimizzato da scaricare (issue #7, punto 1)
# --------------------------------------------------------------------------- #
def _pdf_response(data, report, redactions, filename="documento_anonimizzato.pdf"):
    """Risposta binaria + intestazioni diagnostiche. X-PII-Residual e
    X-PII-Skipped sono le due che l'UI trasforma in AVVISO: valori del documento
    che, per motivi diversi, sono rimasti in chiaro."""
    resp = app.response_class(data, mimetype="application/pdf")
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
    resp.headers["X-PII-Redactions"] = str(redactions)
    resp.headers["X-PII-Residual"] = str(len(report.get("residual", [])))
    resp.headers["X-PII-Skipped"] = str(len(report.get("skipped", [])))
    resp.headers["X-PII-Notfound"] = str(len(report.get("not_found", [])))
    return resp


class _ReqError(Exception):
    """Errore da rendere al client come {"error": ...} con uno status preciso."""

    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.msg, self.status = msg, status


def _build_anonymized_pdf():
    """Costruisce il PDF ANONIMIZZATO dall'input della richiesta. Stesso input di
    /analyze:

    - multipart con un file .pdf  -> REDAZIONE VERA del documento originale: le
      PII sono rimosse dal content stream e sostituite dai placeholder, il layout
      resta quello di partenza (piu' metadati/annotazioni/campi modulo/segnalibri
      ripuliti e allegati rimossi);
    - multipart con .md/.txt oppure {"text": ...} -> PDF ricostruito impaginando
      da zero il solo testo anonimizzato.

    Il dizionario placeholder->valore NON viaggia sulla rete in nessuna delle due
    direzioni: viene ricostruito qui, serve solo a localizzare le PII nel PDF e
    muore con la richiesta. Per questo funziona identico anche con il dizionario
    reversibile disattivato (MAPPING_ENABLED=False), dove il client non ne ha
    nessuno.

    Ritorna (bytes, report, n_redazioni, nome_file). Solleva _ReqError sugli errori.
    """
    up = _uploaded_file()
    meta = {}
    if up is not None:
        name, data = up.filename, up.read()
        try:
            text, meta = _extract_document_text(name, data)
        except OcrFailed as e:
            raise _ReqError(str(e), 400)
        except ValueError as e:
            raise _ReqError(str(e))
        except Exception as e:                       # PDF corrotto / protetto
            raise _ReqError(f"Impossibile leggere il file: {e}")
        raw_excl = request.form.get("exclude_tags")
    else:
        payload = request.get_json(silent=True) or {}
        name, data, text = "", b"", payload.get("text", "")
        raw_excl = payload.get("exclude_tags")

    text = (text or "").strip()
    if not text:
        raise _ReqError("Nessun testo da anonimizzare.")

    stem = os.path.splitext(_safe_name(name, "documento.pdf"))[0] or "documento"
    out_name = f"{stem}_anonimizzato.pdf"

    excl = (server_config.parse_tag_list(raw_excl) if raw_excl is not None
            else _get_prefs()[0])
    # mapping_enabled=True e' interno: il risultato non esce da questa funzione.
    res = analyze(text, excl, mapping_enabled=True)
    if not res["mapping"]:
        raise _ReqError("Nessuna PII trovata: non c'e' niente da "
                        "anonimizzare in questo documento.", 422)

    if data and _is_pdf(name, data):
        ocr_res = meta.get("_ocr_result")
        try:
            if ocr_res:
                # Scansione: la PII e' nei pixel, non nel content stream. La pagina
                # viene rasterizzata e le caselle annerite DENTRO i pixel (burn-in).
                out, report = pdf_export.redact_pdf_ocr(data, res["mapping"], ocr_res)
            else:
                out, report = pdf_export.redact_pdf(data, res["mapping"])
        except pdf_export.PdfError as e:
            raise _ReqError(str(e))
        if report["occurrences"] == 0:
            raise _ReqError("Nessuna occorrenza trovata nel PDF: il testo "
                            "riconosciuto dall'OCR non contiene nessuna PII "
                            "rilevata (prova a rivedere la scansione).", 422)
        return out, report, report["occurrences"], out_name

    try:
        out = pdf_export.text_to_pdf(res["anonymized_text"])
    except pdf_export.PdfError as e:
        raise _ReqError(str(e))
    # testo reimpaginato da zero: nell'output c'e' solo il testo anonimizzato,
    # quindi niente residui e niente valori saltati da segnalare.
    return out, {}, res["n_entities"], out_name


@app.route("/pdf", methods=["POST"])
def pdf_route():
    """Scarica il documento ANONIMIZZATO in PDF (vedi _build_anonymized_pdf)."""
    try:
        out, report, redactions, name = _build_anonymized_pdf()
    except _ReqError as e:
        return jsonify({"error": e.msg}), e.status
    return _pdf_response(out, report, redactions, name)


@app.route("/pdf/preview", methods=["POST"])
def pdf_preview_route():
    """Come /pdf, ma il binario resta in memoria e si guarda a video: la risposta
    e' il descrittore del documento ({doc_id, n_pages, ...}), le pagine si prendono
    poi da /doc/<id>/page/<n>.png e il file da /doc/<id>/file.pdf.

    Serve a non anonimizzare due volte: l'anteprima a destra e il download del PDF
    usano lo STESSO documento, generato una volta sola."""
    try:
        out, report, redactions, name = _build_anonymized_pdf()
    except _ReqError as e:
        return jsonify({"error": e.msg}), e.status
    doc_id, n_pages = _store_doc(out, name)
    body = {
        "doc_id": doc_id,
        "n_pages": n_pages,
        "filename": name,
        "redactions": redactions,
        "residual": len(report.get("residual", [])),
        "skipped": len(report.get("skipped", [])),
        "not_found": len(report.get("not_found", [])),
    }
    # le pagine redatte in burn-in (scansioni): l'UI avvisa che il PDF e' stato
    # rasterizzato e che quelle pagine non hanno piu' testo selezionabile
    if report.get("ocr_pages"):
        body["ocr_pages"] = report["ocr_pages"]
    return jsonify(body)


# --------------------------------------------------------------------------- #
# Estrazione zone PII con coordinate (per OCR esterno)
# --------------------------------------------------------------------------- #
def _extract_pii_zones(pdf_bytes, mapping, excluded_tags=None, mapping_enabled=True):
    """
    Estrae le zone (bounding box) delle PII in un PDF usando lo stesso matching
    char-preciso di pdf_export.redact_pdf.
    
    Ritorna dict con:
    - zones: lista zone per pagina con bbox (punti PDF) e bbox_normalizzato (0-1)
    - page_sizes: dimensioni pagine in punti PDF
    - dpi: DPI usato per il rendering anteprima
    """
    excluded = set(excluded_tags or ())
    items = [(ph, v) for ph, v in mapping.items()
             if ph not in excluded and isinstance(v, str) and v.strip()]
    if not items:
        return {"zones": [], "page_sizes": [], "dpi": PREVIEW_DPI}
    
    # Ordina per lunghezza decrescente (valori lunghi prima, come in redact_pdf)
    items.sort(key=lambda kv: -len(kv[1]))
    
    zones = []
    page_sizes = []
    
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            page_sizes.append({"width": page.rect.width, "height": page.rect.height})
            text, boxes = pdf_export._page_char_index(page)
            
            if not text.strip():
                continue
                
            for ph, val in items:
                if pdf_export._too_noisy(val):
                    continue
                pat = pdf_export._value_pattern(val)
                if not pat:
                    continue
                    
                for m in pat.finditer(text):
                    rects = pdf_export._match_rects(boxes, m)
                    for r in rects:
                        # Normalizza bbox 0-1 rispetto alle dimensioni pagina
                        norm = [
                            r.x0 / page.rect.width,
                            r.y0 / page.rect.height,
                            r.x1 / page.rect.width,
                            r.y1 / page.rect.height
                        ]
                        label = ph.split("_")[0].strip("[]")
                        zones.append({
                            "page": page_num,
                            "label": label,
                            "placeholder": ph,
                            "value": val if mapping_enabled else None,
                            "bbox": [r.x0, r.y0, r.x1, r.y1],
                            "bbox_normalized": norm,
                            "confidence": 1.0  # validated handled by mapping source
                        })
    
    return {"zones": zones, "page_sizes": page_sizes, "dpi": PREVIEW_DPI}


@app.route("/pii-zones", methods=["POST"])
def pii_zones_route():
    """
    Estrae le coordinate (bounding box) delle zone PII in un PDF.
    
    Input: multipart con file .pdf (campo "pdf" o "file")
           campi opzionali: exclude_tags, include_mapping
    Output: JSON con zones, page_sizes, dpi
    
    Utile per: dare coordinate a motore OCR esterno su documenti scansionati,
    o per evidenziazione programmatica zone sensibili.
    """
    up = _uploaded_file()
    if up is None:
        return jsonify({"error": "Nessun file caricato."}), 400
    name, data = up.filename, up.read()
    if not _is_pdf(name, data):
        return jsonify({"error": "L'estrazione zone PII vale solo per i PDF."}), 400
    
    try:
        text, _meta = _extract_document_text(name, data)
    except PdfLocked as e:
        return jsonify({"error": f"Impossibile leggere il file: {e}", "code": "pdf_locked"}), 400
    except OcrFailed as e:
        return jsonify({"error": str(e), "code": e.code}), 400
    except Exception as e:
        return jsonify({"error": f"Impossibile leggere il file: {e}"}), 400
    
    raw_excl = request.form.get("exclude_tags")
    raw_map = request.form.get("include_mapping")
    
    default_excl, default_map = _get_prefs()
    excl = server_config.parse_tag_list(raw_excl) if raw_excl is not None else default_excl
    keep_map = (server_config.parse_bool(raw_map, default_map)
                if raw_map is not None else default_map)
    
    # Analizza per ottenere mapping (serve per sapere COSA cercare nel PDF)
    res = analyze(text, excl, mapping_enabled=True)
    if not res["mapping"]:
        return jsonify({"error": "Nessuna PII trovata nel documento.", "zones": [], "page_sizes": [], "dpi": PREVIEW_DPI}), 422
    
    zones_data = _extract_pii_zones(data, res["mapping"], excl, keep_map)
    return jsonify(zones_data)


@app.route("/ocr-zones", methods=["POST"])
def ocr_zones_route():
    """
    Integra risultati OCR esterni per zone specifiche e ritorna analisi completa.
    
    Input JSON:
    {
        "pdf_base64": "...",           # PDF originale (base64) OPPURE usa doc_id da /preview
        "doc_id": "...",               # alternativo a pdf_base64: documento già in memoria
        "zones": [                     # risultati OCR per zone specifiche
            {"page": 0, "bbox": [x0,y0,x1,y1], "ocr_text": "Mario Rossi"}
        ],
        "exclude_tags": [...],
        "include_mapping": true
    }
    
    Flusso:
    1. Carica PDF (da base64 o da doc_id in memoria)
    2. Analizza layer testuale esistente (già fatto da /analyze)
    3. Per ogni zona OCR: esegue detect_model + detect_regex su ocr_text
    4. Converte coordinate OCR (bbox) in offset globali nel testo completo
    5. Fondi entità OCR con entità layer testuale
    6. Ritorna analisi completa (come /analyze) + opzionale PDF anonimizzato
    
    Nota: le coordinate bbox DEVONO essere nello stesso sistema di riferimento
    del PDF (punti PDF, 1/72 inch), come ritornato da /pii-zones.
    """
    payload = request.get_json(silent=True) or {}
    
    # Carica PDF
    pdf_bytes = None
    if "pdf_base64" in payload:
        import base64
        try:
            pdf_bytes = base64.b64decode(payload["pdf_base64"])
        except Exception as e:
            return jsonify({"error": f"pdf_base64 non valido: {e}"}), 400
    elif "doc_id" in payload:
        d = _get_doc(payload["doc_id"])
        if d is None:
            return jsonify({"error": "Documento non trovato in memoria (scaduto o mai caricato)."}), 404
        pdf_bytes = d["pdf"]
    else:
        return jsonify({"error": "Specificare pdf_base64 o doc_id."}), 400
    
    ocr_zones = payload.get("zones", [])
    if not isinstance(ocr_zones, list):
        return jsonify({"error": "zones deve essere una lista."}), 400
    
    raw_excl = payload.get("exclude_tags")
    raw_map = payload.get("include_mapping")
    
    default_excl, default_map = _get_prefs()
    excl = server_config.parse_tag_list(raw_excl) if raw_excl is not None else default_excl
    keep_map = (server_config.parse_bool(raw_map, default_map)
                if raw_map is not None else default_map)
    
    # 1. Estrai testo dal PDF (layer testuale)
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.needs_pass:
                return jsonify({"error": "PDF protetto da password.", "code": "pdf_locked"}), 400
            text = pdf_text.collect_readable_text(doc)
    except Exception as e:
        return jsonify({"error": f"Impossibile leggere il PDF: {e}"}), 400
    
    # 2. Analisi standard sul layer testuale
    base_res = analyze(text, excl, mapping_enabled=keep_map)
    
    # 3. Processa zone OCR
    ocr_entities = []
    for zone in ocr_zones:
        page_num = zone.get("page")
        bbox = zone.get("bbox")
        ocr_text = zone.get("ocr_text", "").strip()
        
        if page_num is None or bbox is None or not ocr_text:
            continue
        
        # Esegui detection su testo OCR
        model_ents, _ = detect_model(ocr_text)
        complete_time(model_ents, ocr_text)
        regex_ents = detect_regex(ocr_text)
        
        all_ents = model_ents + regex_ents
        if excl:
            all_ents = [e for e in all_ents if e["label"] not in excl]
        
        # Converti coordinate locali (nel testo OCR) in coordinate globali
        # Per semplicità, restituiamo le entità OCR con metadati della zona
        for e in all_ents:
            e["ocr_zone"] = {
                "page": page_num,
                "bbox": bbox,
                "source_text": ocr_text
            }
            e["source"] = "ocr"
        ocr_entities.extend(all_ents)
    
    # 4. Fondi entità base + OCR (sullo stesso testo base)
    # Nota: questo è semplificato - in produzione servirebbe allineamento
    # più sofisticato tra coordinate PDF e offset testo
    all_cands = []
    # Entità base (già con offset globali)
    for e in base_res.get("_raw_entities", []):
        all_cands.append(e)
    # Entità OCR (offset relativi al testo OCR - andrebbero mappati)
    # Per ora le includiamo come entità separate con metadati zona
    
    # 5. Ritorna risultato combinato
    return jsonify({
        "base_analysis": base_res,
        "ocr_entities": ocr_entities,
        "ocr_zones_processed": len(ocr_zones),
        "note": "Le entità OCR hanno offset relativi al testo OCR della zona. Per fusione completa serve allineamento coordinate."
    })


@app.route("/analyze-with-zones", methods=["POST"])
def analyze_with_zones_route():
    """
    Analizza un PDF e restituisce sia l'analisi completa che le zone PII con coordinate.
    Combina /analyze + /pii-zones in una sola chiamata per evitare doppio processing.
    
    Input: multipart con file .pdf (campo "pdf" o "file")
           campi opzionali: exclude_tags, include_mapping
    Output: JSON con tutti i campi di /analyze PIU' zones, page_sizes, dpi
    """
    up = _uploaded_file()
    if up is None:
        return jsonify({"error": "Nessun file caricato."}), 400
    name, data = up.filename, up.read()
    if not _is_pdf(name, data):
        return jsonify({"error": "L'analisi con zone vale solo per i PDF."}), 400
    
    try:
        text, _meta = _extract_document_text(name, data)
    except PdfLocked as e:
        return jsonify({"error": f"Impossibile leggere il file: {e}", "code": "pdf_locked"}), 400
    except OcrFailed as e:
        return jsonify({"error": str(e), "code": e.code}), 400
    except Exception as e:
        return jsonify({"error": f"Impossibile leggere il file: {e}"}), 400
    
    raw_excl = request.form.get("exclude_tags")
    raw_map = request.form.get("include_mapping")
    
    default_excl, default_map = _get_prefs()
    excl = server_config.parse_tag_list(raw_excl) if raw_excl is not None else default_excl
    keep_map = (server_config.parse_bool(raw_map, default_map)
                if raw_map is not None else default_map)
    
    # Analisi completa
    res = analyze(text, excl, mapping_enabled=keep_map)
    
    # Estrazione zone (solo se ci sono PII trovate)
    zones_data = {"zones": [], "page_sizes": [], "dpi": PREVIEW_DPI}
    if res["mapping"]:
        zones_data = _extract_pii_zones(data, res["mapping"], excl, keep_map)
    
    # Combina risultati
    res.update(zones_data)
    return jsonify(res)


# --------------------------------------------------------------------------- #
# Anteprima a video del documento CARICATO + pagine renderizzate
# --------------------------------------------------------------------------- #
@app.route("/preview", methods=["POST"])
def preview_route():
    """Tiene in memoria il PDF appena caricato per mostrarlo a sinistra e ne
    ritorna anche il testo estratto (cosi' l'UI puo' offrire "PDF" o "Testo"
    senza una seconda estrazione)."""
    up = _uploaded_file()
    if up is None:
        return jsonify({"error": "Nessun file caricato."}), 400
    name, data = up.filename, up.read()
    if not _is_pdf(name, data):
        return jsonify({"error": "L'anteprima renderizzata vale solo per i PDF."}), 400
    try:
        text, meta = _extract_document_text(name, data)
        doc_id, n_pages = _store_doc(data, name)
    except PdfLocked as e:
        return jsonify({"error": f"Impossibile leggere il file: {e}", "code": "pdf_locked"}), 400
    except OcrFailed as e:
        return jsonify({"error": str(e), "code": e.code}), 400
    except Exception as e:                           # PDF corrotto / protetto
        return jsonify({"error": f"Impossibile leggere il file: {e}"}), 400
    body = {"doc_id": doc_id, "n_pages": n_pages,
            "filename": _safe_name(name), "text": text}
    body.update(_public_meta(meta))
    return jsonify(body)


@app.route("/doc/<doc_id>/page/<int:n>.png")
def doc_page(doc_id, n):
    """Pagina n (0-based) renderizzata in PNG. 404 se il documento e' scaduto
    dalla LRU: l'UI in quel caso rifa' l'upload."""
    d = _get_doc(doc_id)
    if d is None:
        return ("", 404)
    png = _page_png(d, n)
    if png is None:
        return ("", 404)
    resp = app.response_class(png, mimetype="image/png")
    resp.headers["Cache-Control"] = "private, max-age=600"
    return resp


@app.route("/doc/<doc_id>/file.pdf")
def doc_file(doc_id):
    """Download del PDF tenuto in memoria (quello gia' anonimizzato da
    /pdf/preview): non ricalcola niente."""
    d = _get_doc(doc_id)
    if d is None:
        return jsonify({"error": "Documento non piu' disponibile."}), 404
    resp = app.response_class(d["pdf"], mimetype="application/pdf")
    resp.headers["Content-Disposition"] = f"attachment; filename={d['name']}"
    return resp


# --------------------------------------------------------------------------- #
# Impostazioni: legenda dei tag, tag esclusi, dizionario reversibile on/off
# (/tags resta come alias storico dello stesso endpoint)
# --------------------------------------------------------------------------- #
@app.route("/settings", methods=["GET"])
@app.route("/tags", methods=["GET"])
def settings_get():
    excl, mapping = _get_prefs()
    return jsonify({
        "tags": [{"tag": t, "it": it, "en": en, "example": ex} for t, it, en, ex in TAGS],
        "excluded_tags": excl,
        "mapping_enabled": mapping,
        "ocr_enabled": _get_ocr_enabled(),
        "ocr": ocr.engine_status(),
        "config_path": str(server_config.prefs_path()),
        "env_override": ("PII_EXCLUDE_TAGS" in os.environ
                         or "PII_MAPPING" in os.environ
                         or "PII_OCR" in os.environ),
    })


@app.route("/settings", methods=["POST"])
@app.route("/tags", methods=["POST"])
def settings_post():
    global EXCLUDED_TAGS, MAPPING_ENABLED, OCR_ENABLED
    data = request.get_json(silent=True) or {}
    tags = None
    if "excluded_tags" in data:
        tags = server_config.parse_tag_list(data["excluded_tags"])
        unknown = [t for t in tags if t not in TAG_NAMES]
        if unknown:
            return jsonify({"error": f"Tag sconosciuti: {', '.join(unknown)}"}), 400
    mapping = data.get("mapping_enabled")
    ocr_enabled = data.get("ocr_enabled")
    if tags is None and mapping is None and ocr_enabled is None:
        return jsonify({"error": "Niente da salvare: passa excluded_tags, "
                                 "mapping_enabled e/o ocr_enabled."}), 400
    saved = server_config.save_prefs(excluded_tags=tags, mapping_enabled=mapping,
                                     ocr_enabled=ocr_enabled)
    with _PREFS_LOCK:
        EXCLUDED_TAGS = saved["excluded_tags"]
        MAPPING_ENABLED = saved["mapping_enabled"]
        OCR_ENABLED = saved.get("ocr_enabled", True)
    return jsonify({"ok": True, "excluded_tags": EXCLUDED_TAGS,
                    "mapping_enabled": MAPPING_ENABLED,
                    "ocr_enabled": OCR_ENABLED,
                    "ocr": ocr.engine_status()})


# --------------------------------------------------------------------------- #
# Config host/porta (GET = leggi, POST = salva per il prossimo avvio)
# --------------------------------------------------------------------------- #
@app.route("/config", methods=["GET"])
def config_get():
    cfg = server_config.load_config()
    return jsonify({
        "host": cfg.get("host", server_config.DEFAULT_HOST),
        "port": cfg.get("port", server_config.DEFAULT_PORT),
        "config_path": str(server_config.config_path()),
    })


@app.route("/config", methods=["POST"])
def config_post():
    data = request.get_json(silent=True) or {}
    host = str(data.get("host", server_config.DEFAULT_HOST)).strip()
    try:
        port = int(data.get("port", server_config.DEFAULT_PORT))
    except (ValueError, TypeError):
        return jsonify({"error": "Porta non valida."}), 400
    if not (1024 <= port <= 65535):
        return jsonify({"error": "La porta deve essere tra 1024 e 65535."}), 400
    server_config.save_config(host, port)
    return jsonify({"ok": True, "host": host, "port": port})


@app.route("/port-check")
def port_check():
    host = request.args.get("host", server_config.DEFAULT_HOST)
    try:
        port = int(request.args.get("port", server_config.DEFAULT_PORT))
    except (ValueError, TypeError):
        return jsonify({"available": False})
    return jsonify({"available": server_config.port_available(host, port)})


# --------------------------------------------------------------------------- #
# UI (single page)
# --------------------------------------------------------------------------- #
PAGE = r"""
<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="/assets/mascot_shield.png">
<title>Rizzo PII · locale</title>
<style>
  :root{
    --bg:#f5f4f8; --card:#ffffff; --ink:#211f29; --muted:#6c6677; --soft:#9d97a9;
    --line:#e9e6f1; --line2:#f2f0f7; --brand:#7c3a9e; --brand-dk:#643183;
    --ok:#1d8a4e; --shadow:0 1px 2px rgba(33,26,48,.05),0 10px 28px rgba(33,26,48,.05);
    --r:14px;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--ink);overflow-x:hidden;overflow-y:auto;
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}
  /* desktop: altezza = viewport, lo scroll avviene DENTRO i componenti (no effetto zoom-out);
     su schermi stretti (media query sotto) si sblocca e la pagina scrolla, colonne impilate */
  .app{max-width:1240px;margin:0 auto;padding:14px 22px 10px;height:100vh;
       display:flex;flex-direction:column;overflow:hidden}

  /* header */
  header{display:flex;align-items:center;gap:14px;margin-bottom:14px;flex-wrap:wrap;flex:none}
  .logo{width:46px;height:46px;display:grid;place-items:center;font-size:24px}
  .logo img{width:100%;height:100%;object-fit:contain;
            filter:drop-shadow(0 2px 4px rgba(20,28,46,.18))}
  .empty img{width:128px;height:auto;opacity:.96;margin-bottom:2px}
  header h1{font-size:19px;margin:0;font-weight:700;letter-spacing:-.01em}
  header h1 .ver{font-size:11.5px;font-weight:600;color:var(--soft);vertical-align:middle;margin-left:4px}
  header .tag{font-size:12.5px;color:var(--muted)}
  .badge{margin-left:auto;display:inline-flex;align-items:center;gap:7px;background:#eaf7ef;
         color:var(--ok);border:1px solid #cde8d8;border-radius:999px;padding:6px 13px;
         font-size:12.5px;font-weight:600}
  .badge .dot{width:7px;height:7px;border-radius:50%;background:var(--ok)}
  .lang{margin-left:8px;background:#fff;border:1px solid var(--line);border-radius:10px;
        padding:6px 10px;font:inherit;font-size:12.5px;font-weight:600;color:var(--ink);cursor:pointer}
  .lang:hover{border-color:#cfd5e0}

  /* stepper / tabs */
  .tabs{display:flex;gap:8px;margin-bottom:14px;flex:none}
  .tab{flex:0 0 auto;display:flex;align-items:center;gap:10px;background:var(--card);
       border:1px solid var(--line);border-radius:12px;padding:11px 16px;cursor:pointer;
       color:var(--muted);font-weight:600;font-size:14px;transition:.15s;user-select:none}
  .tab:hover{border-color:#d4d9e4}
  .tab.on{color:var(--ink);border-color:var(--brand);box-shadow:0 0 0 3px rgba(124,58,158,.13)}
  .tab .num{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;
            background:var(--line);color:var(--muted);font-size:12.5px;font-weight:700}
  .tab.on .num{background:var(--brand);color:#fff}
  .tab .arrow{color:var(--soft)}

  /* grid (pane "Ripristina") + workspace (pane "Anonimizza") */
  .grid{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr;gap:16px;flex:1;min-height:0}
  .workspace{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr;gap:16px;
             align-items:stretch;flex:1;min-height:0}
  .card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
        box-shadow:var(--shadow);display:flex;flex-direction:column;overflow:hidden;min-height:0}
  .card .hd{padding:13px 16px;border-bottom:1px solid var(--line2);display:flex;
            align-items:center;gap:10px}
  .card .hd h2{font-size:13px;margin:0;text-transform:uppercase;letter-spacing:.04em;
               color:var(--muted);font-weight:700}
  .card .hd .right{margin-left:auto;display:flex;gap:8px;align-items:center}
  .card .bd{padding:14px 16px;flex:1;min-height:0;display:flex;flex-direction:column}

  textarea{width:100%;flex:1;min-height:0;resize:none;border:1px solid var(--line);
           border-radius:10px;padding:13px 14px;font-size:14.5px;line-height:1.6;color:var(--ink);
           background:#fcfcfe;font-family:inherit}
  textarea:focus{outline:none;border-color:var(--brand);box-shadow:0 0 0 3px rgba(124,58,158,.13)}
  textarea.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13.5px}

  /* dropzone */
  .drop{border:1.5px dashed var(--line);border-radius:10px;padding:11px 14px;margin-top:11px;
        display:flex;align-items:center;gap:11px;color:var(--muted);font-size:13.5px;cursor:pointer;
        transition:.15s;background:#fcfcfe}
  .drop:hover,.drop.hot{border-color:var(--brand);background:#f8f4fc;color:var(--ink)}
  .drop .ic{font-size:18px}
  .drop b{color:var(--ink)}

  /* buttons */
  .row{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:12px}
  button{font:inherit;border:0;border-radius:10px;padding:10px 17px;font-weight:600;font-size:14px;
         cursor:pointer;display:inline-flex;align-items:center;gap:8px;transition:.15s}
  .btn{background:var(--brand);color:#fff;box-shadow:0 1px 2px rgba(124,58,158,.22)}
  .btn:hover{background:var(--brand-dk)}
  .btn.lg{padding:12px 22px;font-size:15px}
  .ghost{background:#fff;color:var(--ink);border:1px solid var(--line)}
  .ghost:hover{border-color:#cfd5e0;background:#fafbfd}
  button:disabled{opacity:.55;cursor:default}
  .spin{width:15px;height:15px;border:2px solid rgba(255,255,255,.45);border-top-color:#fff;
        border-radius:50%;animation:sp .7s linear infinite}
  .ghost .spin{border-color:rgba(124,58,158,.25);border-top-color:var(--brand)}
  @keyframes sp{to{transform:rotate(360deg)}}
  .hint{color:var(--soft);font-size:12.5px;margin-left:auto}

  /* preview */
  .seg-tabs{display:inline-flex;background:var(--line2);border-radius:9px;padding:3px}
  .seg-tabs button{background:transparent;color:var(--muted);padding:5px 12px;font-size:13px;
                   border-radius:7px;box-shadow:none}
  .seg-tabs button.on{background:#fff;color:var(--ink);box-shadow:var(--shadow)}
  .view{flex:1;overflow:auto;border:1px solid var(--line);border-radius:10px;
        padding:14px;background:#fcfcfe;min-height:0}
  /* editor (sx) e anteprima/testo (dx): riempiono la card -> stessa altezza, scroll sincronizzato */
  #src,#anon{flex:1;min-height:0}
  #pane2 textarea{flex:1;min-height:0}

  /* CON RISULTATO: la pagina scrolla e i componenti hanno altezza fissa generosa (no schiacciamento).
     Il dizionario va sotto e si raggiunge scrollando la pagina; editor/anteprima scrollano internamente. */
  .app.has-result{height:auto;overflow:visible}
  .app.has-result .workspace{flex:none}
  .app.has-result #src,.app.has-result #anon,.app.has-result .view{flex:none;height:60vh}
  /* pane "Ripristina": non deve collassare quando l'app e' in modalita' scroll-pagina */
  #pane2 textarea,#pane2 .view{min-height:60vh}

  /* schermi stretti: la pagina scrolla, una colonna, card impilate con altezze fisse leggibili.
     DEVE stare dopo le regole flex:1 qui sopra per vincere a parita' di specificita'. */
  @media(max-width:920px){
    .grid{grid-template-columns:1fr}
    .app{height:auto;overflow:visible}
    .workspace{grid-template-columns:1fr;grid-template-rows:none;flex:none;min-height:auto}
    #src,#anon,.view,#pane2 textarea{flex:none;height:60vh}
  }
  /* render del PDF (pagine PNG servite da /doc/<id>/page/<n>.png).
     Sfondo scuro come un lettore PDF: le pagine bianche si staccano. */
  .pdfview{padding:12px;background:#e8e6ef}
  .pdfview .pg{position:relative;width:calc(100% * var(--z,1));margin:0 auto 12px;background:#fff;
               border-radius:4px;overflow:hidden;
               box-shadow:0 1px 3px rgba(33,26,48,.18),0 6px 18px rgba(33,26,48,.10)}
  .pdfview .pg:last-child{margin-bottom:0}
  .pdfview .pg img{display:block;width:100%;height:auto;min-height:24px}
  .pdfview .pgn{position:absolute;right:7px;bottom:7px;background:rgba(28,35,48,.6);color:#fff;
                font-size:11px;font-weight:700;border-radius:6px;padding:2px 7px;letter-spacing:.02em}
  /* zoom: pulsanti +/- sticky in alto a destra, ctrl+rotellina fa lo stesso (issue #92).
     La barra e' a tutta larghezza ma "vuota" (pointer-events:none) cosi' non ruba i click
     sulla pagina sotto: solo la pillola dei pulsanti li riceve. */
  .zoomctl{position:sticky;top:0;left:0;z-index:3;display:flex;justify-content:flex-end;
           pointer-events:none;margin-bottom:8px}
  .zoomctl .pill{pointer-events:auto;display:flex;align-items:center;gap:1px;
                 background:rgba(255,255,255,.94);border:1px solid var(--line);border-radius:999px;
                 padding:3px;box-shadow:0 1px 4px rgba(33,26,48,.15)}
  .zoomctl button{width:23px;height:23px;border:none;background:transparent;border-radius:999px;
                  font-size:14px;font-weight:700;color:var(--ink);cursor:pointer;line-height:1;
                  transition:.12s}
  .zoomctl button:hover{background:var(--line2)}
  .zoomctl .zpct{font-size:11px;font-weight:700;color:var(--muted);min-width:36px;text-align:center;
                 cursor:pointer;user-select:none}
  .zoomctl .zpct:hover{color:var(--ink)}
  .pdfbusy{display:flex;align-items:center;justify-content:center;gap:10px;height:100%;
           color:var(--muted);font-size:13.5px;font-weight:600}
  .pdfbusy .spin{border-color:rgba(124,58,158,.25);border-top-color:var(--brand)}

  .preview{white-space:pre-wrap;word-wrap:break-word;font-size:14.5px;line-height:1.7}
  .ph{border-radius:6px;padding:1px 7px 2px;font-weight:600;font-size:12.5px;cursor:help;
      border:1px solid;white-space:nowrap;display:inline-block;line-height:1.4;
      transition:.12s}
  .ph .ck{font-size:10px;opacity:.8;margin-left:3px}
  .ph.dim{opacity:.25;filter:grayscale(.6)}
  .empty{color:var(--soft);display:flex;flex-direction:column;align-items:center;justify-content:center;
         height:100%;gap:9px;text-align:center;font-size:14px}
  .empty .big{font-size:34px;opacity:.6}

  /* legend / stats */
  .legend{display:flex;gap:7px;flex-wrap:wrap;padding:12px 16px;border-top:1px solid var(--line2)}
  .chip{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:999px;
        padding:4px 11px;font-size:12.5px;font-weight:600;color:var(--ink);cursor:pointer;
        background:#fff;user-select:none;transition:.12s}
  .chip:hover{border-color:#cfd5e0}
  .chip.off{opacity:.4;text-decoration:line-through}
  .chip .sw{width:10px;height:10px;border-radius:3px}
  .chip .n{color:var(--muted);font-weight:700}
  .meta{display:flex;gap:8px;flex-wrap:wrap;padding:0 16px 12px}
  .stat{background:#f7f8fb;border:1px solid var(--line2);border-radius:9px;padding:6px 11px;
        font-size:12.5px;color:var(--muted)}
  .stat b{color:var(--ink);font-weight:700}

  /* mapping table */
  .tablewrap{max-height:240px;overflow:auto;padding:0 16px 16px}
  table{width:100%;border-collapse:collapse;font-size:13.5px}
  th{position:sticky;top:0;background:#fff;text-align:left;color:var(--soft);font-weight:600;
     font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;padding:8px 8px;border-bottom:1px solid var(--line)}
  td{padding:8px 8px;border-bottom:1px solid var(--line2);vertical-align:top}
  td.k{font-family:ui-monospace,Consolas,monospace;font-weight:600;white-space:nowrap}
  td.v{word-break:break-word}
  tr:hover td{background:#fafbfd}

  /* card "Dizionario": a tutta larghezza sotto le due colonne, scrollabile */
  .dict{margin-top:16px;flex:none}
  .dict .bd{padding:0;display:block}
  .dict .meta{padding:13px 16px 6px}
  .dict .legend{padding:0 16px 12px;border-top:none}
  .dict .tablewrap{max-height:300px;overflow:auto;padding:0 16px 16px}

  /* reverse panel */
  .pane{display:none}
  .pane.on{display:flex;flex-direction:column;flex:1;min-height:0}
  .callout{display:flex;gap:11px;background:#fff7ed;border:1px solid #fde6c8;border-radius:11px;
           padding:12px 14px;font-size:13.5px;color:#7a4d12;margin-bottom:14px;flex:none}
  .callout b{color:#5c3a0d}
  .callout .ic{font-size:17px}

  /* angolo alto a destra: la lingua resta in linea col badge; l'icona "galleggia" sotto (no shift) */
  .topright{position:relative;margin-left:8px;display:flex;align-items:center}
  .info{position:absolute;top:calc(100% + 6px);right:0;display:grid;place-items:center;
        width:25px;height:25px;border-radius:8px;background:#fff7ed;border:1px solid #fde6c8;
        font-size:13px;line-height:1;cursor:pointer;user-select:none;transition:.12s}
  .info:hover{border-color:#f3c98b}
  .info.open{border-color:#f3c98b;background:#ffedd5}
  .info .tip{position:absolute;top:calc(100% + 8px);right:0;width:300px;
             background:#fff;border:1px solid var(--line);border-radius:11px;box-shadow:var(--shadow);
             padding:12px 14px;font-size:12.5px;font-weight:400;color:#5c4326;line-height:1.55;
             text-align:left;opacity:0;visibility:hidden;transform:translateY(-4px);
             transition:.15s;z-index:60;pointer-events:none}
  .info.open .tip{opacity:1;visibility:visible;transform:translateY(0);pointer-events:auto}
  .info .tip b{color:var(--ink)}
  .info .tip a{color:var(--brand);font-weight:700;text-decoration:none;word-break:break-word}
  .info .tip a:hover{text-decoration:underline}
  .info .tip::before{content:"";position:absolute;top:-5px;right:8px;width:9px;height:9px;background:#fff;
                     border-left:1px solid var(--line);border-top:1px solid var(--line);transform:rotate(45deg)}

  /* crediti (footer dell'app) */
  .credits{flex:none;text-align:center;padding:9px 0 2px;font-size:11.5px;color:var(--soft)}
  .credits b{color:var(--muted);font-weight:700}
  .credits .u{color:var(--brand);font-weight:600}

  /* toast */
  #toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(20px);
         background:#1c2330;color:#fff;padding:11px 18px;border-radius:11px;font-size:13.5px;font-weight:600;
         box-shadow:0 12px 32px rgba(0,0,0,.22);opacity:0;pointer-events:none;transition:.22s;z-index:50;
         display:flex;align-items:center;gap:9px}
  #toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
  #toast.ok::before{content:"✓";color:#6ee7a8}
  .kbd{font-family:ui-monospace,Consolas,monospace;background:#eef1f6;border:1px solid var(--line);
       border-radius:5px;padding:1px 6px;font-size:11.5px;color:var(--muted)}

  /* config modal */
  .cfg-overlay{position:fixed;inset:0;background:rgba(33,26,48,.35);z-index:100;
               display:flex;align-items:center;justify-content:center;opacity:0;
               visibility:hidden;transition:.18s}
  .cfg-overlay.open{opacity:1;visibility:visible}
  .cfg-card{background:#fff;border-radius:16px;box-shadow:0 16px 48px rgba(33,26,48,.18);
            padding:26px 28px 22px;width:380px;max-width:92vw}
  .cfg-card h3{margin:0 0 16px;font-size:16px;font-weight:700;display:flex;align-items:center;gap:9px}
  .cfg-row{display:flex;flex-direction:column;gap:5px;margin-bottom:14px}
  .cfg-row label{font-size:12.5px;font-weight:600;color:var(--muted);text-transform:uppercase;
                 letter-spacing:.04em}
  .cfg-row input{border:1px solid var(--line);border-radius:9px;padding:9px 12px;font:inherit;
                 font-size:14px;color:var(--ink);background:#fcfcfe}
  .cfg-row input:focus{outline:none;border-color:var(--brand);box-shadow:0 0 0 3px rgba(124,58,158,.13)}
  .cfg-status{font-size:12.5px;font-weight:600;padding:7px 11px;border-radius:8px;margin-bottom:14px;
              display:none}
  .cfg-status.ok{display:block;background:#eaf7ef;color:var(--ok)}
  .cfg-status.fail{display:block;background:#fef2f2;color:#b91c1c}
  .cfg-btns{display:flex;gap:9px;align-items:center}
  .cfg-note{font-size:11.5px;color:var(--soft);margin-top:12px}
  .gear{background:none;border:1px solid var(--line);width:30px;height:30px;border-radius:8px;
        display:grid;place-items:center;font-size:15px;padding:0;cursor:pointer;transition:.12s;
        margin-left:6px;flex:none}
  .gear:hover{border-color:#cfd5e0;background:#fafbfd}

  /* switch "Dizionario reversibile" (sotto la dropzone, nella card di input).
     OFF non e' uno stato di errore ma una modalita' piu' stretta -> ambra, non grigio. */
  .mapsw{display:flex;align-items:flex-start;gap:12px;margin-top:11px;padding:11px 13px;
         border:1px solid var(--line);border-radius:11px;background:#fcfcfe;transition:.15s}
  .mapsw.off{background:#fff7ed;border-color:#fde6c8}
  /* .tsw e non .sw: .sw e' gia' il quadratino-colore dei chip nella legenda */
  .tsw{position:relative;width:42px;height:24px;border-radius:999px;background:var(--brand);
       border:0;padding:0;flex:none;cursor:pointer;transition:.18s;margin-top:1px}
  .tsw:focus-visible{outline:2px solid var(--brand);outline-offset:2px}
  .tsw .knob{position:absolute;top:3px;left:3px;width:18px;height:18px;border-radius:50%;
             background:#fff;transition:.18s;box-shadow:0 1px 3px rgba(0,0,0,.28)}
  .tsw.off{background:#c2410c}
  .tsw.off .knob{transform:translateX(18px)}
  .mapsw .ttl{font-size:13.5px;font-weight:700;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .mapsw .st{font-size:11px;font-weight:800;letter-spacing:.05em;border-radius:999px;
             padding:2px 8px;background:#f1e9f7;color:var(--brand-dk)}
  .mapsw.off .st{background:#ffedd5;color:#9a3412}
  .mapsw .sub{font-size:12.5px;color:var(--muted);line-height:1.45;margin-top:2px}
  .mapsw.off .sub{color:#7a4d12}
  /* stato dell'OCR: lo stesso blocco dello switch, ma con il motore a destra */
  .ocrsw .eng{font-size:11px;font-weight:800;letter-spacing:.04em;border-radius:999px;
              padding:2px 8px;background:#eef2ff;color:#3730a3;margin-left:auto}
  .ocrsw.none .eng{background:#fee2e2;color:#991b1b}
  .ocrsw.off .eng{background:#f1e9f7;color:var(--brand-dk)}

  /* modale "Tag da anonimizzare": legenda dei 23 tag con toggle per tipo */
  .cfg-card.wide{width:600px}
  .tg-sub{font-size:12.5px;color:var(--muted);margin:-8px 0 14px;line-height:1.5}
  .tg-bar{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
  .tg-bar .mini{background:#fff;color:var(--muted);border:1px solid var(--line);
                border-radius:8px;padding:5px 10px;font-size:12.5px;font-weight:600}
  .tg-bar .mini:hover{border-color:#cfd5e0;color:var(--ink)}
  .tg-bar .count{margin-left:auto;font-size:12.5px;color:var(--soft);font-weight:600}
  .tg-list{max-height:46vh;overflow:auto;border:1px solid var(--line);border-radius:11px;
           padding:5px;background:#fcfcfe}
  .tg-row{display:flex;align-items:flex-start;gap:10px;padding:8px 9px;border-radius:9px;
          cursor:pointer;user-select:none;transition:.12s}
  .tg-row:hover{background:#f4f2f9}
  .tg-row input{margin:3px 0 0;accent-color:var(--brand);width:15px;height:15px;flex:none;cursor:pointer}
  .tg-row .sw{width:10px;height:10px;border-radius:3px;margin-top:5px;flex:none}
  .tg-row .txt{min-width:0}
  .tg-row .nm{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;font-weight:700}
  .tg-row .ds{font-size:12.5px;color:var(--muted);line-height:1.45}
  .tg-row .ex{font-size:11.5px;color:var(--soft)}
  .tg-row.off{opacity:.55}
  .tg-row.off .nm{text-decoration:line-through}
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="logo"><img src="/assets/mascot_shield.png" alt="rizzo-pii"
         onerror="this.parentNode.textContent='🦔'"></div>
    <div>
      <h1>Rizzo PII <span class="ver">v__VERSION__</span></h1>
      <div class="tag" data-i18n="tagline">modello locale su CPU · GDPR compliant</div>
    </div>
    <span class="badge"><span class="dot"></span> <span data-i18n="badge">100% in locale</span></span>
    <div class="topright">
      <select id="lang" class="lang" title="Lingua / Language" aria-label="Lingua / Language">
        <option value="it">🇮🇹 Italiano</option>
        <option value="en">🇬🇧 English</option>
      </select>
      <button class="gear" id="tagsBtn" title="Tag da anonimizzare" aria-label="PII tags" onclick="openTags()">🏷️</button>
      <button class="gear" id="gearBtn" title="Configurazione server" aria-label="Server config" onclick="openConfig()">⚙️</button>
      <span class="info" id="infoBtn" tabindex="0" role="button" aria-label="Avviso / info">⚠️<span class="tip" data-i18n="notice"></span></span>
    </div>
  </header>

  <div class="tabs">
    <div class="tab on" id="tab1" onclick="showTab(1)">
      <span class="num">1</span> <span data-i18n="tab1">Anonimizza</span> <span class="arrow">→</span>
    </div>
    <div class="tab" id="tab2" onclick="showTab(2)">
      <span class="num">2</span> <span data-i18n="tab2">Ripristina la risposta</span>
    </div>
  </div>

  <!-- ============ PANE 1: ANONIMIZZA ============ -->
  <div class="pane on" id="pane1">
    <div class="workspace">
      <!-- input -->
      <div class="card">
        <div class="hd"><h2 data-i18n="in_title">① Il tuo documento</h2>
          <div class="right">
            <span class="hint" id="inHint" data-i18n="in_hint">incolla testo o trascina un PDF / .md</span>
            <div class="seg-tabs" id="srcTabs" style="display:none">
              <button class="on" id="sPdf" onclick="setSrcView('pdf')" data-i18n="v_pdf">Anteprima PDF</button>
              <button id="sText" onclick="setSrcView('text')" data-i18n="v_raw">Testo</button>
            </div>
          </div></div>
        <div class="bd">
          <div class="view pdfview" id="pdfSrcView" style="display:none"></div>
          <textarea id="src" data-i18n-ph="src_ph" placeholder="Incolla qui il testo dell'atto, del contratto o della sentenza…&#10;&#10;Oppure trascina un PDF nell'area qui sotto."></textarea>
          <label class="drop" id="drop">
            <span class="ic">📄</span>
            <span id="dropTxt">Trascina un <b>PDF</b> o un <b>.md</b> qui, oppure <b>scegli un file</b></span>
            <input type="file" id="pdf" accept=".pdf,.md,.markdown,.txt,application/pdf,text/markdown,text/plain" hidden>
          </label>
          <div class="mapsw" id="mapSw">
            <button class="tsw" id="mapToggle" role="switch" aria-checked="true"
                    aria-labelledby="mapTtl" onclick="toggleMapping()"><span class="knob"></span></button>
            <div>
              <div class="ttl" id="mapTtl"><span data-i18n="map_ttl">Dizionario reversibile</span>
                <span class="st" id="mapState">ATTIVO</span></div>
              <div class="sub" id="mapSub">Ogni PII riceve un ID: potrai ripristinare i valori veri dalla risposta dell'LLM.</div>
            </div>
          </div>
          <div class="mapsw ocrsw" id="ocrSw">
            <button class="tsw" id="ocrToggle" role="switch" aria-checked="true"
                    aria-labelledby="ocrTtl" onclick="toggleOcr()"><span class="knob"></span></button>
            <div>
              <div class="ttl" id="ocrTtl"><span data-i18n="ocr_ttl">OCR automatico</span>
                <span class="st" id="ocrState">ATTIVO</span>
                <span class="eng" id="ocrEngine">—</span></div>
              <div class="sub" id="ocrSub">Riconosce il testo dei PDF scansionati.</div>
            </div>
          </div>
          <div class="row">
            <button class="btn lg" id="go">🛡️ <span data-i18n="go">Anonimizza</span></button>
            <button class="ghost" id="clear" data-i18n="clear">Pulisci</button>
            <span class="hint"><span class="kbd">Ctrl</span>+<span class="kbd">Enter</span></span>
          </div>
        </div>
      </div>

      <!-- output -->
      <div class="card out">
        <div class="hd">
          <h2 data-i18n="out_title">② Risultato</h2>
          <div class="right">
            <div class="seg-tabs">
              <button class="on" id="vPrev" onclick="setView('prev')" data-i18n="v_prev">Anteprima</button>
              <button id="vText" onclick="setView('text')" data-i18n="v_text">Testo da copiare</button>
              <button id="vPdf" onclick="setView('pdf')" data-i18n="v_opdf">PDF censurato</button>
            </div>
          </div>
        </div>
        <div class="bd">
          <div class="view" id="viewPrev">
            <div class="empty" id="emptyPrev">
              <img src="/assets/mascot_doc.png" alt="" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'big',textContent:'🕵️'}))">
              <div data-i18n="empty_prev">L'anteprima con le PII evidenziate apparirà qui.</div>
            </div>
            <div class="preview" id="prev" style="display:none"></div>
          </div>
          <div class="view pdfview" id="pdfOutView" style="display:none"></div>
          <textarea class="mono" id="anon" style="display:none" readonly
                    data-i18n-ph="anon_ph" placeholder="Il testo anonimizzato apparirà qui."></textarea>
          <div class="row">
            <button class="btn" id="copy">📋 <span data-i18n="copy">Copia per ChatGPT</span></button>
            <button class="ghost" id="dlpdf">📄 <span data-i18n="dlpdf">Scarica PDF anonimizzato</span></button>
            <button class="ghost" id="dl">⬇️ <span data-i18n="dl">Scarica dizionario</span></button>
            <span class="hint" id="ulock"></span>
          </div>
        </div>
      </div>
    </div>

    <!-- dizionario: staccato, a tutta larghezza sotto le due colonne, scrollabile -->
    <div class="card dict" id="dictCard" style="display:none">
      <div class="hd">
        <h2 data-i18n="dict_title">Dizionario reversibile</h2>
        <div class="right hint" id="dictHint" data-i18n="dict_hint">resta solo qui, in locale</div>
      </div>
      <div class="bd">
        <div class="meta" id="meta"></div>
        <div class="legend" id="legend"></div>
        <div class="callout" id="dictNone" style="display:none;margin:0 16px 14px">
          <span class="ic">🔒</span><div data-i18n="dict_off">Nessun dizionario: l'anonimizzazione di questo testo è <b>definitiva</b>.</div>
        </div>
        <div class="tablewrap" id="tablewrap">
          <table><thead><tr><th data-i18n="th_id">ID</th><th data-i18n="th_val">Valore originale</th><th data-i18n="th_type">Tipo</th></tr></thead>
          <tbody id="maprows"></tbody></table>
        </div>
      </div>
    </div>
  </div>

  <!-- ============ PANE 2: RIPRISTINA ============ -->
  <div class="pane" id="pane2">
    <div class="callout" id="rOff" style="display:none">
      <span class="ic">🔒</span>
      <div data-i18n="r_off">Lo switch <b>Dizionario reversibile</b> è su DISATTIVO: le nuove
      anonimizzazioni non producono chiavi. Qui puoi comunque ripristinare con un dizionario
      <b>.json salvato in precedenza</b>.</div>
    </div>
    <div class="callout">
      <span class="ic">💡</span>
      <div data-i18n="callout">Incolla qui la <b>risposta dell'LLM</b> (che contiene i placeholder come
      <span class="kbd">[FULLNAME_1]</span>): l'app rimette i valori veri usando il dizionario
      di questa sessione. Se hai chiuso e riaperto l'app, <b>carica il dizionario .json</b> che
      avevi salvato.</div>
    </div>
    <div class="grid">
      <div class="card">
        <div class="hd"><h2 data-i18n="r_title1">Risposta con i placeholder</h2>
          <div class="right">
            <label class="chip" style="cursor:pointer"><span data-i18n="loaddict">📁 Carica dizionario</span>
              <input type="file" id="dictFile" accept="application/json" hidden></label>
          </div></div>
        <div class="bd">
          <textarea id="rin" data-i18n-ph="rin_ph" placeholder="Incolla qui la risposta di ChatGPT…"></textarea>
          <div class="row">
            <button class="btn lg" id="rev">🔓 <span data-i18n="rev">Ripristina valori</span></button>
            <button class="ghost" id="rclear" data-i18n="clear">Pulisci</button>
            <span class="hint" id="dictInfo"></span>
          </div>
        </div>
      </div>
      <div class="card">
        <div class="hd"><h2 data-i18n="r_title2">Testo ripristinato</h2></div>
        <div class="bd">
          <div class="view"><div class="preview" id="rout">
            <div class="empty"><img src="/assets/mascot_doc.png" alt="" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'big',textContent:'🔓'}))">
            <div data-i18n="empty_rout">Il testo con i valori reali apparirà qui.</div></div>
          </div></div>
          <div class="row"><button class="btn" id="rcopy">📋 <span data-i18n="rcopy">Copia testo ripristinato</span></button></div>
        </div>
      </div>
    </div>
  </div>

  <div class="credits">Realizzato da <b>Simone Rizzo</b> · sponsorizzato da <b>Rizzo AI Academy</b> · <span class="u">www.rizzoaiacademy.com</span></div>
</div>

<div id="toast"></div>

<!-- config modal -->
<div class="cfg-overlay" id="cfgOverlay">
  <div class="cfg-card">
    <h3>⚙️ <span data-i18n="cfg_title">Configurazione server</span></h3>
    <div class="cfg-row">
      <label data-i18n="cfg_host">Indirizzo</label>
      <input id="cfgHost" type="text" value="127.0.0.1" spellcheck="false">
    </div>
    <div class="cfg-row">
      <label data-i18n="cfg_port">Porta</label>
      <input id="cfgPort" type="number" min="1024" max="65535" value="5005">
    </div>
    <div class="cfg-status" id="cfgStatus"></div>
    <div class="cfg-btns">
      <button class="btn" id="cfgSave" onclick="saveConfig()">💾 <span data-i18n="cfg_save">Salva</span></button>
      <button class="ghost" id="cfgCheck" onclick="checkPort()"><span data-i18n="cfg_check">Verifica porta</span></button>
      <button class="ghost" onclick="closeConfig()" data-i18n="cfg_cancel">Annulla</button>
    </div>
    <div class="cfg-note" data-i18n="cfg_restart_note">Le modifiche avranno effetto al prossimo avvio.</div>
  </div>
</div>

<!-- tags modal: legenda + selezione dei tag da anonimizzare -->
<div class="cfg-overlay" id="tagsOverlay">
  <div class="cfg-card wide">
    <h3>🏷️ <span data-i18n="tg_title">Tag da anonimizzare</span></h3>
    <div class="tg-sub" data-i18n="tg_sub">Deseleziona i tipi che vuoi <b>lasciare in chiaro</b>: verranno comunque rilevati, ma non sostituiti da un placeholder.</div>
    <div class="tg-bar">
      <button class="mini" onclick="setAllTags(true)" data-i18n="tg_all">Seleziona tutti</button>
      <button class="mini" onclick="setAllTags(false)" data-i18n="tg_none">Nessuno</button>
      <span class="count" id="tgCount"></span>
    </div>
    <div class="tg-list" id="tgList"></div>
    <div class="cfg-status" id="tgStatus"></div>
    <div class="cfg-btns" style="margin-top:14px">
      <button class="btn" onclick="saveTags()">💾 <span data-i18n="cfg_save">Salva</span></button>
      <button class="ghost" onclick="closeTags()" data-i18n="cfg_cancel">Annulla</button>
    </div>
    <div class="cfg-note" id="tgNote"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let DATA = null;            // ultimo risultato analyze
let MAP = {};              // {placeholder -> valore} sessione corrente
const off = new Set();     // label nascoste nella preview
let L = 'it';              // lingua UI corrente
let TAGS = [];             // legenda dei tag servita da /settings
let EXCL = new Set();      // tag da NON anonimizzare (scelta corrente)
let TAGS_LOADED = false;   // /settings ha risposto -> possiamo mandare gli override
let MAPPING = true;        // dizionario reversibile on/off (switch nella card di input)
let OCR = true;            // OCR automatico sulle scansioni on/off
let OCR_INFO = null;       // {available, engine, detail} da /settings
let SRC_DOC = null;        // PDF caricato, renderizzato dal server (colonna sinistra)
let OUT_DOC = null;        // PDF anonimizzato (colonna destra), generato pigramente
let VIEW = 'prev';         // vista attiva a destra: prev | text | pdf

/* ---- i18n (IT default, EN opzionale) ---- */
const T = {
 it:{
  tagline:"modello locale su CPU · GDPR compliant", badge:"100% in locale",
  notice:"<b>Versione in sviluppo.</b> Il modello AI non è perfetto e può commettere errori: verifica sempre il risultato prima di usarlo. Queste sono le prime versioni e il progetto è completamente <b>open source</b>. Se ti è utile, <b>lascia una ⭐ alla repo</b> e contribuisci a migliorarlo: <a href=\"https://github.com/Rizzo-AI-Academy/rizzo-pii\" target=\"_blank\" rel=\"noopener\">apri la repo su GitHub ↗</a>",
  tab1:"Anonimizza", tab2:"Ripristina la risposta",
  in_title:"① Il tuo documento", in_hint:"incolla testo o trascina un PDF / .md",
  src_ph:"Incolla qui il testo dell'atto, del contratto o della sentenza…\n\nOppure trascina un PDF o un file .md nell'area qui sotto.",
  drop:"Trascina un <b>PDF</b> o un <b>.md</b> qui, oppure <b>scegli un file</b>",
  go:"Anonimizza", clear:"Pulisci",
  out_title:"② Risultato", v_prev:"Anteprima", v_text:"Testo da copiare",
  v_pdf:"Anteprima PDF", v_raw:"Testo", v_opdf:"PDF censurato",
  t_prev_loading:"Renderizzo il PDF…", t_prev_err:"Anteprima del PDF non disponibile",
  t_pdf_render:"Genero il PDF anonimizzato…",
  t_pdf_locked:"PDF protetto da password: rimuovi la protezione e riprova",
  t_zoom_out:"Riduci zoom", t_zoom_in:"Aumenta zoom", t_zoom_reset:"Reimposta zoom",
  empty_prev:"L'anteprima con le PII evidenziate apparirà qui.",
  anon_ph:"Il testo anonimizzato apparirà qui.",
  copy:"Copia per ChatGPT", dl:"Scarica dizionario", dlpdf:"Scarica PDF anonimizzato",
  t_pdf_making:"Genero il PDF…", t_pdf_ok:"PDF anonimizzato scaricato",
  t_pdf_err:"Errore nella creazione del PDF",
  t_pdf_warn:(r,s)=>"PDF scaricato · ATTENZIONE: "+(r+s)+" valori sono rimasti in chiaro"
    +" ("+r+" non redatti, "+s+" troppo corti per essere cercati): controlla il file"
    +" prima di condividerlo",
  dict_title:"Dizionario reversibile", dict_hint:"resta solo qui, in locale",
  th_id:"ID", th_val:"Valore originale", th_type:"Tipo",
  callout:"Incolla qui la <b>risposta dell'LLM</b> (che contiene i placeholder come <span class=\"kbd\">[FULLNAME_1]</span>): l'app rimette i valori veri usando il dizionario di questa sessione. Se hai chiuso e riaperto l'app, <b>carica il dizionario .json</b> che avevi salvato.",
  r_title1:"Risposta con i placeholder", loaddict:"📁 Carica dizionario",
  rin_ph:"Incolla qui la risposta di ChatGPT…",
  rev:"Ripristina valori", r_title2:"Testo ripristinato",
  empty_rout:"Il testo con i valori reali apparirà qui.", rcopy:"Copia testo ripristinato",
  st_ent:"entità", st_uniq:"valori unici", st_model:"dal modello",
  st_regex:"da regex/checksum", st_chars:"caratteri", analyzing:"Analizzo…",
  t_need_input:"Inserisci del testo o un PDF", t_error:"Errore",
  t_copied:"Testo anonimizzato copiato", t_need_anon:"Prima anonimizza un testo",
  t_nothing_dl:"Niente da scaricare", t_dl_ok:"Dizionario scaricato",
  t_paste_restore:"Incolla la risposta da ripristinare",
  t_no_dict:"Nessun dizionario: caricane uno .json", t_restored:"Valori ripristinati",
  t_nothing_copy:"Niente da copiare", t_restored_copied:"Testo ripristinato copiato",
  t_dict_loaded:"Dizionario caricato", t_json_invalid:"JSON non valido",
  t_drag_pdf:"Formato non supportato: usa PDF, .md o .txt",
  pii_found:(n,u)=>n+" PII trovate · "+u+" valori unici",
  dict_n:n=>n+" ID nel dizionario",
  dict_loaded_n:n=>"dizionario caricato · "+n+" ID",
  dict_session:n=>"dizionario sessione · "+n+" ID",
  chars:n=>n.toLocaleString('it'),
  cfg_title:"Configurazione server", cfg_host:"Indirizzo", cfg_port:"Porta",
  cfg_check:"Verifica porta", cfg_save:"Salva", cfg_cancel:"Annulla",
  cfg_available:"Porta disponibile ✓", cfg_in_use:"Porta occupata ✗",
  cfg_saved:"Configurazione salvata (riavvia per applicare)",
  cfg_restart_note:"Le modifiche avranno effetto al prossimo avvio.",
  tg_title:"Tag da anonimizzare",
  tg_sub:"Deseleziona i tipi che vuoi <b>lasciare in chiaro</b>: verranno comunque rilevati, ma non sostituiti da un placeholder.",
  tg_all:"Seleziona tutti", tg_none:"Nessuno",
  tg_saved:"Selezione salvata", tg_err:"Salvataggio non riuscito",
  tg_note:p=>"Salvato in "+p+" · vale anche per l'API /analyze.",
  tg_env:"⚠️ La variabile d'ambiente PII_EXCLUDE_TAGS ha la precedenza al prossimo avvio.",
  tg_count:(on,tot)=>on+" di "+tot+" tag anonimizzati",
  st_excl:"tag esclusi",
  map_ttl:"Dizionario reversibile", map_on:"ATTIVO", map_off:"DISATTIVO",
  map_sub_on:"Ogni PII riceve un ID: potrai ripristinare i valori veri dalla risposta dell'LLM.",
  map_sub_off:"Anonimizzazione <b>definitiva</b>: nessuna chiave placeholder → valore viene creata, salvata o scaricabile. Il ripristino non sarà possibile.",
  t_map_on:"Dizionario reversibile attivo", t_map_off:"Dizionario disattivato: anonimizzazione definitiva",
  dict_off:"Nessun dizionario: l'anonimizzazione di questo testo è <b>definitiva</b>.",
  dict_off_hint:"lo switch è su DISATTIVO",
  r_off:"Lo switch <b>Dizionario reversibile</b> è su DISATTIVO: le nuove anonimizzazioni non producono chiavi. Qui puoi comunque ripristinare con un dizionario <b>.json salvato in precedenza</b>.",
  ocr_ttl:"OCR automatico", ocr_on:"ATTIVO", ocr_off:"DISATTIVO",
  ocr_sub_on:"Riconosce il testo dei PDF scansionati (quelli senza layer testuale) e lo rende anonimizzabile.",
  ocr_sub_off:"L'OCR è disattivato: sui PDF scansionati l'app non trova testo e lo segnala.",
  ocr_none:"nessun motore installato",
  ocr_engine_t:"Tesseract", ocr_engine_o:"ONNX",
  t_ocr_on:"OCR automatico attivo", t_ocr_off:"OCR automatico disattivato",
  ocr_badge:(e,p,s)=>"OCR · "+e+" · "+p+" pagine · "+s+" s",
  ocr_burnin:"Le pagine scansionate sono state rasterizzate per coprire le PII: in quelle pagine il testo non è più selezionabile.",
  t_ocr_missing:"Serve l'OCR ma nessun motore è installato. Installa Tesseract oppure rapidocr-onnxruntime.",
 },
 en:{
  tagline:"local model on CPU · GDPR compliant", badge:"100% local",
  notice:"<b>Work in progress.</b> The AI model isn't perfect and can make mistakes: always double-check the result before relying on it. These are the very first versions and the project is fully <b>open source</b>. If you find it useful, <b>leave a ⭐ on the repo</b> and help improve it: <a href=\"https://github.com/Rizzo-AI-Academy/rizzo-pii\" target=\"_blank\" rel=\"noopener\">open the repo on GitHub ↗</a>",
  tab1:"Anonymize", tab2:"Restore the answer",
  in_title:"① Your document", in_hint:"paste text or drop a PDF / .md",
  src_ph:"Paste here the text of the deed, contract or judgment…\n\nOr drop a PDF or a .md file onto the area below.",
  drop:"Drop a <b>PDF</b> or a <b>.md</b> here, or <b>choose a file</b>",
  go:"Anonymize", clear:"Clear",
  out_title:"② Result", v_prev:"Preview", v_text:"Text to copy",
  v_pdf:"PDF preview", v_raw:"Text", v_opdf:"Redacted PDF",
  t_prev_loading:"Rendering the PDF…", t_prev_err:"PDF preview not available",
  t_pdf_render:"Building the anonymized PDF…",
  t_pdf_locked:"Password-protected PDF: remove the protection and try again",
  t_zoom_out:"Zoom out", t_zoom_in:"Zoom in", t_zoom_reset:"Reset zoom",
  empty_prev:"The preview with highlighted PII will appear here.",
  anon_ph:"The anonymized text will appear here.",
  copy:"Copy for ChatGPT", dl:"Download dictionary", dlpdf:"Download anonymized PDF",
  t_pdf_making:"Building the PDF…", t_pdf_ok:"Anonymized PDF downloaded",
  t_pdf_err:"Error while creating the PDF",
  t_pdf_warn:(r,s)=>"PDF downloaded · WARNING: "+(r+s)+" values were left in clear"
    +" ("+r+" not redacted, "+s+" too short to be searched safely): check the file"
    +" before sharing it",
  dict_title:"Reversible dictionary", dict_hint:"stays here only, locally",
  th_id:"ID", th_val:"Original value", th_type:"Type",
  callout:"Paste here the <b>LLM's answer</b> (containing placeholders like <span class=\"kbd\">[FULLNAME_1]</span>): the app puts the real values back using this session's dictionary. If you closed and reopened the app, <b>load the .json dictionary</b> you saved.",
  r_title1:"Answer with placeholders", loaddict:"📁 Load dictionary",
  rin_ph:"Paste ChatGPT's answer here…",
  rev:"Restore values", r_title2:"Restored text",
  empty_rout:"The text with the real values will appear here.", rcopy:"Copy restored text",
  st_ent:"entities", st_uniq:"unique values", st_model:"from the model",
  st_regex:"from regex/checksum", st_chars:"characters", analyzing:"Analyzing…",
  t_need_input:"Enter some text or a PDF", t_error:"Error",
  t_copied:"Anonymized text copied", t_need_anon:"Anonymize a text first",
  t_nothing_dl:"Nothing to download", t_dl_ok:"Dictionary downloaded",
  t_paste_restore:"Paste the answer to restore",
  t_no_dict:"No dictionary: load a .json one", t_restored:"Values restored",
  t_nothing_copy:"Nothing to copy", t_restored_copied:"Restored text copied",
  t_dict_loaded:"Dictionary loaded", t_json_invalid:"Invalid JSON",
  t_drag_pdf:"Unsupported format: use PDF, .md or .txt",
  pii_found:(n,u)=>n+" PII found · "+u+" unique values",
  dict_n:n=>n+" IDs in the dictionary",
  dict_loaded_n:n=>"dictionary loaded · "+n+" IDs",
  dict_session:n=>"session dictionary · "+n+" IDs",
  chars:n=>n.toLocaleString('en'),
  cfg_title:"Server configuration", cfg_host:"Host", cfg_port:"Port",
  cfg_check:"Check port", cfg_save:"Save", cfg_cancel:"Cancel",
  cfg_available:"Port available ✓", cfg_in_use:"Port in use ✗",
  cfg_saved:"Config saved (restart to apply)",
  cfg_restart_note:"Changes take effect on next startup.",
  tg_title:"Tags to anonymize",
  tg_sub:"Untick the types you want to <b>leave in clear text</b>: they are still detected, but not replaced by a placeholder.",
  tg_all:"Select all", tg_none:"None",
  tg_saved:"Selection saved", tg_err:"Could not save",
  tg_note:p=>"Saved to "+p+" · also applies to the /analyze API.",
  tg_env:"⚠️ The PII_EXCLUDE_TAGS environment variable takes precedence on next startup.",
  tg_count:(on,tot)=>on+" of "+tot+" tags anonymized",
  st_excl:"excluded tags",
  map_ttl:"Reversible dictionary", map_on:"ON", map_off:"OFF",
  map_sub_on:"Every PII gets an ID: you will be able to restore the real values from the LLM's answer.",
  map_sub_off:"<b>Irreversible</b> anonymization: no placeholder → value key is created, stored or downloadable. Restoring will not be possible.",
  t_map_on:"Reversible dictionary on", t_map_off:"Dictionary off: anonymization is irreversible",
  dict_off:"No dictionary: anonymization of this text is <b>irreversible</b>.",
  dict_off_hint:"the switch is OFF",
  r_off:"The <b>Reversible dictionary</b> switch is OFF: new anonymizations produce no keys. You can still restore here using a <b>.json dictionary saved earlier</b>.",
  ocr_ttl:"Automatic OCR", ocr_on:"ON", ocr_off:"OFF",
  ocr_sub_on:"Recognizes the text of scanned PDFs (the ones with no text layer) and makes them anonymizable.",
  ocr_sub_off:"OCR is off: on scanned PDFs the app finds no text and says so.",
  ocr_none:"no engine installed",
  ocr_engine_t:"Tesseract", ocr_engine_o:"ONNX",
  t_ocr_on:"Automatic OCR on", t_ocr_off:"Automatic OCR off",
  ocr_badge:(e,p,s)=>"OCR · "+e+" · "+p+" pages · "+s+" s",
  ocr_burnin:"Scanned pages were rasterized to cover the PII: in those pages the text is no longer selectable.",
  t_ocr_missing:"OCR is needed but no engine is installed. Install Tesseract or rapidocr-onnxruntime.",
 }
};
const tt=k=>T[L][k];

function routEmpty(){
  return '<div class="empty"><img src="/assets/mascot_doc.png" alt="" onerror="this.replaceWith(Object.assign(document.createElement(\'div\'),{className:\'big\',textContent:\'🔓\'}))"><div>'+tt('empty_rout')+'</div></div>';
}

function applyLang(l){
  L=(l==='en')?'en':'it';
  localStorage.setItem('pii_lang',L);
  document.documentElement.lang=L;$('lang').value=L;
  document.querySelectorAll('[data-i18n]').forEach(el=>{
    const v=T[L][el.getAttribute('data-i18n')]; if(v!=null) el.innerHTML=v;});
  document.querySelectorAll('[data-i18n-ph]').forEach(el=>{
    const v=T[L][el.getAttribute('data-i18n-ph')]; if(v!=null) el.placeholder=v;});
  if(!$('pdf').files.length) $('dropTxt').innerHTML=tt('drop');   // dropzone: solo se nessun file
  if(!$('rout')._raw) $('rout').innerHTML=routEmpty();
  renderMapping();
  renderOcr();
  if(TAGS.length && $('tagsOverlay').classList.contains('open')) renderTags();
  if(DATA) render();
}

/* ---- scroll sincronizzato editor (sx) <-> anteprima/testo (dx) ---- */
let syncing=false;
function matchScroll(target,from){
  const rf=from.scrollHeight-from.clientHeight, rt=target.scrollHeight-target.clientHeight;
  target.scrollTop = rf>0 ? (from.scrollTop/rf)*rt : 0;
}
function linkScroll(a,b){
  a.addEventListener('scroll',()=>{
    if(syncing)return; syncing=true; matchScroll(b,a);
    setTimeout(()=>syncing=false,0);});   // reset robusto (rAF puo' non scattare in bg)
}

/* ---- colore deterministico per tipo di tag ---- */
function hue(s){let h=0;for(const c of s)h=(h*31+c.charCodeAt(0))%360;return h;}
function colors(label){const h=hue(label);
  return {bg:`hsl(${h} 56% 96%)`,bd:`hsl(${h} 42% 81%)`,tx:`hsl(${h} 40% 38%)`};}

function toast(msg,ok=true,ms=1800){const t=$('toast');t.textContent=msg;t.className='show'+(ok?' ok':'');
  clearTimeout(t._t);t._t=setTimeout(()=>t.className='',ms);}

/* ---- render di un PDF a video ----------------------------------------------
   Le pagine arrivano gia' renderizzate dal server (/doc/<id>/page/<n>.png):
   niente viewer PDF del browser (dentro Tauri non c'e' garanzia che ci sia) e
   niente libreria JS esterna (l'app e' offline, nessuna CDN raggiungibile).
   `loading=lazy` -> un PDF di 200 pagine non scarica 200 immagini all'apertura. */
function renderPages(el,doc){
  el.innerHTML='';
  el.appendChild(zoomBar(el));
  for(let i=0;i<doc.n_pages;i++){
    const pg=document.createElement('div');pg.className='pg';
    const im=document.createElement('img');
    im.loading='lazy';im.alt=(i+1)+' / '+doc.n_pages;   // alt = pagina: se il render manca si vede quale
    im.src=`/doc/${doc.doc_id}/page/${i}.png`;
    pg.appendChild(im);
    const n=document.createElement('span');n.className='pgn';
    n.textContent=(i+1)+' / '+doc.n_pages;pg.appendChild(n);
    el.appendChild(pg);
  }
  el.scrollTop=0;
}

/* barra zoom (issue #92): +/- e reset sulla %, piu' ctrl/cmd+rotellina sull'intera vista.
   Il livello vive in --z su el stesso, letto da .pdfview .pg{width:...} in CSS; si azzera
   a ogni nuovo rendering (nuovo documento = vista pulita, come per gli altri stati del pane). */
function zoomBar(el){
  const bar=document.createElement('div');bar.className='zoomctl';
  const pill=document.createElement('div');pill.className='pill';
  const out=document.createElement('button');out.type='button';out.textContent='−';out.title=tt('t_zoom_out');
  const pct=document.createElement('span');pct.className='zpct';pct.title=tt('t_zoom_reset');
  const inn=document.createElement('button');inn.type='button';inn.textContent='+';inn.title=tt('t_zoom_in');
  const setZ=z=>{z=Math.max(.5,Math.min(3,z));el.style.setProperty('--z',z);pct.textContent=Math.round(z*100)+'%';};
  const curZ=()=>parseFloat(el.style.getPropertyValue('--z'))||1;
  out.onclick=()=>setZ(curZ()-.25);
  inn.onclick=()=>setZ(curZ()+.25);
  pct.onclick=()=>setZ(1);
  el.onwheel=e=>{if(!e.ctrlKey&&!e.metaKey)return;e.preventDefault();setZ(curZ()+(e.deltaY<0?.1:-.1));};
  pill.append(out,pct,inn);bar.appendChild(pill);
  setZ(1);
  return bar;
}
function busy(el,msg){
  el.innerHTML='<div class="pdfbusy"><span class="spin"></span>'+msg+'</div>';}

/* ---- tabs ---- */
function showTab(n){
  $('tab1').classList.toggle('on',n===1);$('tab2').classList.toggle('on',n===2);
  $('pane1').classList.toggle('on',n===1);$('pane2').classList.toggle('on',n===2);
}

/* ---- documento caricato: anteprima renderizzata (sx) ----------------------
   Appena si sceglie un PDF lo si manda a /preview: il server lo tiene in memoria,
   ne ritorna il testo estratto e le pagine da renderizzare. Da li' in poi la
   colonna sinistra ha due viste, "Anteprima PDF" (default) e "Testo".
   Per .md/.txt non c'e' niente da renderizzare: resta la sola textarea. */
function setSrcView(v){
  const p=(v==='pdf'&&SRC_DOC);
  $('sPdf').classList.toggle('on',!!p);$('sText').classList.toggle('on',!p);
  $('pdfSrcView').style.display=p?'':'none';
  $('src').style.display=p?'none':'';
}

async function onFile(f){
  $('dropTxt').innerHTML='📎 <b>'+escapeHtml(f.name)+'</b>';
  SRC_DOC=null;$('srcTabs').style.display='none';$('inHint').style.display='';
  setSrcView('text');
  if(!(f.type==='application/pdf'||/\.pdf$/i.test(f.name)))return;
  $('inHint').textContent=tt('t_prev_loading');
  busy($('pdfSrcView'),tt('t_prev_loading'));
  $('srcTabs').style.display='';$('pdfSrcView').style.display='';$('src').style.display='none';
  $('sPdf').classList.add('on');$('sText').classList.remove('on');
  try{
    const fd=new FormData();fd.append('pdf',f);
    const r=await fetch('/preview',{method:'POST',body:fd});
    const d=await r.json();
    if(d.code==='pdf_locked'){        // non si legge: niente da anonimizzare, via il file
      SRC_DOC=null;$('srcTabs').style.display='none';setSrcView('text');
      $('pdf').value='';$('dropTxt').innerHTML=tt('drop');
      $('inHint').innerHTML=tt('in_hint');$('inHint').style.display='';
      toast(tt('t_pdf_locked'),false,4000);return;
    }
    if(!r.ok)throw new Error(d.error||'');
    SRC_DOC=d;$('src').value=d.text||'';
    renderPages($('pdfSrcView'),d);
    if(d.ocr){                       // il PDF era una scansione: l'OCR ha fatto il suo lavoro
      const badge='🔍 '+T[L].ocr_badge(d.ocr.engine==='tesseract'?tt('ocr_engine_t'):tt('ocr_engine_o'),
                                       d.ocr.pages,d.ocr.seconds);
      $('inHint').style.display='';$('inHint').textContent=badge;
      toast(badge,true,4000);
    } else $('inHint').style.display='none';
  }catch(e){
    SRC_DOC=null;$('srcTabs').style.display='none';setSrcView('text');
    $('inHint').innerHTML=tt('in_hint');$('inHint').style.display='';
    toast(tt('t_prev_err'),false);          // il file resta valido: l'analisi funziona lo stesso
  }
}

/* ---- analyze ---- */
async function run(){
  const file=$('pdf').files[0];const text=$('src').value.trim();
  if(!file&&!text){toast(tt('t_need_input'),false);return;}
  $('go').disabled=true;const old=$('go').innerHTML;
  $('go').innerHTML='<span class="spin"></span> '+tt('analyzing');
  try{
    let resp;
    // override della UI. Se /settings non ha risposto non mandiamo nulla: comanda il server.
    const excl=TAGS_LOADED?[...EXCL]:null;
    if(file){const fd=new FormData();fd.append('pdf',file);
      if(excl){fd.append('exclude_tags',excl.join(','));fd.append('include_mapping',MAPPING?'1':'0');}
      resp=await fetch('/analyze',{method:'POST',body:fd});}
    else{const body={text};if(excl){body.exclude_tags=excl;body.include_mapping=MAPPING;}
      resp=await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify(body)});}
    const d=await resp.json();
    if(!resp.ok){
      const long=(d.code==='ocr_unavailable'||d.code==='ocr_disabled');
      toast(d.error||tt('t_error'),false,long?9000:1800);return;}
    if(d.source_text&&file)$('src').value=d.source_text;
    DATA=d;off.clear();
    // senza dizionario non tocchiamo MAP ne' il localStorage: nessuna chiave nuova nasce
    if(d.mapping_enabled!==false){MAP=d.mapping;localStorage.setItem('pii_map',JSON.stringify(MAP));}
    render();
    toast(T[L].pii_found(d.n_entities,d.n_unique));
  }catch(e){toast(tt('t_error')+': '+e.message,false);}
  finally{$('go').disabled=false;$('go').innerHTML=old;}
}

function render(){
  const d=DATA;
  OUT_DOC=null;$('pdfOutView').innerHTML='';  // risultato nuovo -> il PDF censurato va rifatto
  $('dictCard').style.display='';            // mostra la card dizionario (sotto le due colonne)
  document.querySelector('.app').classList.add('has-result');  // -> scroll pagina, niente schiacciamento
  // preview evidenziata
  const prev=$('prev');prev.innerHTML='';prev.style.display='';$('emptyPrev').style.display='none';
  for(const s of d.segments){
    if(s.label){
      const c=colors(s.label);const sp=document.createElement('span');
      sp.className='ph'+(off.has(s.label)?' dim':'');
      sp.style.background=c.bg;sp.style.borderColor=c.bd;sp.style.color=c.tx;
      // senza dizionario il server non manda il valore originale: niente da mostrare al passaggio
      sp.title=(s.t?s.t+'\n':'')+`(${s.src}${s.validated?' · checksum ✓':''})`;
      sp.innerHTML=s.ph.replace(/[\[\]]/g,'')+(s.validated?'<span class="ck">✓</span>':'');
      prev.appendChild(sp);
    }else prev.appendChild(document.createTextNode(s.t));
  }
  // testo da copiare
  $('anon').value=d.anonymized_text;
  // meta
  $('meta').innerHTML=
    `<span class="stat"><b>${d.n_entities}</b> ${tt('st_ent')}</span>`+
    `<span class="stat"><b>${d.n_unique}</b> ${tt('st_uniq')}</span>`+
    `<span class="stat"><b>${(d.by_source.modello||0)}</b> ${tt('st_model')}</span>`+
    `<span class="stat"><b>${(d.by_source.regex||0)}</b> ${tt('st_regex')}</span>`+
    `<span class="stat"><b>${T[L].chars(d.n_chars)}</b> ${tt('st_chars')}</span>`+
    ((d.excluded_tags&&d.excluded_tags.length)
      ? `<span class="stat" title="${d.excluded_tags.join(', ')}"><b>${d.excluded_tags.length}</b> ${tt('st_excl')}</span>` : '')+
    (d.ocr?`<span class="stat" title="${d.ocr.engine||''}">🔍 ${T[L].ocr_badge(
        d.ocr.engine==='tesseract'?tt('ocr_engine_t'):tt('ocr_engine_o'),
        d.ocr.pages,d.ocr.seconds)}</span>`:'');
  // legenda cliccabile (toggle highlight)
  const lg=$('legend');lg.innerHTML='';
  for(const [k,v] of Object.entries(d.by_label)){
    const c=colors(k);const el=document.createElement('span');
    el.className='chip'+(off.has(k)?' off':'');
    el.innerHTML=`<span class="sw" style="background:${c.bd}"></span>${k}<span class="n">${v}</span>`;
    el.onclick=()=>{off.has(k)?off.delete(k):off.add(k);render();};
    lg.appendChild(el);
  }
  // dizionario (assente quando lo switch e' su DISATTIVO)
  const hasMap=d.mapping_enabled!==false;
  const rows=$('maprows');rows.innerHTML='';
  const keys=hasMap?Object.keys(d.mapping):[];
  $('tablewrap').style.display=keys.length?'':'none';
  $('dictNone').style.display=hasMap?'none':'';
  $('dictHint').innerHTML=hasMap?tt('dict_hint'):tt('dict_off_hint');
  $('dl').style.display=hasMap?'':'none';
  for(const ph of keys){const lab=ph.slice(1,ph.lastIndexOf('_'));const c=colors(lab);
    const tr=document.createElement('tr');
    tr.innerHTML=`<td class="k" style="color:${c.tx}">${ph}</td>`+
      `<td class="v">${escapeHtml(d.mapping[ph])}</td>`+
      `<td><span class="chip" style="cursor:default"><span class="sw" style="background:${c.bd}"></span>${lab}</span></td>`;
    rows.appendChild(tr);}
  $('ulock').textContent=keys.length?T[L].dict_n(keys.length):'';
  if(VIEW==='pdf')setView('pdf');            // stavo guardando il PDF: lo rigenero
}

function escapeHtml(s){return s.replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));}

/* ---- view toggle (dx): anteprima con i tag | testo da copiare | PDF censurato ---- */
async function setView(v){
  if(v==='pdf'&&!DATA&&!$('pdf').files.length&&!$('src').value.trim()){
    toast(tt('t_need_anon'),false);return;}
  VIEW=v;
  $('vPrev').classList.toggle('on',v==='prev');
  $('vText').classList.toggle('on',v==='text');
  $('vPdf').classList.toggle('on',v==='pdf');
  $('viewPrev').style.display=v==='prev'?'':'none';
  $('anon').style.display=v==='text'?'':'none';
  $('pdfOutView').style.display=v==='pdf'?'':'none';
  if(v!=='pdf'){
    matchScroll(v==='prev'?$('viewPrev'):$('anon'),$('src'));  // allinea la vista appena mostrata
    return;
  }
  if(OUT_DOC){renderPages($('pdfOutView'),OUT_DOC);return;}
  busy($('pdfOutView'),tt('t_pdf_render'));
  const d=await buildOutPdf();
  if(VIEW!=='pdf')return;                    // l'utente ha cambiato vista nel frattempo
  if(!d){setView('prev');return;}
  renderPages($('pdfOutView'),d);
  if(d.residual+d.skipped>0)toast(T[L].t_pdf_warn(d.residual,d.skipped),false,9000);
}

/* Genera UNA volta il PDF anonimizzato e lo lascia sul server: la stessa copia
   serve sia l'anteprima a destra sia il download (una sola inferenza).
   Come /pdf, il dizionario non viene inviato: il server lo ricostruisce e lo
   butta, quindi funziona anche con lo switch "dizionario" su DISATTIVO. */
let PDF_JOB=null;
function buildOutPdf(){
  if(OUT_DOC)return Promise.resolve(OUT_DOC);
  if(PDF_JOB)return PDF_JOB;                 // click su tab + download insieme -> una richiesta sola
  const file=$('pdf').files[0];const text=$('src').value.trim();
  if(!DATA&&!file&&!text){toast(tt('t_need_anon'),false);return Promise.resolve(null);}
  const excl=TAGS_LOADED?[...EXCL]:null;
  PDF_JOB=(async()=>{
    try{
      let resp;
      if(file){const fd=new FormData();fd.append('pdf',file);
        if(excl)fd.append('exclude_tags',excl.join(','));
          resp=await fetch('/pdf/preview',{method:'POST',body:fd});}
      else{const body={text};if(excl)body.exclude_tags=excl;
          resp=await fetch('/pdf/preview',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify(body)});}
      const d=await resp.json();
      if(!resp.ok){toast(d.error||tt('t_pdf_err'),false,9000);return null;}
      OUT_DOC=d;
      if(d.ocr_pages&&d.ocr_pages.length)toast(tt('ocr_burnin'),false,7000);
      return d;
    }catch(e){toast(tt('t_pdf_err')+': '+e.message,false);return null;}
    finally{PDF_JOB=null;}
  })();
  return PDF_JOB;
}

/* ---- copy / download ---- */
$('copy').onclick=()=>{if(!DATA){toast(tt('t_need_anon'),false);return;}
  navigator.clipboard.writeText(DATA.anonymized_text).then(()=>toast(tt('t_copied')));};
$('dl').onclick=()=>{if(!DATA||!Object.keys(MAP).length){toast(tt('t_nothing_dl'),false);return;}
  const blob=new Blob([JSON.stringify(MAP,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='dizionario_anonimizzazione.json';a.click();URL.revokeObjectURL(a.href);
  toast(tt('t_dl_ok'));};

/* ---- PDF anonimizzato (issue #7): redazione vera del PDF caricato, oppure PDF
       ricostruito dal testo anonimizzato quando l'input non era un PDF.
       Se lo si e' gia' guardato nella vista "PDF censurato" il file esiste gia'
       sul server: si scarica quello, senza una seconda anonimizzazione. ---- */
$('dlpdf').onclick=async()=>{
  const btn=$('dlpdf');btn.disabled=true;const old=btn.innerHTML;
  btn.innerHTML='<span class="spin"></span> '+tt('t_pdf_making');
  try{
    const d=await buildOutPdf();
    if(!d)return;
    const a=document.createElement('a');
    a.href='/doc/'+d.doc_id+'/file.pdf';
    a.download=d.filename||'documento_anonimizzato.pdf';a.click();
    // residui = valori ancora leggibili nell'output; saltati = valori troppo corti
    // per essere cercati senza devastare il documento. Entrambi restano IN CHIARO.
    if(d.residual+d.skipped>0)toast(T[L].t_pdf_warn(d.residual,d.skipped),false,9000);
    else toast(tt('t_pdf_ok'));
  }finally{btn.disabled=false;btn.innerHTML=old;}
};

/* ---- reverse ---- */
function reverse(){
  const txt=$('rin').value;
  if(!txt.trim()){toast(tt('t_paste_restore'),false);return;}
  const keys=Object.keys(MAP);
  if(!keys.length){toast(tt('t_no_dict'),false);return;}
  // Una sola passata sul testo ORIGINALE, non N replace() sequenziali su un
  // `out` che si accumula: se il valore sostituito per una chiave contenesse
  // per caso il nome di una chiave successiva (es. una ragione sociale con
  // dentro "CF_2"), il replace seguente lo ri-sostituiva -- un falso positivo
  // di secondo ordine che nessuna delle due chiavi, prese da sole, produce.
  // Il lookup avviene ricostruendo il nome dal testo appena matchato, quindi
  // l'ordine delle chiavi nella mappa e' ora irrilevante (non serve piu'
  // ordinare per lunghezza: parentesi TUTTO-O-NIENTE e confine di parola
  // bastano a evitare che FULLNAME_1 matchi dentro FULLNAME_10, si veda sotto).
  const byInner=new Map(keys.map(ph=>[ph.slice(1,-1),MAP[ph]]));
  const alts=keys.map(ph=>{
    const inner=ph.slice(1,-1);                 // FULLNAME_1
    const esc=inner.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
    // tollerante: parentesi o forma nuda, spazi, eventuale grassetto markdown.
    // Gli spazi si consumano SOLO insieme alla parentesi: con '\[?\s*' un placeholder
    // scritto senza parentesi si portava via anche gli spazi intorno, e "Il FULLNAME_1 ha"
    // tornava "IlMario Rossiha". Con un tab o un a-capo spariva la colonna o la riga.
    // E le parentesi sono TUTTO-O-NIENTE, con un confine di parola sulla forma nuda:
    // se ognuna e' opzionale per conto suo, con CF_1 in mappa un indice inventato dal
    // modello ([CF_12]) matcha per meta' e il ripristino scrive un codice fiscale
    // SBAGLIATO - un segnaposto rimasto si vede, un valore sbagliato no.
    // Il confine e' Unicode-aware (specchia _is_word() lato Python, che tratta le
    // lettere accentate come interne alla parola): \b di JS e' ASCII-only, quindi
    // su "CF_1e' gia'" non vedeva un confine e il valore tornava incollato alla
    // lettera accentata -- lo stesso difetto che questa funzione deve evitare,
    // riaperto in modo asimmetrico solo per le parole italiane.
    return '\\[\\s*'+esc+'\\s*\\]|(?<![\\p{L}\\p{N}_])'+esc+'(?![\\p{L}\\p{N}_])';
  }).join('|');
  // Gli asterischi di grassetto markdown si assorbono in coppie SIMMETRICHE
  // (stesso numero prima e dopo, fino a 2, col backreference \1): una sequenza
  // di asterischi condivisa fra due placeholder adiacenti si divide cosi' in
  // modo deterministico, invece di dipendere da quale chiave viene elaborata
  // prima (con N pattern separati, un lato mangiava tutta la sequenza e
  // lasciava l'altro placeholder non risolto o incollato al primo valore).
  const rx=new RegExp('(\\*{0,2})(?:'+alts+')\\1','gu');
  const out=txt.replace(rx,(m)=>{
    const core=m.replace(/^\*+|\*+$/g,'');
    const br=core.match(/^\[\s*([\s\S]*?)\s*\]$/);
    const val=byInner.get(br?br[1]:core);
    return val===undefined?m:val;               // per costruzione sempre trovato
  });
  const o=$('rout');o.textContent=out;o._raw=out;
  toast(tt('t_restored'));
}
$('rev').onclick=reverse;
$('rcopy').onclick=()=>{const o=$('rout');if(!o._raw){toast(tt('t_nothing_copy'),false);return;}
  navigator.clipboard.writeText(o._raw).then(()=>toast(tt('t_restored_copied')));};
$('rclear').onclick=()=>{$('rin').value='';$('rout').innerHTML=routEmpty();$('rout')._raw='';};

/* ---- carica dizionario da file (per sessioni diverse) ---- */
$('dictFile').onchange=e=>{const f=e.target.files[0];if(!f)return;
  const r=new FileReader();r.onload=()=>{try{
    const parsed=JSON.parse(r.result);
    const dictKeys=parsed&&typeof parsed==='object'&&!Array.isArray(parsed)?Object.keys(parsed):[];
    // Un file dizionario corrotto/troncato con una chiave vuota o non testuale
    // degenera il pattern di reverse() in un match quasi universale e corrompe
    // l'intero documento al ripristino, non solo un placeholder.
    const valid=dictKeys.length>0&&dictKeys.every(k=>/^\[.+\]$/.test(k)&&typeof parsed[k]==='string');
    if(!valid){toast(tt('t_json_invalid'),false);return;}
    MAP=parsed;
    $('dictInfo').textContent=T[L].dict_loaded_n(dictKeys.length);
    toast(tt('t_dict_loaded'));}catch{toast(tt('t_json_invalid'),false);}};
  r.readAsText(f);};

/* ---- input helpers ---- */
$('go').onclick=run;
$('clear').onclick=()=>{$('src').value='';$('pdf').value='';$('dropTxt').innerHTML=tt('drop');
  DATA=null;$('prev').style.display='none';$('emptyPrev').style.display='';
  $('anon').value='';$('meta').innerHTML='';$('legend').innerHTML='';
  $('dictCard').style.display='none';$('ulock').textContent='';
  // la card era solo nascosta: senza queste tre righe il dizionario resta in MAP e su
  // disco, e al riavvio ricompare zitto al posto di quello del documento nuovo
  MAP={};localStorage.removeItem('pii_map');$('dictInfo').textContent='';
  SRC_DOC=null;OUT_DOC=null;$('pdfSrcView').innerHTML='';$('pdfOutView').innerHTML='';
  $('srcTabs').style.display='none';$('inHint').innerHTML=tt('in_hint');
  $('inHint').style.display='';setSrcView('text');setView('prev');
  document.querySelector('.app').classList.remove('has-result');};
$('src').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')run();});

/* pdf picker + dropzone */
const drop=$('drop');
$('pdf').onchange=e=>{const f=e.target.files[0];if(f)onFile(f);};
['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('hot');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('hot');}));
const OK_EXT=/\.(pdf|md|markdown|txt|text)$/i;
drop.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];
  if(f&&(f.type==='application/pdf'||OK_EXT.test(f.name))){
    const dt=new DataTransfer();dt.items.add(f);$('pdf').files=dt.files;
    onFile(f);}else toast(tt('t_drag_pdf'),false);});

/* scroll sincronizzato: editor (sx) <-> anteprima e testo (dx) */
linkScroll($('src'),$('viewPrev'));linkScroll($('viewPrev'),$('src'));
linkScroll($('src'),$('anon'));linkScroll($('anon'),$('src'));
/* e le due viste PDF fra loro: stesso documento, prima e dopo la censura */
linkScroll($('pdfSrcView'),$('pdfOutView'));linkScroll($('pdfOutView'),$('pdfSrcView'));

/* lingua: selettore + applicazione iniziale (default IT, preferenza salvata) */
$('lang').onchange=e=>applyLang(e.target.value);
applyLang(localStorage.getItem('pii_lang')||'it');

/* avviso: popup a click (non hover), si chiude cliccando fuori o con Esc */
$('infoBtn').addEventListener('click',e=>{e.stopPropagation();$('infoBtn').classList.toggle('open');});
$('infoBtn').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('infoBtn').classList.toggle('open');}});
document.addEventListener('click',e=>{if(!$('infoBtn').contains(e.target))$('infoBtn').classList.remove('open');});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('infoBtn').classList.remove('open');closeConfig();closeTags();}});

/* ---- config modal ---- */
async function openConfig(){
  const r=await fetch('/config');const d=await r.json();
  $('cfgHost').value=d.host||'127.0.0.1';
  $('cfgPort').value=d.port||5005;
  $('cfgStatus').className='cfg-status';$('cfgStatus').textContent='';
  $('cfgOverlay').classList.add('open');
}
function closeConfig(){$('cfgOverlay').classList.remove('open');}
$('cfgOverlay').addEventListener('click',e=>{if(e.target===$('cfgOverlay'))closeConfig();});
async function checkPort(){
  const h=$('cfgHost').value.trim(),p=parseInt($('cfgPort').value);
  if(!p||p<1024||p>65535){$('cfgStatus').className='cfg-status fail';$('cfgStatus').textContent=tt('cfg_in_use');return;}
  const r=await fetch(`/port-check?host=${encodeURIComponent(h)}&port=${p}`);
  const d=await r.json();
  $('cfgStatus').className=d.available?'cfg-status ok':'cfg-status fail';
  $('cfgStatus').textContent=d.available?tt('cfg_available'):tt('cfg_in_use');
}
async function saveConfig(){
  const h=$('cfgHost').value.trim(),p=parseInt($('cfgPort').value);
  if(!p||p<1024||p>65535){toast(tt('cfg_in_use'),false);return;}
  await fetch('/config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({host:h,port:p})});
  toast(tt('cfg_saved'));
  closeConfig();
}

/* ---- switch "Dizionario reversibile" ---- */
function renderMapping(){
  const on=MAPPING;
  $('mapToggle').classList.toggle('off',!on);
  $('mapToggle').setAttribute('aria-checked',String(on));
  $('mapSw').classList.toggle('off',!on);
  $('mapState').textContent=on?tt('map_on'):tt('map_off');
  $('mapSub').innerHTML=on?tt('map_sub_on'):tt('map_sub_off');
  $('rOff').style.display=on?'none':'';
  $('dl').style.display=on?'':'none';      // niente dizionario -> niente download
}
async function toggleMapping(){
  MAPPING=!MAPPING;
  renderMapping();
  toast(MAPPING?tt('t_map_on'):tt('t_map_off'),MAPPING);
  try{                                 // persiste: vale anche per l'API /analyze
    await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({mapping_enabled:MAPPING})});
  }catch(e){}
}

/* ---- switch "OCR automatico" (PDF scansionati) ---- */
function renderOcr(){
  const on=OCR;
  $('ocrToggle').classList.toggle('off',!on);
  $('ocrToggle').setAttribute('aria-checked',String(on));
  $('ocrSw').classList.toggle('off',!on);
  $('ocrState').textContent=on?tt('ocr_on'):tt('ocr_off');
  $('ocrSub').innerHTML=on?tt('ocr_sub_on'):tt('ocr_sub_off');
  const info=OCR_INFO||{};
  const eng=$('ocrEngine');
  if(!OCR_INFO){eng.textContent='—';$('ocrSw').classList.remove('none');}
  else if(info.available){
    eng.textContent=info.engine==='tesseract'?tt('ocr_engine_t'):tt('ocr_engine_o');
    $('ocrSw').classList.remove('none');
  }else{
    eng.textContent=tt('ocr_none');
    $('ocrSw').classList.add('none');
  }
}
async function toggleOcr(){
  OCR=!OCR;
  renderOcr();
  toast(OCR?tt('t_ocr_on'):tt('t_ocr_off'),OCR);
  try{                                 // persiste: vale anche per l'API
    await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ocr_enabled:OCR})});
  }catch(e){}
}

/* ---- tags modal: legenda + selezione dei tag da anonimizzare ---- */
let TAGS_META={};
async function loadTags(){
  try{
    const d=await (await fetch('/settings')).json();
    TAGS=d.tags||[];EXCL=new Set(d.excluded_tags||[]);TAGS_META=d;TAGS_LOADED=true;
    MAPPING=d.mapping_enabled!==false;renderMapping();
    OCR=d.ocr_enabled!==false;OCR_INFO=d.ocr||null;renderOcr();
  }catch(e){/* server vecchio o offline: si continua con il comportamento di default */}
}
function renderTags(){
  const list=$('tgList');list.innerHTML='';
  for(const t of TAGS){
    const c=colors(t.tag),on=!EXCL.has(t.tag);
    const row=document.createElement('label');
    row.className='tg-row'+(on?'':' off');
    row.innerHTML=`<input type="checkbox" ${on?'checked':''}>`+
      `<span class="sw" style="background:${c.bd}"></span>`+
      `<span class="txt"><span class="nm" style="color:${c.tx}">${t.tag}</span>`+
      `<div class="ds">${escapeHtml(L==='en'?t.en:t.it)}</div>`+
      `<div class="ex">${escapeHtml(t.example||'')}</div></span>`;
    row.querySelector('input').onchange=e=>{
      e.target.checked?EXCL.delete(t.tag):EXCL.add(t.tag);
      row.classList.toggle('off',!e.target.checked);tgCount();};
    list.appendChild(row);
  }
  tgCount();
}
function tgCount(){$('tgCount').textContent=T[L].tg_count(TAGS.length-EXCL.size,TAGS.length);}
function setAllTags(on){
  EXCL=on?new Set():new Set(TAGS.map(t=>t.tag));
  renderTags();
}
async function openTags(){
  if(!TAGS.length)await loadTags();
  if(!TAGS.length){toast(tt('t_error'),false);return;}
  $('tgStatus').className='cfg-status';$('tgStatus').textContent='';
  $('tgNote').innerHTML=(TAGS_META.env_override?'<b>'+tt('tg_env')+'</b><br>':'')+
    T[L].tg_note(TAGS_META.config_path||'');
  renderTags();
  $('tagsOverlay').classList.add('open');
}
function closeTags(){$('tagsOverlay').classList.remove('open');loadTags();}  // ricarica: annulla = scarta
$('tagsOverlay').addEventListener('click',e=>{if(e.target===$('tagsOverlay'))closeTags();});
async function saveTags(){
  try{
    const r=await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({excluded_tags:[...EXCL]})});
    const d=await r.json();
    if(!r.ok)throw new Error(d.error||'');
    EXCL=new Set(d.excluded_tags||[]);
    toast(tt('tg_saved'));
    $('tagsOverlay').classList.remove('open');
  }catch(e){$('tgStatus').className='cfg-status fail';$('tgStatus').textContent=tt('tg_err');}
}
loadTags();

/* recupera dizionario da sessione precedente (dopo applyLang -> testo nella lingua giusta) */
try{const m=localStorage.getItem('pii_map');if(m){MAP=JSON.parse(m);
  if(Object.keys(MAP).length)$('dictInfo').textContent=T[L].dict_session(Object.keys(MAP).length);}}catch{}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Rizzo PII — server locale di anonimizzazione.")
    _p.add_argument("--host", default=None, help="indirizzo su cui ascoltare (default da config/env/127.0.0.1)")
    _p.add_argument("--port", type=int, default=None, help="porta su cui ascoltare (default da config/env/5005)")
    _p.add_argument("--exclude-tags", default=None, metavar="TAG,TAG",
                    help="tag PII da NON anonimizzare, es. AGE,GENDER (default da env/prefs.json)")
    _p.add_argument("--no-mapping", action="store_true",
                    help="anonimizzazione definitiva: non costruire il dizionario di ripristino")
    _p.add_argument("--no-ocr", action="store_true",
                    help="non usare l'OCR sui PDF scansionati (nessun layer testuale)")
    _args = _p.parse_args()

    if _args.exclude_tags is not None:
        EXCLUDED_TAGS = server_config.parse_tag_list(_args.exclude_tags)
        _unknown = [t for t in EXCLUDED_TAGS if t not in TAG_NAMES]
        if _unknown:
            print(f"ERRORE: tag sconosciuti in --exclude-tags: {', '.join(_unknown)}")
            print(f"Tag validi: {', '.join(TAG_NAMES)}")
            sys.exit(2)
    if _args.no_mapping:
        MAPPING_ENABLED = False
    if _args.no_ocr:
        OCR_ENABLED = False
    if EXCLUDED_TAGS:
        print(f"Tag esclusi dall'anonimizzazione: {', '.join(EXCLUDED_TAGS)}")
    print("Dizionario reversibile: " +
          ("ATTIVO (si puo' ripristinare)" if MAPPING_ENABLED
           else "DISATTIVO (anonimizzazione definitiva)"))
    _ocr_status = ocr.engine_status()
    print("OCR sulle scansioni: " +
          ("ATTIVO" if OCR_ENABLED else "DISATTIVO") +
          f" · motore: {_ocr_status['engine'] or 'nessuno'}"
          + (f" ({_ocr_status['detail']})" if _ocr_status["available"] else ""))

    _host, _port = server_config.resolve(cli_host=_args.host, cli_port=_args.port)

    if not server_config.port_available(_host, _port):
        print(f"ERRORE: porta {_port} occupata su {_host}")
        sys.exit(server_config.EXIT_PORT_CONFLICT)

    print(f"Server su http://{_host}:{_port}")
    try:
        app.run(host=_host, port=_port, threaded=True)
    except OSError as e:
        print(f"ERRORE bind: {e}")
        sys.exit(server_config.EXIT_PORT_CONFLICT)
