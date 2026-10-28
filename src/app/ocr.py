# -*- coding: utf-8 -*-
"""
OCR integrato: il fallback per i PDF che NON hanno layer testuale (le scansioni).

Prima di questa modulo l'app sapeva solo *parlare* di OCR: `/pii-zones` restituiva
le coordinate delle zone PII e `/ocr-zones` accettava i risultati di un motore
esterno, ma il motore non c'era. Un PDF scansionato finiva con un 422
("Nessuna occorrenza trovata nel PDF: se il documento e' una scansione ...").

Qui il motore c'e': quando una pagina non ha testo utilizzabile, la pagina viene
renderizzata e passata all'OCR, e il testo riconosciuto entra nell'analisi come
qualsiasi altro testo. Il risultato e' che la scansione diventa anonimizzabile
al pari di un PDF con layer testuale.

MOTORI SUPPORTATI (plug-in, il primo disponibile vince):

  1. "tesseract" - il binario Tesseract OCR. Non e' una dipendenza Python: e' un
     programma di sistema (apt/brew/choco). Qualita' migliore su documenti
     rumorosi, e restituisce le parole con le loro coordinate, quindi la
     redazione puo' coprire esattamente la parola.
  2. "onnx" - RapidOCR (PP-OCR) via onnxruntime: si installa con pip
     (`pip install rapidocr-onnxruntime`), non richiede nessun binario di
     sistema ed e' l'unico che si puo' impacchettare dentro l'eseguibile
     Tauri/PyInstaller. Riconosce la pagina per righe, non per parole: la
     redazione copre la riga (vedi OCR_CHUNK in pdf_export).

Nessuna delle due e' obbligatoria: senza un motore l'app funziona esattamente
come prima, ma sui documenti scansionati dice chiaramente cosa installare.

Il modulo non importa torch/transformers: lavora su bytes PDF e restituisce
strutture pure, quindi e' testabile in isolamento (come detectors/pdf_export).
"""

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time

import fitz  # PyMuPDF

# --------------------------------------------------------------------------- #
# Parametri
# --------------------------------------------------------------------------- #
DEFAULT_LANG = "ita+eng"     # lingue richieste al motore (fallback automatici)
OCR_DPI = 200                # risoluzione del rendering per l'OCR
OCR_MIN_CHARS_PER_PAGE = 20  # sotto questa soglia la pagina e' "senza testo"
OCR_MAX_PAGES = 40           # oltre, l'OCR viene rifiutato (troppo lento)
OCR_TIMEOUT_S = 120          # timeout per pagina


class OcrError(RuntimeError):
    """Errore dell'OCR (motore rotto, timeout, immagine illeggibile...)."""


class OcrUnavailable(OcrError):
    """Nessun motore OCR installato: il chiamante deve dire cosa installare."""


# --------------------------------------------------------------------------- #
# Motori
# --------------------------------------------------------------------------- #
class _Engine:
    """Contratto di un motore: `recognize(png_bytes) -> [blocchi]`.

    Un "blocco" e' una riga riconosciuta: {"text": str, "box": (x0,y0,x1,y1)
    in PIXEL, "conf": float, "words": [{"text","box","conf"}] | None}. Le
    coordinate sono in pixel dell'immagine: chi chiama le converte in punti
    PDF (conosce la pagina).

    "words" e' la parte che conta: se il motore sa dove finisce ogni parola,
    la redazione copre la PII e non l'intera riga. Se e' None (motori che
    riconoscono per riga) le parole vengono stimate dal testo e dalla
    larghezza della riga.
    """

    name = "?"
    detail = ""

    def recognize(self, image_bytes):  # pragma: no cover - contratto
        raise NotImplementedError

    def describe(self):
        return self.detail or self.name


class TesseractEngine(_Engine):
    """Tesseract OCR via CLI (nessuna dipendenza Python, nessun PIL).

    Si usa l'output TSV: ha le coordinate di ogni parola, che servono per
    coprire la parola esatta e non tutta la riga.
    """

    name = "tesseract"

    def __init__(self, exe=None, langs=(DEFAULT_LANG, "eng", "")):
        self.exe = exe or _find_tesseract()
        if not self.exe:
            raise OcrUnavailable("Binario 'tesseract' non trovato nel PATH.")
        self.langs = tuple(langs)
        self.detail = f"Tesseract ({self.exe})"

    def recognize(self, image_bytes):
        last_err = None
        for lang in self.langs:
            try:
                return _tesseract_blocks(self.exe, image_bytes, lang)
            except OcrError as e:
                # Le lingue non installate fanno fallire Tesseract: si riprova
                # con la catena di fallback (ita+eng -> eng -> default).
                last_err = e
        raise last_err or OcrError("Tesseract non ha restituito nulla.")

    def describe(self):
        return self.detail


class OnnxEngine(_Engine):
    """RapidOCR (PP-OCR) su onnxruntime: solo pip, niente binario di sistema."""

    name = "onnx"

    def __init__(self):
        self._engine = None
        self.detail = "RapidOCR/ONNX"

    def _load(self):
        if self._engine is not None:
            return self._engine
        try:                                   # v1.x (pacchetto rapidocr_onnxruntime)
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            return self._engine
        except Exception:
            pass
        try:                                   # v2.x (pacchetto rapidocr)
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
            return self._engine
        except Exception as e:
            raise OcrUnavailable(
                "RapidOCR non installato (pip install rapidocr-onnxruntime)."
            ) from e

    def recognize(self, image_bytes):
        engine = self._load()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.png")
            with open(path, "wb") as fh:
                fh.write(image_bytes)
            try:
                res = engine(path)
            except Exception as e:
                raise OcrError(f"RapidOCR: {e}") from e
        return _rapidocr_blocks(res)

    def describe(self):
        return self.detail


def _find_tesseract():
    """Dove sta il binario tesseract: env, PATH, installazioni standard."""
    env = os.environ.get("PII_TESSERACT")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    if os.name == "nt":
        for base in (r"C:\Program Files\Tesseract-OCR",
                     r"C:\Program Files (x86)\Tesseract-OCR",
                     os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR")):
            cand = os.path.join(base, "tesseract.exe")
            if os.path.isfile(cand):
                return cand
    return None


# --------------------------------------------------------------------------- #
# Parsing dell'output dei motori
# --------------------------------------------------------------------------- #
# Un "blocco" e' una riga riconosciuta dal motore:
#   {"text": str, "box": (x0,y0,x1,y1) in PIXEL, "conf": float,
#    "words": [{"text", "box", "conf"}] | None}
#
# "words" e' None solo quando il motore riconosce per RIGA (RapidOCR): li' le
# parole vengono ricavate dal testo e dalla larghezza del box (stima). Tutti
# gli altri motori (Tesseract) danno la parola con le sue coordinate: e' il
# livello di dettaglio giusto per coprire la PII e NON tutta la riga.
def _union_box(boxes):
    xs0 = min(b[0] for b in boxes)
    ys0 = min(b[1] for b in boxes)
    xs1 = max(b[2] for b in boxes)
    ys1 = max(b[3] for b in boxes)
    return (xs0, ys0, xs1, ys1)


def _tesseract_blocks(exe, image_bytes, lang):
    """Tesseract su un PNG -> blocchi (righe) con le coordinate delle PAROLE."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "page.png")
        with open(path, "wb") as fh:
            fh.write(image_bytes)
        cmd = [exe, path, "stdout"]
        if lang:
            cmd += ["-l", lang]
        cmd += ["--psm", "6", "tsv"]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=OCR_TIMEOUT_S)
        except subprocess.TimeoutExpired as e:
            raise OcrError(f"Tesseract: timeout dopo {OCR_TIMEOUT_S}s.") from e
        except OSError as e:
            raise OcrError(f"Tesseract non eseguibile: {e}") from e
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise OcrError(f"Tesseract ha restituito {proc.returncode}: {err[:300]}")
    return _parse_tsv(proc.stdout.decode("utf-8", "replace"))


def _parse_tsv(tsv):
    """TSV di Tesseract -> blocchi (righe) con le parole al loro interno.

    Le parole (level 5) sono raggruppate per (blocco, paragrafo, riga): la
    riga e' il testo, ma ogni PAROLA mantiene il suo box: e' quello che permette
    di coprire solo la PII invece di tutta la riga."""
    words, header = [], False
    for raw in tsv.splitlines():
        if not header:
            if raw.lower().startswith("level\t"):
                header = True
            continue
        parts = raw.split("\t", 11)
        if len(parts) < 12:
            continue
        text = parts[11].strip()
        if not text:
            continue
        try:
            level = int(parts[0])
            left, top = int(parts[6]), int(parts[7])
            width, height = int(parts[8]), int(parts[9])
            conf = float(parts[10])
        except ValueError:
            continue
        if level != 5:                      # solo le parole hanno un box utile
            continue
        words.append({"text": text, "box": (left, top, left + width, top + height),
                      "conf": conf, "key": (parts[2], parts[3], parts[4])})
    groups = {}
    for w in words:
        groups.setdefault(w["key"], []).append(w)
    blocks = []
    for ws in groups.values():
        txt = " ".join(w["text"] for w in ws)
        blocks.append({
            "text": txt,
            "box": _union_box([w["box"] for w in ws]),
            "conf": min(w["conf"] for w in ws) / 100.0,
            "words": [{"text": w["text"], "box": w["box"], "conf": w["conf"] / 100.0}
                      for w in ws],
        })
    return blocks


def _rapidocr_blocks(res):
    """RapidOCR -> blocchi (righe). Due forme di risultato:
      - v1: lista di [box, text, score] (box = 4 punti)
      - v2: oggetto con .boxes / .txts / .scores
    Le parole NON ci sono: il modello riconosce per riga, e le parole
    vengono stimate dal testo e dalla larghezza della riga.
    """
    if res is None:
        return []
    if isinstance(res, tuple):             # (risultato, elapsed)
        res = res[0]
    if res is None:
        return []
    blocks = []
    # forma v2
    if hasattr(res, "txts") and hasattr(res, "boxes"):
        scores = getattr(res, "scores", None)
        for i, (box, txt) in enumerate(zip(res.boxes, res.txts)):
            sc = float(scores[i]) if scores is not None else 1.0
            t = str(txt).strip()
            if not t:
                continue
            blocks.append({"text": t, "box": _poly_box(box), "conf": sc, "words": None})
        return blocks
    # forma v1
    try:
        items = list(res)
    except TypeError:
        return []
    for it in items:
        try:
            box, txt, sc = it[0], it[1], it[2]
        except (TypeError, IndexError):
            continue
        t = str(txt).strip()
        if not t:
            continue
        blocks.append({"text": t, "box": _poly_box(box), "conf": float(sc), "words": None})
    return blocks


def _poly_box(box):
    """Polilatero -> (x0, y0, x1, y1)."""
    pts = [(float(p[0]), float(p[1])) for p in box]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------- #
# Rilevamento del motore
# --------------------------------------------------------------------------- #
_ENGINE = None
_ENGINE_LOCK = threading.Lock()
_ENGINE_PROBED = False


def detect_engine(prefer=None, refresh=False):
    """Il motore OCR da usare, o None se non c'e' nessun motore installato.

    `prefer` forza un motore ("tesseract" / "onnx"). Senza, si prova Tesseract
    e poi RapidOCR: Tesseract e' piu' preciso, RapidOCR e' l'unico installabile
    come pacchetto Python (quindi l'unico impacchettabile nell'eseguibile).
    """
    global _ENGINE, _ENGINE_PROBED
    with _ENGINE_LOCK:
        if _ENGINE_PROBED and not refresh and prefer is None:
            return _ENGINE
        if prefer == "tesseract":
            try:
                _ENGINE = TesseractEngine()
            except OcrError:
                _ENGINE = None
            _ENGINE_PROBED = True
            return _ENGINE
        if prefer == "onnx":
            try:
                _ENGINE = OnnxEngine()
            except OcrError:
                _ENGINE = None
            _ENGINE_PROBED = True
            return _ENGINE
        if _ENGINE_PROBED and not refresh:
            return _ENGINE
        _ENGINE_PROBED = True
        for factory in (TesseractEngine, OnnxEngine):
            try:
                _ENGINE = factory()
                return _ENGINE
            except OcrError:
                continue
            except Exception:
                continue
        _ENGINE = None
        return None


def engine_status():
    """Stato del motore per /health e /settings: {"available", "engine", "detail"}."""
    eng = detect_engine()
    if eng is None:
        return {
            "available": False,
            "engine": None,
            "detail": ("nessun motore OCR installato: installa Tesseract "
                       "(tesseract-ocr + tesseract-ocr-ita) oppure "
                       "pip install rapidocr-onnxruntime"),
        }
    return {"available": True, "engine": eng.name, "detail": eng.describe()}


def install_hint():
    """Testo con cosa installare quando l'OCR serve ma non e' possibile."""
    return ("Serve l'OCR ma nessun motore e' installato. Installa Tesseract "
            "(Linux: apt install tesseract-ocr tesseract-ocr-ita | macOS: "
            "brew install tesseract tesseract-lang | Windows:UB Mannheim "
            "Tesseract) oppure rapidocr-onnxruntime (pip install "
            "rapidocr-onnxruntime).")


# --------------------------------------------------------------------------- #
# Lato PDF: quali pagine sono scansioni, e il loro OCR
# --------------------------------------------------------------------------- #
def _page_has_ink(page):
    """La pagina ha qualcosa disegnato (immagine o vettori): se non ha niente
    e' una pagina vuota e non serve OCR."""
    try:
        if page.get_images():
            return True
    except Exception:
        pass
    try:
        drawings = page.get_drawings()
        if drawings and len(drawings) > 1:
            return True
    except Exception:
        pass
    return False


def page_texts(doc):
    """Testo del layer testuale, pagina per pagina."""
    out = []
    for page in doc:
        try:
            out.append(page.get_text() or "")
        except Exception:
            out.append("")
    return out


def scanned_pages(doc, min_chars=OCR_MIN_CHARS_PER_PAGE):
    """Indici delle pagine senza testo utilizzabile: sono le scansioni.

    Una pagina e' "scansionata" se ha meno di `min_chars` caratteri di testo E
    ha qualcosa disegnato. Una pagina vuota non e' una scansione: e' vuota, e
    non c'e' niente da riconoscere.
    """
    out = []
    for i, page in enumerate(doc):
        try:
            txt = (page.get_text() or "").strip()
        except Exception:
            txt = ""
        if len(txt) >= min_chars:
            continue
        if _page_has_ink(page):
            out.append(i)
    return out


def _split_words_in_rect(text, rect):
    """Stima le parole di una riga a partire dal testo e dal suo rettangolo.

    Serve per i motori che riconoscono per RIGA (RapidOCR): la riga e' un unico
    box, ma la redazione deve coprire la PII e non tutta la riga. Le parole
    vengono distribuite sulla larghezza del box in proporzione alla larghezza
    REALE dei glifi (helv, corpo = altezza del box): e' un'approssimazione, ma
    molto piu' vicina al vero di "tutta la riga".
    """
    spans = [(m.start(), m.end(), m.group())
             for m in re.finditer(r"\S+", text or "")]
    if not spans:
        return []
    fs = max(rect.height, 1.0)
    widths = []
    for _s, _e, w in spans:
        try:
            wl = fitz.get_text_length(w, fontname="helv", fontsize=fs)
        except Exception:
            wl = 0.0
        widths.append(max(wl, 0.1))
    total = sum(widths)
    if total <= 0:
        total = 1.0
    # il box della riga e' la misura vera della riga: le parole vengono
    # distribuite su tutta la sua larghezza, in proporzione alla larghezza
    # dei glifi. Il risultato e' una stima (il motore non ha dato le parole),
    # ma molto piu' vicino al vero di "tutta la riga".
    scale = rect.width / total if rect.width > 0 else 1.0
    out, x = [], rect.x0
    for (_s, _e, w), wl in zip(spans, widths):
        ww = wl * scale
        out.append({"text": w, "box": (x, rect.y0, x + ww, rect.y1)})
        x += ww
    return out


def ocr_pdf(pdf_bytes, pages=None, dpi=OCR_DPI, lang=DEFAULT_LANG, engine=None):
    """OCR delle pagine indicate (o di tutte) di un PDF.

    Ritorna:
      {
        "engine": "tesseract", "dpi": 200, "lang": "ita+eng", "seconds": 3.4,
        "pages": [{"page": 0, "width": 595, "height": 842, "text": "...",
                   "lines": [{"text": "...", "x0":.., "y0":.., "x1":.., "y1":..,
                              "conf": 0.98, "start": 0, "end": 12}],
                   "words": [{"text": "...", "x0":.., "y0":.., "x1":.., "y1":..,
                              "conf": 0.98, "start": 0, "end": 5}]}],
        "text": "testo di tutte le pagine, separate da \\n"
      }

    "words" e' il livello che serve alla redazione: il box di ogni PAROLA, con
    gli offset del suo testo nella pagina. "lines" resta per compatibilita' (e
    come fallback se un motore non sa dare le parole).

    Le coordinate sono in PUNTI PDF con origine in alto a sinistra (convenzione
    PyMuPDF, la stessa di `pdf_export._page_char_index`): cosi' il rettangolo
    trovato qui si usa direttamente per coprire il testo nella pagina.
    """
    if engine is None:
        engine = detect_engine()
    if engine is None:
        raise OcrUnavailable(install_hint())
    t0 = time.time()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.needs_pass:
            raise OcrError("PDF protetto da password: l'OCR non puo' leggerlo.")
        idx = list(range(doc.page_count)) if pages is None else [p for p in pages
                                                                if 0 <= p < doc.page_count]
        if len(idx) > OCR_MAX_PAGES:
            raise OcrError(
                f"Troppe pagine da riconoscere ({len(idx)}): il limite e' "
                f"{OCR_MAX_PAGES}. Raccogli il documento in un file piu' piccolo "
                f"oppure l'OCR si mette in coda.")
        out_pages = []
        for n in idx:
            page = doc.load_page(n)
            rect = page.rect
            pix = page.get_pixmap(dpi=dpi)
            png = pix.tobytes("png")
            blocks = engine.recognize(png) or []
            scale = 72.0 / dpi          # pixel -> punti PDF
            text_parts, plines, pwords = [], [], []
            cursor = 0
            for blk in blocks:
                txt = (blk.get("text") or "").strip()
                if not txt:
                    continue
                box = blk.get("box") or (0, 0, 0, 0)
                x0, y0, x1, y1 = [v * scale for v in box]
                line = {
                    "text": txt,
                    "x0": round(x0, 2), "y0": round(y0, 2),
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "conf": float(blk.get("conf") or 0.0),
                    "start": cursor, "end": cursor + len(txt),
                }
                plines.append(line)
                # le parole: quelle del motore, se le ha, altrimenti stimate
                raw_words = blk.get("words") or None
                if raw_words:
                    est = [{"text": (w.get("text") or "").strip(),
                            "box": w.get("box") or (0, 0, 0, 0),
                            "conf": float(w.get("conf") or 0.0)}
                           for w in raw_words]
                    est = [w for w in est if w["text"]]
                else:
                    # stima nel sistema dei PIXEL (come i box del motore): la
                    # scala si applica dopo, a tutto insieme
                    est = [{"text": w["text"], "box": w["box"], "conf": 0.0}
                           for w in _split_words_in_rect(txt, fitz.Rect(*box))]
                # gli offset delle parole dentro il testo della pagina
                wcursor = cursor
                for w in est:
                    t = w["text"]
                    pos = txt.find(t, wcursor - cursor)
                    if pos < 0:
                        pos = txt.find(t)
                    if pos < 0:
                        continue
                    wcursor = cursor + pos + len(t)
                    bx = w["box"]
                    pwords.append({
                        "text": t,
                        "x0": round(bx[0] * scale, 2), "y0": round(bx[1] * scale, 2),
                        "x1": round(bx[2] * scale, 2), "y1": round(bx[3] * scale, 2),
                        "conf": float(w.get("conf") or 0.0),
                        "start": cursor + pos, "end": cursor + pos + len(t),
                    })
                text_parts.append(txt)
                cursor += len(txt) + 1     # il "\n" fra una riga e l'altra
            out_pages.append({
                "page": n,
                "width": round(rect.width, 2),
                "height": round(rect.height, 2),
                "text": "\n".join(text_parts),
                "lines": plines,
                "words": pwords,
            })
    return {
        "engine": engine.name,
        "dpi": dpi,
        "lang": lang,
        "seconds": round(time.time() - t0, 2),
        "pages": out_pages,
        "text": "\n".join(p["text"] for p in out_pages),
    }
