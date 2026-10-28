# -*- coding: utf-8 -*-
"""
Generazione del PDF anonimizzato (issue #7, punto 1).

Due modalita', entrambe 100% in locale e senza dipendenze dal modello: il modulo
lavora solo su bytes + dizionario {placeholder -> valore}, quindi e' testabile in
isolamento (nessun import di torch/transformers).

  redact_pdf(pdf_bytes, mapping)
      Redazione VERA del PDF originale: per ogni valore del dizionario cerca le
      occorrenze nelle pagine, RIMUOVE il testo dal content stream
      (page.apply_redactions(), non un rettangolo disegnato sopra) e scrive il
      placeholder al suo posto. Layout conservato.

      Il matching e' CHAR-PRECISO con CONFINI DI PAROLA: la pagina viene
      indicizzata carattere per carattere (get_text("rawdict"), ogni glifo con il
      suo bbox) e i valori sono cercati con regex ancorate (?<!\\w)...(?!\\w).
      Cosi' "DE" NON viene redatto dentro "CORDELLA" e una cifra non cancella i
      numeri di un referto: si redige il token/la sequenza esatta, ovunque ma
      intera. Il pattern tollera la SILLABAZIONE a fine riga ("Fran-\\ncesco") e
      gli spazi flessibili tra i token (anche a cavallo di riga).

  text_to_pdf(text)
      PDF "ricostruito" solo testo, impaginato da zero a partire dal testo gia'
      anonimizzato (input incollato oppure file .md/.txt).

Cosa viene ripulito oltre al testo di pagina (tutti posti dove si nasconde PII e
che apply_redactions() da solo NON tocca):
  - metadati classici + XMP;
  - contenuto delle ANNOTAZIONI (commenti, FreeText, note);
  - valore dei CAMPI MODULO (widget AcroForm);
  - titoli dei SEGNALIBRI (outline/TOC);
  - ALLEGATI incorporati (embedded files), rimossi in blocco.

Rete di sicurezza finale: `report["residual"]` elenca i placeholder il cui valore
e' ANCORA leggibile nell'output (testo di pagina + annotazioni + widget + TOC).
Deve essere vuota; l'UI avvisa se non lo e'.

Limiti noti (= punti 2-4 della issue #7):
  - testo dentro immagini raster (scansioni, loghi, blocchi firma): non esiste nel
    layer testuale, quindi non puo' essere trovato ne' redatto (serve OCR). Se in
    tutto il PDF non si trova NESSUNA occorrenza, il chiamante deve rifiutare:
    meglio un errore che un PDF "anonimizzato" che non lo e';
  - valori troppo corti/ambigui (< 2 caratteri alfanumerici, o 2 sole cifre) NON
    vengono redatti: cercarli ovunque devasterebbe il documento. Finiscono in
    `report["skipped"]` e l'UI DEVE avvisare, perche' restano in chiaro.
"""

import re
import unicodedata

import fitz  # PyMuPDF
import pdf_text


class PdfError(ValueError):
    """Errore d'uso (PDF non valido, protetto, dizionario vuoto...)."""


# --------------------------------------------------------------------------- #
# Utility comuni
# --------------------------------------------------------------------------- #
# caratteri tipografici frequenti nei PDF estratti -> equivalenti Latin-1
# (il font base "helv" copre Latin-1: cosi' la sostituzione e' deterministica)
_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", "€": "EUR",
    " ": " ", "•": "-", "ﬁ": "fi", "ﬂ": "fl",
})


def _norm(s):
    """Spazi compressi + casefold: stessa normalizzazione usata in app.py."""
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()


def _fit_fontsize(text, rect, max_fs=10.0, min_fs=4.0):
    """Corpo del testo per far stare `text` dentro `rect` (0 = non ci sta:
    meglio nessuna etichetta che un'etichetta illeggibile o troncata)."""
    try:
        w10 = fitz.get_text_length(text, fontname="helv", fontsize=10.0)
    except Exception:
        return 0
    if w10 <= 0:
        return 0
    fs = min(max_fs, rect.height * 0.82, 10.0 * max(rect.width - 2.0, 0.0) / w10)
    return round(fs, 1) if fs >= min_fs else 0


def _covered(rect, taken, thr=0.85):
    """True se `rect` e' gia' (quasi) tutto dentro una redazione precedente:
    evita doppioni quando un valore e' contenuto in un altro (es. "Rossi"
    dentro "Mario Rossi", redatto prima perche' piu' lungo)."""
    area = rect.get_area()
    if area <= 0:
        return True
    for t in taken:
        inter = fitz.Rect(rect)
        inter.intersect(t)
        if not inter.is_empty and inter.get_area() / area >= thr:
            return True
    return False


# --------------------------------------------------------------------------- #
# Indice char-preciso della pagina + ricerca con confini di parola
# --------------------------------------------------------------------------- #
def _page_char_index(page):
    """(testo, [bbox per carattere]) dalla pagina: ogni carattere del layer
    testuale con il suo rettangolo (None per i newline di fine riga)."""
    raw = page.get_text("rawdict")
    chars, boxes = [], []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:          # solo blocchi di testo
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    c = ch.get("c") or ""
                    for cc in c:            # ligature -> piu' caratteri, stesso bbox
                        chars.append(cc)
                        boxes.append(fitz.Rect(ch["bbox"]))
            chars.append("\n")
            boxes.append(None)
    return "".join(chars), boxes


# Sillabazione a fine riga: "Fran-\ncesco", "Fran­\ncesco" o anche solo un
# a-capo dentro la parola. Ammesso TRA due caratteri qualsiasi del valore, ma solo
# se c'e' davvero un newline: senza, il gruppo non consuma nulla e il matching
# resta char-per-char (niente tolleranza allo spacing, che aprirebbe a
# sovra-redazioni tipo "MI" dentro "M I L A N O").
_HYPHEN_BREAK = r"(?:[­‐-]?[ \t]*\n[ \t]*)?"


# Sotto questa soglia di caratteri alfanumerici un frammento NON puo' rinunciare
# ai confini di parola: ".it" senza ancore matcherebbe "it" dentro ogni parola.
_UNANCHORED_MIN_ALNUM = 4


def _value_pattern(value):
    """Regex del valore: caratteri esatti con CONFINI DI PAROLA agli estremi,
    sillabazione a fine riga tollerata, whitespace flessibile tra i token.
    Niente match di sottostringhe dentro altre parole.

    Eccezione: quando il valore ha punteggiatura ai bordi il modello ha tagliato
    a meta' una parola spezzata a fine riga (tipico: "Fran-\\ncesco Cordella" ->
    il modello etichetta "-\\ncesco Cordella"). Li' il confine di parola su quel
    lato e' proprio cio' che impedisce di trovare il valore, che resterebbe in
    chiaro: si toglie la punteggiatura e con essa l'ancora, ma solo se quel che
    resta e' abbastanza lungo da non trasformarsi in una sottostringa qualunque."""
    v = _norm(value)
    core = re.sub(r"^[^\w]+", "", v)
    core = re.sub(r"[^\w]+$", "", core)
    toks = [t for t in core.split(" ") if t]
    if not toks:
        return None
    if len(re.sub(r"[\W_]+", "", core)) >= _UNANCHORED_MIN_ALNUM:
        left = "" if re.match(r"^[^\w]", v) else r"(?<!\w)"
        right = "" if re.search(r"[^\w]$", v) else r"(?!\w)"
    else:                                  # troppo corto: ancore obbligatorie
        toks = [t for t in v.split(" ") if t]
        left, right = r"(?<!\w)", r"(?!\w)"
    body = r"\s*".join(_HYPHEN_BREAK.join(re.escape(c) for c in tok) for tok in toks)
    return re.compile(left + body + right, re.IGNORECASE)


def _match_rects(boxes, m):
    """Bbox dei caratteri del match, uniti per riga (overlap verticale)."""
    rects, cur = [], None
    for i in range(m.start(), m.end()):
        b = boxes[i]
        if b is None or b.is_empty:
            continue
        if cur is None:
            cur = fitz.Rect(b)
        elif b.y0 < cur.y1 and b.y1 > cur.y0:      # stessa riga
            cur |= b
        else:                                       # riga nuova
            rects.append(cur)
            cur = fitz.Rect(b)
    if cur is not None and not cur.is_empty:
        rects.append(cur)
    return rects


def _too_noisy(value):
    """Valori non localizzabili in modo sicuro in un PDF: frammenti con meno di 2
    caratteri alfanumerici, o di 2 sole cifre (es. "1", "C", "05", prodotti a
    volte dal modello su testi tabellari). Cercarli ovunque cancellerebbe pezzi di
    documento non-PII: si saltano, MA il chiamante deve avvisare l'utente (restano
    in chiaro)."""
    alnum = re.sub(r"[\W_]+", "", _norm(value))
    return len(alnum) < 2 or (len(alnum) == 2 and alnum.isdigit())


# --------------------------------------------------------------------------- #
# Pulizia di tutto cio' che apply_redactions() non tocca
# --------------------------------------------------------------------------- #
def _scrub_metadata(doc):
    """Azzera i metadati classici e l'XMP: possono contenere PII (autore...)."""
    try:
        # set_metadata aggiorna SOLO le chiavi passate: vanno azzerate tutte
        # esplicitamente, altrimenti autore/titolo originali restano nel file
        doc.set_metadata({
            "title": "", "author": "", "subject": "", "keywords": "",
            "creationDate": "", "modDate": "", "trapped": "",
            "creator": "rizzo-pii", "producer": "rizzo-pii",
        })
    except Exception:
        pass
    f = getattr(doc, "del_xml_metadata", None) or getattr(doc, "delXmlMetadata", None)
    if f:
        try:
            f()
        except Exception:
            pass


def _sub_all(patterns, s):
    """Applica tutte le (regex, placeholder) alla stringa. Ritorna (nuova, n)."""
    n = 0
    for pat, ph in patterns:
        s, k = pat.subn(ph, s)
        n += k
    return s, n


def _scrub_annots(page, patterns):
    """Contenuto e titolo delle annotazioni (commenti, FreeText, note adesive):
    testo vero e proprio, invisibile a get_text() e non toccato dalle redazioni."""
    done = 0
    try:
        annots = list(page.annots())
    except Exception:
        return 0
    for a in annots:
        try:
            if a.type[0] == fitz.PDF_ANNOT_REDACT:      # le nostre, gia' gestite
                continue
            info = a.info
            new = dict(info)
            hit = 0
            for k in ("content", "subject", "title"):
                if info.get(k):
                    new[k], k_hit = _sub_all(patterns, info[k])
                    hit += k_hit
            if hit:
                a.set_info(new)
                a.update()
                done += hit
        except Exception:
            continue
    return done


def _scrub_widgets(page, patterns):
    """Valore dei campi modulo AcroForm: sopravvive a qualsiasi redazione del
    content stream ed e' leggibile aprendo il PDF."""
    done = 0
    try:
        widgets = list(page.widgets())
    except Exception:
        return 0
    for w in widgets:
        try:
            val = w.field_value
            if not isinstance(val, str) or not val:
                continue
            new, hit = _sub_all(patterns, val)
            if hit:
                w.field_value = new
                w.update()
                done += hit
        except Exception:
            continue
    return done


def _scrub_toc(doc, patterns):
    """Titoli dei segnalibri: spesso ricalcano intestazioni con nomi e numeri."""
    try:
        toc = doc.get_toc(simple=True)
    except Exception:
        return 0
    if not toc:
        return 0
    done, new_toc = 0, []
    for entry in toc:
        lvl, title, pg = entry[0], entry[1], entry[2]
        title, hit = _sub_all(patterns, title or "")
        done += hit
        new_toc.append([lvl, title, pg])
    if done:
        try:
            doc.set_toc(new_toc)
        except Exception:
            return 0
    return done


def _strip_embedded(doc):
    """Allegati incorporati: non ispezionabili in modo affidabile (possono essere
    di qualunque formato), quindi si rimuovono tutti."""
    removed = 0
    try:
        names = list(doc.embfile_names())
    except Exception:
        return 0
    for name in names:
        try:
            doc.embfile_del(name)
            removed += 1
        except Exception:
            continue
    return removed


def _readable_text(doc):
    """TUTTO il testo leggibile del documento: pagine + annotazioni + campi
    modulo + segnalibri. E' la base della verifica dei residui: se un valore
    compare qui, l'anonimizzazione NON e' completa.
    NB: non vede il testo dentro immagini raster (loghi, firme, scansioni)."""
    return pdf_text.collect_readable_text(doc, redact_annot_type=fitz.PDF_ANNOT_REDACT)


def _verify_residuals(pdf_bytes, items):
    """Placeholder il cui valore e' ANCORA leggibile nell'output. Usa lo STESSO
    pattern della redazione (sillabazione inclusa), altrimenti dichiarerebbe
    "0 residui" proprio nei casi che il matcher non sa gestire."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = _readable_text(doc)
    residual = []
    for ph, val in items:
        pat = _value_pattern(val)
        if pat and pat.search(text):
            residual.append(ph)
    return residual


# --------------------------------------------------------------------------- #
# API pubblica
# --------------------------------------------------------------------------- #
# Colore della casella di redazione: il viola del brand (#7c3a9e) pieno, con il
# placeholder in bianco sopra. Deve SALTARE ALL'OCCHIO scorrendo il documento —
# una casella tenue si confonde con un'evidenziazione qualsiasi, e chi rilegge il
# PDF prima di mandarlo fuori deve vedere subito cosa e' stato coperto.
REDACT_FILL = (0.486, 0.227, 0.620)
REDACT_TEXT = (1.0, 1.0, 1.0)


def _prepare_items(mapping):
    """Preparazione comune a redact_pdf e redact_pdf_ocr.

    Ritorna (items, skipped, usable, by_ph, patterns):
      items    -> [(placeholder, valore)] come sono nel dizionario
      skipped  -> placeholder troppo corti/ambigui (NON cercati: restano in chiaro)
      usable   -> [(placeholder, valore, pattern)] cercabili nel documento
      by_ph    -> {placeholder: n_occorrenze}
      patterns -> [(pattern, placeholder)] per annotazioni/widget/TOC
    """
    if not isinstance(mapping, dict) or not mapping:
        raise PdfError("Dizionario vuoto: anonimizza prima il documento.")
    items = [(ph, v) for ph, v in mapping.items()
             if isinstance(ph, str) and isinstance(v, str) and v.strip()]
    if not items:
        raise PdfError("Dizionario non valido.")

    # valori lunghi per primi: cosi' "Rossi" non spezza la redazione di
    # "Mario Rossi" (i rect gia' coperti vengono saltati da _covered)
    items.sort(key=lambda kv: -len(kv[1]))

    skipped, usable = [], []
    for ph, val in items:
        if _too_noisy(val):
            skipped.append(ph)
            continue
        pat = _value_pattern(val)
        if pat:
            usable.append((ph, val, pat))
        else:
            skipped.append(ph)

    by_ph = {ph: 0 for ph, _, _ in usable}
    patterns = [(pat, ph) for ph, _, pat in usable]   # per annot/widget/TOC
    return items, skipped, usable, by_ph, patterns


def redact_pdf(pdf_bytes, mapping, fill=REDACT_FILL, text_color=REDACT_TEXT,
               exclude=None):
    """PDF originale -> PDF con redazione vera + placeholder al posto delle PII.

    mapping: {"[FULLNAME_1]": "Mario Rossi", ...} (il dizionario di analyze()).
    `exclude`: insieme di id di occorrenze da NON redigere (l'utente ha
    annullato la copertura dall'anteprima: vedi /doc/<id>/redactions). Le
    occorrenze escluse restano in chiaro e finiscono in `report["residual"]`.

    Ritorna (bytes, report) con report = {
        "occurrences":    occorrenze redatte nel testo di pagina,
        "found":          occorrenze trovate (comprese quelle annullate),
        "by_placeholder": {placeholder: n_occorrenze},
        "not_found":      placeholder cercati ma senza occorrenze di pagina,
        "skipped":        placeholder NON cercati perche' troppo corti/ambigui
                          (restano in chiaro: va segnalato all'utente),
        "residual":       placeholder ancora leggibili nell'output (deve essere []),
        "annots":         sostituzioni nelle annotazioni,
        "widgets":        sostituzioni nei campi modulo,
        "toc":            sostituzioni nei segnalibri,
        "embedded":       allegati rimossi,
        "hits":           le occorrenze con la loro geometria (per l'anteprima
                          interattiva: un click annulla la copertura),
        "uncovered":      placeholder rimasti in chiaro perche' l'utente ha
                          annullato la copertura (nel burn-in i pixel non sono
                          verificabili dal testo: qui sono le uniche prove),
    }
    """
    items, skipped, usable, by_ph, patterns = _prepare_items(mapping)
    exclude = set(exclude or ())

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise PdfError("File PDF non valido o danneggiato.")
    if doc.needs_pass:
        doc.close()
        raise PdfError("PDF protetto da password: rimuovi la protezione e riprova.")

    total = n_annots = n_widgets = 0
    hits = []
    next_id = 0

    # 1) la geometria delle occorrenze (dipende SOLO dal documento e dal
    #    dizionario: annullare una copertura non sposta le altre)
    for pno, page in enumerate(doc):
        next_id = _page_hits_vector(page, usable, by_ph, hits, next_id, pno)
    # 2) la copertura vera (saltando le occorrenze annullate)
    for pno, page in enumerate(doc):
        total += _apply_hits_vector(page, hits, pno, exclude, fill, text_color)
        # dopo le redazioni: annotazioni e campi modulo non sono content stream
        n_annots += _scrub_annots(page, patterns)
        n_widgets += _scrub_widgets(page, patterns)

    n_toc = _scrub_toc(doc, patterns)
    n_emb = _strip_embedded(doc)
    _scrub_metadata(doc)
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()

    checked = [(ph, v) for ph, v in items if ph in by_ph]
    return out, {
        "occurrences": total,
        "found": len(hits),
        "by_placeholder": by_ph,
        "not_found": [ph for ph, n in by_ph.items() if n == 0],
        "skipped": skipped,
        "residual": _verify_residuals(out, checked),
        "annots": n_annots,
        "widgets": n_widgets,
        "toc": n_toc,
        "embedded": n_emb,
        "hits": hits,
        "excluded": sorted(i for i in exclude if any(h["id"] == i for h in hits)),
        "uncovered": sorted({h["placeholder"] for h in hits if h["id"] in exclude}),
    }


def _page_hits_vector(page, usable, by_ph, hits, next_id, pno):
    """Le occorrenze da coprire nel layer testuale di una pagina.

    Non disegna niente: calcola solo la geometria (rettangoli + testo del
    placeholder). Ritorna il prossimo id libero.
    """
    text, boxes = _page_char_index(page)
    if not text.strip():
        return next_id
    taken = []
    for ph, val, pat in usable:
        for m in pat.finditer(text):
            rects = _match_rects(boxes, m)
            placed, labeled = [], False
            for r in rects:
                if _covered(r, taken):
                    continue
                fs = 0 if labeled else _fit_fontsize(ph, r)
                placed.append({"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1,
                               "fs": fs, "text": ph if fs else None})
                taken.append(fitz.Rect(r))
                labeled = labeled or bool(fs)
            if placed:
                hits.append({"id": next_id, "page": pno, "placeholder": ph,
                             "kind": "vector", "rects": placed,
                             "w": round(page.rect.width, 2),
                             "h": round(page.rect.height, 2)})
                by_ph[ph] += 1
                next_id += 1
    return next_id


def _apply_hits_vector(page, hits, pno, exclude, fill, text_color):
    """Disegna (e applaca) le redazioni vettoriali di una pagina.

    Stessa logica di sempre: il matching e' char-preciso e con confini di
    parola, e il testo viene rimosso dal content stream con apply_redactions().
    Le occorrenze con id in `exclude` vengono saltate: restano in chiaro."""
    n = 0
    for h in hits:
        if h["page"] != pno or h["id"] in exclude:
            continue
        for r in h["rects"]:
            rect = fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
            _add_redact_annot(page, rect, r.get("text"), r.get("fs") or 6,
                              fill, text_color)
            n += 1
    if n:
        page.apply_redactions()   # rimozione VERA dal content stream
    return n


# --------------------------------------------------------------------------- #
# Redazione delle pagine SCANSIONATE (burn-in)
#
# In una scansione la PII e' nel pixel, non nel content stream: nessuna
# redazione vettoriale la puo' toccare. Qui la pagina viene RASTERIZZATA e i
# rettangoli delle PII vengono anneriti dentro i pixel stessi (non disegnati
# sopra: sopra i pixel originali resterebbero nel file e la PII sarebbe
# recuperabile con un pizzico di zucchero). L'originale sparisce, il risultato
# e' un PDF di immagini con i placeholder scritti sopra.
# --------------------------------------------------------------------------- #
def _ocr_rects(words, start, end):
    """Rettangoli (in punti PDF) che coprono l'intervallo [start, end) del testo OCR.

    Si lavora sulle PAROLE, non sulle righe: ogni parola ha il suo box, quindi
    la casella copre esattamente la PII ("Mario Rossi") e non tutta la riga
    ("Mario Rossi nato a Milano il 3 giu")."""
    rects, cur = [], None
    for w in words:
        s, e = w.get("start", 0), w.get("end", 0)
        if e <= start or s >= end:
            continue
        r = fitz.Rect(w.get("x0", 0), w.get("y0", 0),
                      w.get("x1", 0), w.get("y1", 0))
        if r.is_empty:
            continue
        if cur is None:
            cur = fitz.Rect(r)
        elif r.y0 < cur.y1 and r.y1 > cur.y0:      # stessa riga: si fondono
            cur |= r
        else:                                       # riga diversa: casella nuova
            rects.append(cur)
            cur = fitz.Rect(r)
    if cur is not None and not cur.is_empty:
        rects.append(cur)
    return rects


def _page_words(ocr_page):
    """Le parole della pagina OCR; se il motore non le ha date si cade sulle
    righe (meno preciso, ma meglio di niente)."""
    words = ocr_page.get("words") or []
    if words:
        return words, False
    return ocr_page.get("lines") or [], True


def _ocr_page_hits(ocr_page, usable, by_ph, hits, next_id, pno, w, h):
    """Le occorrenze da coprire in una pagina OCR (geometria, senza disegnare).

    Il testo della pagina e' lo stesso che e' stato analizzato (e' l'OCR), quindi
    il matching dei valori e' identico a quello del layer testuale: stessi
    pattern, stessi confini di parola. Le caselle pero' sono quelle delle
    PAROLE coperte, non della riga intera.
    """
    text = ocr_page.get("text") or ""
    words, _coarse = _page_words(ocr_page)
    if not text.strip():
        return next_id
    taken = []
    for ph, val, pat in usable:
        for m in pat.finditer(text):
            rects = _ocr_rects(words, m.start(), m.end())
            placed = []
            for r in rects:
                if _covered(r, taken):
                    continue
                taken.append(fitz.Rect(r))
                placed.append({"x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1})
            if placed:
                hits.append({"id": next_id, "page": pno, "placeholder": ph,
                             "kind": "burnin", "rects": placed,
                             "w": round(w, 2), "h": round(h, 2)})
                by_ph[ph] += 1
                next_id += 1
    return next_id


def _raster_page(page, dpi):
    """Pagina rasterizzata: [pixmap, rettangoli delle PII] -> la pagina intera
    viene resa immagine, quindi i pixel originali non restano nel file."""
    pix = page.get_pixmap(dpi=dpi)
    if pix.colorspace is None or pix.n - pix.alpha not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    return pix


def redact_pdf_ocr(pdf_bytes, mapping, ocr, dpi=200, fill=REDACT_FILL,
                   text_color=REDACT_TEXT, exclude=None):
    """PDF con pagine scansionate -> PDF anonimizzato (burn-in sulle scansioni).

    `ocr` e' il risultato di `ocr.ocr_pdf`: {"pages": [{"page": n, "text": ...,
    "words": [{"text", "x0", "y0", "x1", "y1", "start", "end"}]}]}.

    `exclude`: id di occorrenze da NON coprire (annullate dall'anteprima): i
    pixel originali di quei punti restano nel file, quindi la PII torna
    leggibile: finisce in `report["residual"]` e l'app avvisa.

    Le pagine con OCR vengono rasterizzate e annerite; le altre (layer testuale)
    vengono redatte come sempre, in vettore. Il risultato e' un PDF misto.

    ATTENZIONE (perdita di qualita' deliberata): la pagina rasterizzata non ha
    piu' il testo selezionabile. E' il prezzo della privacy: i pixel originali
    della scansione non restano nel file.
    """
    items, skipped, usable, by_ph, patterns = _prepare_items(mapping)
    exclude = set(exclude or ())
    ocr_pages = {p.get("page"): p for p in (ocr or {}).get("pages", [])}

    try:
        src = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise PdfError("File PDF non valido o danneggiato.")
    if src.needs_pass:
        src.close()
        raise PdfError("PDF protetto da password: rimuovi la protezione e riprova.")

    out_doc = fitz.open()
    total = 0
    hits = []
    next_id = 0
    try:
        for i in range(src.page_count):
            page = src.load_page(i)
            op = ocr_pages.get(i)
            if op is None:
                # pagina con layer testuale: redaction vettoriale come sempre
                try:
                    out_doc.insert_pdf(src, from_page=i, to_page=i)
                except Exception:
                    out_doc.insert_pdf(src, from_page=i, to_page=i,
                                       annots=False, links=False)
                p2 = out_doc.load_page(out_doc.page_count - 1)
                next_id = _page_hits_vector(p2, usable, by_ph, hits, next_id, i)
                total += _apply_hits_vector(p2, hits, i, exclude, fill, text_color)
                continue

            # pagina scansionata: raster + caselle nere dentro i pixel
            next_id = _ocr_page_hits(op, usable, by_ph, hits, next_id, i,
                                     page.rect.width, page.rect.height)
            tmp = fitz.open()
            try:
                tp = tmp.new_page(width=page.rect.width, height=page.rect.height)
                pix = _raster_page(page, dpi)
                tp.insert_image(tp.rect, pixmap=pix)
                for h in hits:
                    if h["page"] != i or h["id"] in exclude:
                        continue
                    for r in h["rects"]:
                        rect = fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
                        black = fitz.Rect(rect.x0 - 1, rect.y0 - 0.5,
                                          rect.x1 + 1, rect.y1 + 0.5)
                        tp.draw_rect(black, color=None, fill=fill)
                # rasterizzo ANCHE le caselle: i pixel originali non restano nel file
                pix2 = tp.get_pixmap(dpi=dpi)
            finally:
                tmp.close()

            np_ = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            np_.insert_image(np_.rect, pixmap=pix2)
            for h in hits:
                if h["page"] != i or h["id"] in exclude:
                    continue
                for r in h["rects"]:
                    rect = fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
                    fs = _fit_fontsize(h["placeholder"], rect)
                    if fs:
                        _insert_placeholder(np_, rect, h["placeholder"], fs, text_color)
            total += sum(1 for h in hits if h["page"] == i and h["id"] not in exclude)
    finally:
        src.close()

    n_annots = n_widgets = 0
    for page in out_doc:
        n_annots += _scrub_annots(page, patterns)
        n_widgets += _scrub_widgets(page, patterns)
    n_toc = _scrub_toc(out_doc, patterns)
    n_emb = _strip_embedded(out_doc)
    _scrub_metadata(out_doc)
    out = out_doc.tobytes(garbage=3, deflate=True)
    out_doc.close()

    checked = [(ph, v) for ph, v in items if ph in by_ph]
    return out, {
        "occurrences": total,
        "found": len(hits),
        "by_placeholder": by_ph,
        "not_found": [ph for ph, n in by_ph.items() if n == 0],
        "skipped": skipped,
        "residual": _verify_residuals(out, checked),
        "annots": n_annots,
        "widgets": n_widgets,
        "toc": n_toc,
        "embedded": n_emb,
        "ocr_pages": sorted(ocr_pages),
        "hits": hits,
        "excluded": sorted(i for i in exclude if any(h["id"] == i for h in hits)),
        "uncovered": sorted({h["placeholder"] for h in hits if h["id"] in exclude}),
    }


def _insert_placeholder(page, rect, ph, fontsize, color):
    """Scrive il placeholder dentro il rettangolo, centrato.

    Non si usa insert_textbox: se il testo non entra nel rettangolo (la
    casella OCR e' alta quanto la riga riconosciuta) insert_textbox NON
    scrive niente e restituisce un codice negativo. Qui la larghezza e'
    calcolata a mano e la baseline centrata: il placeholder c'e' sempre, e
    il testo resta selezionabile nella pagina rasterizzata.
    """
    try:
        w = fitz.get_text_length(ph, fontname="helv", fontsize=fontsize)
    except Exception:
        w = 0
    x = rect.x0 + max(0.0, (rect.width - w) / 2.0)
    y = rect.y0 + (rect.height + fontsize * 0.72) / 2.0
    try:
        page.insert_text((x, y), ph, fontname="helv", fontsize=fontsize, color=color)
    except Exception:
        pass


def _add_redact_annot(page, rect, text, fontsize, fill, text_color):
    """add_redact_annot con cross_out disattivato dove il binding lo supporta
    (le X diagonali sporcano il placeholder); fallback per le versioni vecchie."""
    kw = dict(text=text, fontname="helv", fontsize=fontsize,
              align=fitz.TEXT_ALIGN_CENTER, fill=fill, text_color=text_color)
    try:
        return page.add_redact_annot(rect, cross_out=False, **kw)
    except TypeError:
        return page.add_redact_annot(rect, **kw)


def text_to_pdf(text, margin=56.0, fontsize=10.5, leading=15.5):
    """Testo (gia' anonimizzato) -> PDF A4 solo testo, impaginato da zero.
    Nessun contenuto del documento originale finisce nell'output."""
    text = unicodedata.normalize("NFKC", text or "").translate(_TRANSLATE)
    # helv copre Latin-1: sostituzione esplicita dei caratteri fuori codifica
    text = text.encode("latin-1", "replace").decode("latin-1")
    if not text.strip():
        raise PdfError("Nessun testo da impaginare.")

    a4 = fitz.paper_rect("a4")
    width = a4.width - 2 * margin

    def tl(s):
        return fitz.get_text_length(s, fontname="helv", fontsize=fontsize)

    # In una riga larga `width` non entrano piu' di `cap` caratteri, qualunque siano:
    # oltre quel punto non c'e' niente da misurare. Serve perche' hard_split misurava
    # e ricopiava TUTTA la stringa residua a ogni giro -> O(n^2): una "parola" senza
    # spazi (JSON incollato, base64, riga di tabella coi soli tab) da 16.000 caratteri
    # costava 15 s, da 128.000 un quarto d'ora.
    # Il conto va fatto sul glifo PIU' STRETTO, non su "l": il testo e' gia' forzato in
    # latin-1 poche righe sopra, quindi il repertorio e' chiuso e il minimo si misura.
    # In helv e' l'apostrofo (2,0055 pt a corpo 10,5) contro i 2,331 di "l": con "l" cap
    # varrebbe 209 dove in colonna ne entrano 240, la finestra ci starebbe, il ciclo
    # uscirebbe subito e la coda finirebbe tutta su una riga larga oltre il bordo pagina.
    stretto = min(tl(chr(c)) for c in list(range(32, 127)) + list(range(160, 256)))
    cap = max(2, int(width / max(stretto, 0.1)) + 2)   # >= 2: con margini assurdi width <= 0

    def hard_split(line):
        """Spezza le 'parole' piu' larghe della riga (IBAN, URL...)."""
        if tl(line) <= width:          # caso normale: la riga ci sta gia', una sola misura
            return [line]
        out, i = [], 0
        while len(line) - i > 1:
            head = line[i:i + cap]
            if tl(head) <= width:
                break
            k = max(1, int(len(head) * width / tl(head)))
            while k > 1 and tl(head[:k]) > width:
                k -= 1
            while k < len(head) and tl(head[:k + 1]) <= width:
                k += 1
            out.append(head[:k])
            i += k
        out.append(line[i:])
        return out

    def wrap(par):
        if not par:
            return [""]
        lines, cur = [], ""
        for w in par.split(" "):
            cand = (cur + " " + w) if cur else w
            if not cur or tl(cand) <= width:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        out = []
        for ln in lines:
            out.extend(hard_split(ln))
        return out

    doc = fitz.open()
    page = doc.new_page(width=a4.width, height=a4.height)
    y = margin + fontsize
    for par in text.split("\n"):
        for ln in wrap(par.rstrip()):
            if y > a4.height - margin:
                page = doc.new_page(width=a4.width, height=a4.height)
                y = margin + fontsize
            if ln:
                page.insert_text((margin, y), ln, fontname="helv",
                                 fontsize=fontsize)
            y += leading

    _scrub_metadata(doc)
    out = doc.tobytes(deflate=True)
    doc.close()
    return out
