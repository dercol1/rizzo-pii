# -*- coding: utf-8 -*-
"""OCR integrato: un PDF scansionato (immagine, nessun layer testuale).

Prima dell'OCR integrato un PDF scansionato finiva con un 422: la redazione
del layer testuale non puo' agire su un testo che sta dentro un'immagine.

Qui si verifica la catena completa:
  1. `ocr.scanned_pages` riconosce la pagina come scansione (e NON riconosce
     una pagina con testo);
  2. `ocr.ocr_pdf` trasforma la pagina in testo + box (in punti PDF);
  3. `pdf_export.redact_pdf_ocr` copre la PII DENTRO i pixel: il valore non e'
     piu' leggibile nel PDF di output e la casella c'e' davvero (si controlla
     il colore del pixel, non solo il conteggio).

Nessun motore OCR vero: si usa un motore finto che riconosce una riga
sempre uguale. I valori sono SINTETICI.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "app"))

import fitz  # noqa: E402

import ocr  # noqa: E402
import pdf_export  # noqa: E402


class FakeEngine:
    """Motore finto: riconosce sempre la stessa riga, a coordinate fisse."""

    name = "fake"
    detail = "motore finto di test"

    def __init__(self, righe=None):
        self.righe = righe or [{"text": "Mario Rossi",
                                "box": (100, 200, 400, 230), "conf": 0.98}]
        self.chiamate = 0

    def recognize(self, image_bytes):
        self.chiamate += 1
        return list(self.righe)

    def describe(self):
        return self.detail


def pdf_scansionato(pagine=1):
    """PDF con immagine e SENZA testo: la classica scansione."""
    doc = fitz.open()
    for _ in range(pagine):
        page = doc.new_page(width=595, height=842)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 300))
        pix.set_rect(pix.irect, (255, 255, 255))
        page.insert_image(page.rect, pixmap=pix)
    out = doc.tobytes()
    doc.close()
    return out


def pdf_con_testo(testo="Il sottoscritto Mario Rossi dichiara."):
    """PDF con layer testuale: NON e' una scansione."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), testo, fontname="helv", fontsize=12)
    out = doc.tobytes()
    doc.close()
    return out


class Scansioni(unittest.TestCase):
    def test_la_pagina_senza_testo_e_una_scansione(self):
        with fitz.open(stream=pdf_scansionato(), filetype="pdf") as doc:
            self.assertEqual(ocr.scanned_pages(doc), [0])

    def test_la_pagina_con_testo_non_e_una_scansione(self):
        with fitz.open(stream=pdf_con_testo(), filetype="pdf") as doc:
            self.assertEqual(ocr.scanned_pages(doc), [])

    def test_la_pagina_vuota_non_e_una_scansione(self):
        """Una pagina bianca non ha niente da riconoscere: OCR inutile."""
        doc = fitz.open()
        doc.new_page(width=595, height=842)
        out = doc.tobytes()
        doc.close()
        with fitz.open(stream=out, filetype="pdf") as doc:
            self.assertEqual(ocr.scanned_pages(doc), [])


class OcrPdf(unittest.TestCase):
    def test_il_testo_riconosciuto_e_le_coordinate_in_punti_pdf(self):
        res = ocr.ocr_pdf(pdf_scansionato(), pages=[0], dpi=200, engine=FakeEngine())
        self.assertEqual(res["pages"][0]["text"], "Mario Rossi")
        self.assertEqual(res["text"], "Mario Rossi")
        # i pixel del motore (100,200)-(400,230) a 200 DPI sono 72/200 = 0.36 punti
        line = res["pages"][0]["lines"][0]
        self.assertAlmostEqual(line["x0"], 36.0, places=1)
        self.assertAlmostEqual(line["y0"], 72.0, places=1)
        self.assertAlmostEqual(line["x1"], 144.0, places=1)
        self.assertAlmostEqual(line["y1"], 82.8, places=1)

    def test_gli_offset_del_testo_corrispondono_alle_righe(self):
        """Il testo della pagina e' la concatenazione delle righe: gli offset
        servono a mappare una PII sulla casella da coprire."""
        eng = FakeEngine([{"text": "Mario Rossi", "box": (0, 0, 100, 20), "conf": 0.9},
                          {"text": "IBAN IT60X0542811101000000123456",
                           "box": (0, 40, 300, 60), "conf": 0.9}])
        res = ocr.ocr_pdf(pdf_scansionato(), pages=[0], dpi=200, engine=eng)
        page = res["pages"][0]
        self.assertEqual(page["text"], "Mario Rossi\nIBAN IT60X0542811101000000123456")
        for line in page["lines"]:
            self.assertEqual(page["text"][line["start"]:line["end"]], line["text"])


class RedazioneBurnIn(unittest.TestCase):
    def _redige(self, pdf_bytes, mapping, righe=None):
        eng = FakeEngine(righe)
        res = ocr.ocr_pdf(pdf_bytes, pages=[0], dpi=200, engine=eng)
        return pdf_export.redact_pdf_ocr(pdf_bytes, mapping, res, dpi=200)

    def test_la_pii_non_e_piu_leggibile_nel_pdf_di_output(self):
        out, report = self._redige(pdf_scansionato(),
                                   {"[FULLNAME_1]": "Mario Rossi"})
        self.assertEqual(report["occurrences"], 1)
        self.assertEqual(report["residual"], [])
        with fitz.open(stream=out, filetype="pdf") as doc:
            testo = "".join(p.get_text() for p in doc)
        self.assertNotIn("Mario Rossi", testo)
        self.assertIn("[FULLNAME_1]", testo)     # il placeholder c'e', e non la PII

    def test_la_casella_e_davvero_disegnata_nei_pixel(self):
        """Il controllo vero: il pixel sotto la PII non e' piu' bianco.

        Se la casella fosse solo un rettangolo sovrapposto, il pixel resterebbe
        bianco (e la PII sarebbe recuperabile dal file)."""
        out, _report = self._redige(pdf_scansionato(),
                                    {"[FULLNAME_1]": "Mario Rossi"})
        with fitz.open(stream=out, filetype="pdf") as doc:
            pix = doc.load_page(0).get_pixmap(dpi=200)
        # il box del motore era (100,200)-(400,230) px a 200 DPI -> centro (250,215)
        x, y = 250, 215
        i = (y * pix.width + x) * pix.n
        px = tuple(pix.samples[i:i + 3])
        self.assertNotEqual(px, (255, 255, 255),
                            "il pixel sotto la PII e' ancora bianco: la casella "
                            "non e' stata rasterizzata")

    def test_il_valore_troppo_corto_viene_ignorato_e_segnalato(self):
        out, report = self._redige(
            pdf_scansionato(),
            {"[BUILDINGNUM_1]": "24"},
            righe=[{"text": "via Roma 24", "box": (0, 0, 300, 20), "conf": 0.9}])
        self.assertEqual(report["skipped"], ["[BUILDINGNUM_1]"])
        self.assertEqual(report["occurrences"], 0)

    def test_dizionario_vuoto_e_un_errore(self):
        with self.assertRaises(pdf_export.PdfError):
            pdf_export.redact_pdf_ocr(pdf_scansionato(), {}, {"pages": []})


class MotoreNonInstallato(unittest.TestCase):
    def test_lo_stato_del_motore_e_una_cosa_sola(self):
        st = ocr.engine_status()
        self.assertIn("available", st)
        self.assertIn("engine", st)
        if not st["available"]:
            self.assertIn("tesseract", st["detail"].lower())


if __name__ == "__main__":
    unittest.main()
