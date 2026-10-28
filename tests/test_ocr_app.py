# -*- coding: utf-8 -*-
"""L'OCR integrato visto dall'API: `/analyze` e `/pdf` su un PDF scansionato.

Prima dell'OCR integrato un PDF senza layer testuale rispondeva 400 ("Nessun
testo da analizzare") e `/pdf` 422. Ora:
  - se l'OCR e' attivo e c'e' un motore, il testo riconosciuto entra nell'analisi;
  - se non c'e' nessun motore, l'errore lo dice (e il codice e' "ocr_unavailable");
  - se l'OCR e' disattivato, l'errore lo dice (codice "ocr_disabled").

I valori sono SINTETICI.
"""

import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import fitz  # noqa: E402
from test_ora import APP  # noqa: E402  (stesso caricamento di app coi pesi finti)

IBAN = "IT60X0542811101000000123456"


class FakeEngine:
    """Motore finto: riconosce un IBAN valido a coordinate fisse."""

    name = "fake"
    detail = "motore finto di test"

    def recognize(self, image_bytes):
        return [{"text": f"IBAN {IBAN}", "box": (50, 100, 700, 130), "conf": 0.95}]

    def describe(self):
        return self.detail


def pdf_scansionato():
    """PDF con immagine e senza testo."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 300))
    pix.set_rect(pix.irect, (255, 255, 255))
    page.insert_image(page.rect, pixmap=pix)
    out = doc.tobytes()
    doc.close()
    return out


def pdf_con_testo(testo="Il bonifico va sull'IBAN IT60X0542811101000000123456."):
    """PDF con layer testuale (niente OCR: il percorso normale)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), testo, fontname="helv", fontsize=12)
    out = doc.tobytes()
    doc.close()
    return out


@unittest.skipIf(APP is None, "app.py non importabile (torch/flask/fitz assenti)")
class OcrNellaApi(unittest.TestCase):
    def setUp(self):
        self.pdf = pdf_scansionato()
        self._engine = APP.ocr.detect_engine
        self._enabled = APP.OCR_ENABLED
        APP.ocr.detect_engine = lambda prefer=None, refresh=False: FakeEngine()
        APP.OCR_ENABLED = True
        APP._OCR_CACHE.clear()
        self.client = APP.app.test_client()

    def tearDown(self):
        APP.ocr.detect_engine = self._engine
        APP.OCR_ENABLED = self._enabled
        APP._OCR_CACHE.clear()

    def _upload(self, path="/analyze", **extra):
        data = {"pdf": (io.BytesIO(self.pdf), "scansione.pdf")}
        data.update(extra)
        return self.client.post(path, data=data, content_type="multipart/form-data")

    def test_analyze_usa_il_testo_riconosciuto(self):
        r = self._upload()
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual(d["ocr"]["engine"], "fake")
        self.assertEqual(d["ocr"]["pages"], 1)
        self.assertGreaterEqual(d["n_entities"], 1)
        self.assertIn("[IBAN_1]", d["anonymized_text"])
        self.assertNotIn(IBAN, d["anonymized_text"])

    def test_il_pdf_redatto_e_generato_e_il_valore_non_c_e_piu(self):
        r = self._upload("/pdf/preview")
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertTrue(d["doc_id"])
        self.assertEqual(d["ocr_pages"], [0])          # pagina redatta in burn-in
        pdf = self.client.get(f"/doc/{d['doc_id']}/file.pdf")
        self.assertEqual(pdf.status_code, 200)
        with fitz.open(stream=pdf.data, filetype="pdf") as doc:
            testo = "".join(p.get_text() for p in doc)
        self.assertNotIn(IBAN, testo)
        self.assertIn("[IBAN_1]", testo)

    def test_senza_motore_l_errore_dice_cosa_installare(self):
        APP.ocr.detect_engine = lambda prefer=None, refresh=False: None
        r = self._upload()
        self.assertEqual(r.status_code, 400)
        d = r.get_json()
        self.assertEqual(d["code"], "ocr_unavailable")
        self.assertIn("tesseract", d["error"].lower())

    def test_ocr_disattivato_l_errore_lo_dice(self):
        APP.OCR_ENABLED = False
        r = self._upload()
        self.assertEqual(r.status_code, 400)
        d = r.get_json()
        self.assertEqual(d["code"], "ocr_disabled")

    def test_health_dice_se_c_e_un_motore(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertIn("ocr", d)
        self.assertTrue(d["ocr"]["available"])
        self.assertEqual(d["ocr"]["engine"], "fake")

    def test_settings_permette_di_spegnere_l_ocr(self):
        r = self.client.get("/settings")
        self.assertTrue(r.get_json()["ocr_enabled"])
        r = self.client.post("/settings", json={"ocr_enabled": False})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.get_json()["ocr_enabled"])
        self.assertFalse(self.client.get("/settings").get_json()["ocr_enabled"])
        # ripristina (le preferenze sono su disco: non lasciare il server spento)
        self.client.post("/settings", json={"ocr_enabled": True})


@unittest.skipIf(APP is None, "app.py non importabile (torch/flask/fitz assenti)")
class CopertureAnnullabili(unittest.TestCase):
    """L'anteprima permette di annullare la copertura di una PII coperta per
    errore: un click sulla casella -> il server rigenera il PDF senza quella
    copertura, e il documento viene segnalato come "valori rimasti in chiaro"."""

    def setUp(self):
        self._engine = APP.ocr.detect_engine
        APP.ocr.detect_engine = lambda prefer=None, refresh=False: FakeEngine()
        APP.OCR_ENABLED = True
        APP._OCR_CACHE.clear()
        self.client = APP.app.test_client()

    def tearDown(self):
        APP.ocr.detect_engine = self._engine
        APP._OCR_CACHE.clear()

    def _preview(self, pdf):
        data = {"pdf": (io.BytesIO(pdf), "documento.pdf")}
        r = self.client.post("/pdf/preview", data=data,
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        return r.get_json()

    def _testo_del_file(self, doc_id):
        pdf = self.client.get(f"/doc/{doc_id}/file.pdf")
        with fitz.open(stream=pdf.data, filetype="pdf") as doc:
            return "".join(p.get_text() for p in doc)

    def test_l_anteprima_elenca_le_coperture(self):
        d = self._preview(pdf_con_testo())
        self.assertTrue(d["hits"])
        h = d["hits"][0]
        self.assertEqual(h["page"], 0)
        self.assertEqual(h["placeholder"], "[IBAN_1]")
        self.assertTrue(h["rects"])
        self.assertGreater(h["w"], 0)

    def test_annullare_la_copertura_riporta_il_valore_in_chiaro(self):
        d = self._preview(pdf_con_testo())
        self.assertNotIn(IBAN, self._testo_del_file(d["doc_id"]))
        r = self.client.post(f"/doc/{d['doc_id']}/redactions",
                             json={"excluded": [d["hits"][0]["id"]]})
        self.assertEqual(r.status_code, 200)
        d2 = r.get_json()
        self.assertEqual(d2["excluded"], [d["hits"][0]["id"]])
        self.assertEqual(d2["redactions"], 0)
        self.assertGreaterEqual(d2["residual"], 1)      # avviso: resta in chiaro
        self.assertIn(IBAN, self._testo_del_file(d["doc_id"]))

    def test_ripristinare_la_copertura_rioccupa_il_valore(self):
        d = self._preview(pdf_con_testo())
        hid = d["hits"][0]["id"]
        self.client.post(f"/doc/{d['doc_id']}/redactions", json={"excluded": [hid]})
        r = self.client.post(f"/doc/{d['doc_id']}/redactions", json={"excluded": []})
        d2 = r.get_json()
        self.assertEqual(d2["excluded"], [])
        self.assertEqual(d2["redactions"], d["redactions"])
        self.assertNotIn(IBAN, self._testo_del_file(d["doc_id"]))

    def test_il_documento_inesistente_da_errore(self):
        r = self.client.post("/doc/inesistente/redactions", json={"excluded": []})
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
