# rizzo-pii — Manuale di Sviluppo Completo

## Indice

1. [Panoramica Architetturale](#panoramica-architetturale)
2. [Struttura del Repository](#struttura-del-repository)
3. [Pipeline Dati](#pipeline-dati)
4. [Training del Modello](#training-del-modello)
5. [App di Anonimizzazione Locale](#app-di-anonimizzazione-locale)
6. [API HTTP](#api-http)
7. [Rete Regex + Checksum](#rete-regex--checksum)
8. [Redazione PDF](#redazione-pdf)
9. [App Desktop Tauri](#app-desktop-tauri)
10. [Testing](#testing)
11. [Build & Packaging](#build--packaging)
12. [Estrazione Zone PII con Coordinate (Nuova Feature)](#estrazione-zone-pii-con-coordinate-nuova-feature)
13. [Convenzioni di Codice](#convenzioni-di-codice)
14. [Troubleshooting Comune](#troubleshooting-comune)

---

## Panoramica Architetturale

### Principio Fondamentale: "LLM Autore, Codice Etichettatore"

Il cuore della generazione sintetica: **l'LLM scrive solo la prosa con segnaposto `{SLOT}`, il codice inietta i dati reali**. Questo risolve tre problemi insieme:
1. Label BIO esatte (sappiamo esattamente dove iniettiamo)
2. Checksum matematicamente validi (CF/PIVA/IBAN)
3. Nessuna PII reale rigurgitata dall'LLM

### Flusso Dati Principale

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DI ADDESTRAMENTO                     │
├─────────────────────────────────────────────────────────────────┤
│  llm_template_bank.py     → template legali con {SLOT}          │
│  generate_synthetic_pii.py → inietta valori validi (checksum)   │
│  augment_real_pii.py       → entità sintetiche in frasi reali   │
│  prepare_deepmount.py      → rimappa DeepMount 56→22 tag       │
│  build_validation.py       → validation reale unificata (7k IT) │
│  build_subset.py           → subset stratificati per smoke test │
│                              ↓                                  │
│  train_pii.py              → mmBERT fine-tuning, salva modello  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    APP DI ANONIMIZZAZIONE                        │
├─────────────────────────────────────────────────────────────────┤
│  Modello mmBERT (token-classification)                          │
│       +                                                         │
│  Rete Regex + Checksum (deterministica)                         │
│       =                                                         │
│  Entità fuse → ID reversibili [TAG_N] → Testo anonimizzato      │
│       ↓                                                         │
│  Dizionario locale {placeholder → valore reale}                 │
│       ↓                                                         │
│  PDF: redazione vera (PyMuPDF apply_redactions)                 │
│  Testo: PDF ricostruito da zero                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tassonomia: 22 Tag + 2 Solo-Regex

| Tag | Descrizione | Esempio | Origine |
|-----|-------------|---------|---------|
| `FULLNAME` | Nome persona (ruoli legali inclusi) | Mario Rossi | Real + Synth |
| `AGE` | Età | 45 anni | Real |
| `GENDER` | Sesso/genere | Femmina | Real |
| `DATE` | Data calendario | 12/06/1985 | Real + Synth |
| `TIME` | Ora | ore 15:30 | Real |
| `STREET` | Via/piazza | Via Garibaldi | Real + Synth |
| `BUILDINGNUM` | Numero civico | 24 | Real + Synth |
| `ZIPCODE` | CAP | 00185 | Real + Synth |
| `CITY` | Città | Milano | Real + Synth |
| `PROVINCE` | Sigla provincia | MI | Synth |
| `EMAIL` | Email (PEC inclusa) | m.rossi@studio.it | Real + Synth |
| `TELEPHONENUM` | Telefono | +39 333 1234567 | Real + Synth |
| `CF` | Codice fiscale (checksum) | RSSMRA85H12F205Y | Synth |
| `PIVA` | Partita IVA (checksum) | 12345678903 | Real + Synth |
| `ID_DOC` | Doc identità/passaporto/patente | CA12345AB | Real + Synth |
| `IBAN` | IBAN/conto (checksum mod-97) | IT60X05428... | Synth |
| `CREDITCARDNUMBER` | Carta credito (Luhn) | 4111 1111 1111 1111 | Real |
| `AMOUNT` | Importo denaro | € 12.500,00 | Synth |
| `TARGA` | Targa veicolo | AB 123 CD | Synth |
| `ORG` | Società/studio/banca | Edilnord S.r.l. | Synth |
| `DOCID` | Codice atto (RG, protocollo...) | 1234/2024 | Synth |
| `CATASTO` | Dati catastali | Foglio 12, part. 345 | Synth |
| `URL` | **Solo regex** | https://example.it | Regex |
| `IPADDR` | **Solo regex** | 192.168.1.1 | Regex |

**Decisioni chiave:**
- Ruoli legali (giudice, avvocato, parte, teste) → `FULLNAME`
- `PEC` → `EMAIL`, `TAXNUM` → `PIVA`
- `IDCARDNUM`/`PASSPORTNUM`/`DRIVERLICENSENUM`/`SOCIALNUM` → `ID_DOC`
- `CONTO` → `IBAN`
- `TITLE` (appellativi), `TRIBUNAL` (enti pubblici) → `O` (scartati)

---

## Struttura del Repository

```
rizzo-pii/
├─ README.md                    # Documentazione utente completa
├─ DEVELOPMENT_MANUAL.md        # Questo file
├─ CLAUDE.md                    # Istruzioni operative + vincoli ambiente
├─ CONTRIBUTING.md              # Come contribuire (codice, docs, dati)
├─ LICENSE                      # MIT
├─ THIRD_PARTY_LICENSES.md      # Licenze dipendenze nei binari
├─ requirements.txt             # Dipendenze Python (vedi nota cu128)
├─ Dockerfile                   # Web app in container (CPU + modello)
├─ Dockerfile.linux             # Build env riproducibile Linux .deb/AppImage
├─ build.spec                   # PyInstaller legacy
├─ build_sidecar.spec           # PyInstaller sidecar per Tauri
├─ installer.iss                # NSIS installer Windows
├─ docker-compose.yml           # Compose per deployment
├─ report/                      # Report tecnico (PDF + Typst source)
├─ docs/
│   ├─ DATASET.md               # Composizione completa train/validation
│   ├─ TASSONOMIA_TAG.md        # 22 tag finali e decisioni merge
│   ├─ TRAINING.md              # Training run: union dataset, iperparametri
│   ├─ FORMATO_DATI.md          # Formato esatto row dataset (JSONL/BIO)
│   ├─ BUILD.md                 # Build desktop (Tauri + PyInstaller legacy)
│   ├─ CHANGELOG.md             # Change log con rationale
│   └─ star_history_*.png       # Grafici stelle (rigenerabili)
├─ scripts/
│   └─ plot_star_history.py     # Rigenera grafici stelle
├─ src/
│   ├─ data_pipeline/           # Generazione & preparazione dati
│   │   ├─ llm_template_bank.py         # Gemini scrive template → legal_templates.json
│   │   ├─ generate_synthetic_pii.py    # Inietta valori checksum-validi
│   │   ├─ augment_real_pii.py          # Entità sintetiche in frasi reali Ai4Privacy
│   │   ├─ prepare_deepmount.py         # Rimappa DeepMount 56→22 tag
│   │   ├─ build_validation.py          # Validation reale unificata (7k IT)
│   │   ├─ build_subset.py              # Subset stratificati 10k/5k
│   │   └─ contribute_dataset.py        # Contribuzione dataset HF community
│   ├─ training/
│   │   ├─ train_pii.py                 # Training, evaluation, salvataggio modello
│   │   ├─ evaluate_pii.py              # Valutazione per-tag (P/R/F1)
│   │   └─ test_pii.py                  # CLI inference su modello salvato
│   ├─ inspect/                         # Utility read-only
│   │   ├─ validate_checksums.py        # Blueprint rete regex+checksum
│   │   ├─ inspect_ai4privacy.py        # Conteggi lingue/tag
│   │   ├─ inspect_lengths.py           # Analisi lunghezze
│   │   └─ inspect_no_iban.py           # Analisi IBAN mancanti
│   └─ app/                             # App anonimizzazione locale
│       ├─ app.py                       # Flask server: anonimizzazione + regex/checksum
│       ├─ pdf_export.py                # PDF anonimizzato: redazione vera
│       ├─ pdf_text.py                  # Estrazione testo leggibile (pagine+annot+widget+TOC)
│       ├─ detectors.py                 # Rete regex+checksum (standalone, testabile)
│       ├─ serve.py                     # Entry headless (backend Tauri)
│       ├─ desktop_app.py               # Entry PyInstaller legacy (apre browser)
│       ├─ server_config.py             # Config host/porta + prefs.json
│       └─ assets/                      # Mascotte + icone
├─ tauri/                               # App desktop nativa (Tauri)
│   ├─ src-tauri/                       # Rust backend
│   ├─ ui/                              # Frontend HTML/JS (servito da Flask)
│   ├─ package.json / Cargo.toml        # Dipendenze Node/Rust
│   └─ build_sidecar.spec               # Spec per bundling sidecar Python
├─ dataset/           (gitignored, rigenerabile)
├─ models/            (gitignored, modelli per versione)
├─ experiments/       (gitignored, artefatti run)
└─ wandb/             (gitignored, log W&B)
```

**Punti chiave:**
- Tutti gli script risolvono i path da `__file__` → girano da qualsiasi CWD
- Forzano UTF-8 su stdout/stderr
- Artefatti (`dataset/`, `models/`, `experiments/`, `wandb/`, `build/`, `dist/`, `tauri/node_modules/`, `tauri/src-tauri/target/`) sono **gitignored**

---

## Pipeline Dati

### Ordine di Esecuzione Obbligatorio

```bash
# 1) Template legali via Gemini (opzionale, richiede GEMINI_API_KEY)
python src/data_pipeline/llm_template_bank.py --per-type 5 --append

# 2) Sintetico da template (200k righe, copre tag legali IT)
python src/data_pipeline/generate_synthetic_pii.py -n 200000 \
  --out dataset/synthetic/synthetic_pii_it_200k.jsonl

# 3) Augment in testo reale (40k, spezza legame template/posizione)
python src/data_pipeline/augment_real_pii.py -n 40000 \
  --out dataset/synthetic/synthetic_pii_it_realaug.jsonl

# 4) DeepMount rimappato 56→22 tag (richiede hf auth login)
python src/data_pipeline/prepare_deepmount.py

# 5) Validation reale unificata (7k, solo italiano)
python src/data_pipeline/build_validation.py

# 6) Subset stratificati per smoke test
python src/data_pipeline/build_subset.py
```

### Formato Dati (JSONL, BIO)

Ogni riga:
```json
{
  "source_text": "Il Sig. Mario Rossi, C.F. RSSMRA85H12F205Y...",
  "tokens": ["Il", "Sig.", "Mario", "Rossi", ",", "C.F.", "RSSMRA85H12F205Y", "..."],
  "bio_labels": ["O", "O", "B-FULLNAME", "I-FULLNAME", "O", "O", "B-CF", "..."],
  "entities": [
    {"label": "FULLNAME", "start": 7, "end": 18, "text": "Mario Rossi"},
    {"label": "CF", "start": 25, "end": 41, "text": "RSSMRA85H12F205Y"}
  ],
  "schema": "BIO"
}
```

**Vincoli obbligatori:**
- Checksum CF/PIVA/IBAN **validi** per ogni entità sintetica
- Nessuna PII reale nei dati sintetici
- Vedi `docs/FORMATO_DATI.md` per specifica completa

### Fonti Dati (totale ~745k righe train)

| Fonte | Righe | Lingue | Note |
|-------|-------|--------|------|
| Ai4Privacy `open-pii-masking-500k` | ~464k | 8 (IT ~55k) | Reale, `mbert_tokens`+`mbert_token_classes` |
| Sintetico da template | 200k | IT | Tag legali IT-only (CF, PIVA, CATASTO, DOCID, PROVINCE) |
| Augment reali | 40k | IT | Entità sintetiche in frasi reali held-out |
| DeepMount `pii-masking-ita` | ~41k | IT | Faker IT, contesto reale per IBAN/ORG/AMOUNT/TARGA |

**Pool train ≈ 745k righe** (multilingue; italiano ~45%). ~38% sintetico.

### Validation: UNA Validation Reale Unificata

`validation_real.jsonl` (7k righe, **solo italiano**):
- Base reale held-out = Ai4Privacy val (IT) + DeepMount test
- I 5 tag senza dato reale (CF, PIVA, CATASTO, DOCID, PROVINCE) **iniettati in frasi reali held-out** (frasi validation Ai4, non nel training) → contesto reale, niente leakage

---

## Training del Modello

### Script Principale: `src/training/train_pii.py`

```bash
# Smoke test / tuning su subset (~3 min) → experiments/subset_smoke/
python src/training/train_pii.py --type subset

# Run grande su tutto → models/rizzo-pii-0.3B-v{VERSION}/ + experiments/full_run_v{VERSION}/
python src/training/train_pii.py --type full
python src/training/train_pii.py --type full --version 1.2.0  # versione esplicita
```

### Parametri Critici (in `train_pii.py`)

| Parametro | Valore | Rationale |
|-----------|--------|-----------|
| `LANG` | `None` (multilingue) | 8 lingue Ai4Privacy; IT rinforzato da synth/DeepMount |
| `MAX_LEN` | 768 | Copre sintetici (max 771 subword) e DeepMount (660); con 512 si troncava 33% sintetici |
| `BATCH` | 16 | + `GRAD_ACCUM=2` → batch effettivo 32; batch >16 saturano VRAM 16GB condivisa → thrashing |
| `EPOCHS` | 1 | Loss valida monotona decrescente; seconda epoch spingerebbe ancora più basso |
| `group_by_length` | `True` | `LengthGroupedTrainer` usa lunghezze precalcolate, no ri-tokenizzazione lazy |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Mitiga frammentazione allocatore CUDA |

### Vincoli Ambiente Critici (da `CLAUDE.md`)

- **GPU RTX 5060 Ti (Blackwell, sm_120)**: `torch DEVE essere build cu128` (`pip install torch --index-url https://download.pytorch.org/whl/cu128`). Build cpu/cu121 non supportano sm_120.
- **torchvision/torchaudio vanno disinstallati** se vecchie versioni: rompono import transformers (`operator torchvision::nms does not exist`).
- **accelerate ≥ 1.14** (transformers 4.57 chiama `unwrap_model(keep_torch_compile=...)`).
- **Windows**: `dataloader_num_workers=0` nel Trainer (altrimenti `RuntimeError ... bootstrapping`).
- **seqeval non si compila** (bug setuptools_scm) → metriche entity-level calcolate a mano negli script.

### Output Training

```
models/rizzo-pii-0.3B-v{VERSION}/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt

experiments/full_run_v{VERSION}/
├── loss_plot.png
└── out/ (checkpoint Hugging Face)

models/registry.json  # storico versioni
```

### Modello Versionato

- Ogni run grande salva in `models/rizzo-pii-0.3B-v{VERSION}/` (storico, **niente sovrascrittura**)
- Versione da `MODEL_VERSION` in cima a `train_pii.py` o `--version`
- `test_pii.py` risolve automaticamente l'**ultima** versione `-v*` disponibile
- Override runtime: `PII_MODEL_DIR` env var

### Weights & Biases

- `train_pii.py` carica `.env` con `load_dotenv()`
- Se `WANDB_API_KEY` presente → `report_to=["wandb"]` (progetto `pii-mmbert-it`)
- Logga: train loss live per step + metriche finali (`final/train_*`, `final/val_*`)
- `.env` è **gitignored**: non committare

---

## App di Anonimizzazione Locale

### Entry Points

| Script | Descrizione |
|--------|-------------|
| `app.py` | Flask server **con UI** (browser) — `python src/app/app.py` → http://127.0.0.1:5005 |
| `serve.py` | **Headless** (solo Flask, niente browser) — backend dell'app Tauri |
| `desktop_app.py` | Legacy PyInstaller (apre browser) |

### Configurazione Host/Porta (Precedenza)

1. CLI `--host`/`--port`
2. Env `PII_HOST`/`PII_PORT`
3. `config.json` in:
   - Windows: `%LOCALAPPDATA%\rizzo-pii\`
   - Linux: `~/.local/share/rizzo-pii/`
   - macOS: `~/Library/Application Support/rizzo-pii/`
4. Default: `127.0.0.1:5005`

**Pre-check porta**: tutti e 3 gli entry point fanno `port_available()` **prima** di caricare il modello. Se occupata → `sys.exit(76)`. Tauri riconosce codice 76 e mostra form configurazione nello splash.

### Preferenze Anonimizzazione (Precedenza)

| Impostazione | CLI | Env | File (`prefs.json`) | Default |
|--------------|-----|-----|---------------------|---------|
| `excluded_tags` | `--exclude-tags` | `PII_EXCLUDE_TAGS` | `prefs.json` | `[]` |
| `mapping_enabled` | `--no-mapping` | `PII_MAPPING=0` | `prefs.json` | `true` |

Per-request override: campi `exclude_tags` e `include_mapping` in `/analyze` e `/pdf`.

### Funzionamento Core: `analyze(text, excluded, mapping_enabled)`

```python
def analyze(text, excluded=None, mapping_enabled=True):
    # 1. Chunking word-safe (MAX_WORDS=120, OVERLAP=20)
    chunks = chunk_text(text)
    
    # 2. Inference modello su tutti i chunk
    model_ents = detect_model(text)  # offset globali
    
    # 3. Completamento ore (modello taglia minuti)
    complete_time(model_ents, text)
    
    # 4. Rete regex + checksum
    regex_ents = detect_regex(text)
    
    # 5. Esclusione tag (rilevati ma NON sostituiti)
    if excluded: cands = [e for e in cands if e["label"] not in excluded]
    
    # 6. Fusione greedy no-overlap
    # Priorità: checksum-valido > regex (non soft) > score > lunghezza
    kept = _merge(cands, text)
    
    # 7. Allineamento confini parola (estende per coprire parola intera)
    # 8. Merge span adiacenti/overlapping stesso label
    
    # 9. ID reversibili: stesso (label, valore_normalizzato) → stesso placeholder
    # mapping_enabled=False: niente dizionario, segmenti perdono campo 't'
    
    return {
        "segments": [...],           # per preview UI
        "anonymized_text": "...",    # testo con placeholder
        "mapping": {...},            # {placeholder: valore} se mapping_enabled
        "mapping_enabled": bool,
        "n_chunks": int,
        "n_chars": int,
        "n_entities": int,
        "n_unique": int,
        "by_label": {...},
        "by_source": {...},
        "excluded_tags": [...]
    }
```

### Anteprima PDF a Video (Server-Side Rendering)

- **Perché**: Tauri (WebView2/WKWebView) non ha viewer PDF affidabile; app offline → niente pdf.js da CDN
- **Come**: PyMuPDF `page.get_pixmap(dpi=110)` → PNG per pagina
- **Store**: LRU in memoria (`_DOCS`, `MAX_DOCS=6`), **mai su disco**
- **Lazy loading**: pagine renderizzate on-demand, poi cached
- **Endpoints**:
  - `POST /preview` → carica PDF, ritorna `{doc_id, n_pages, text}`
  - `GET /doc/<id>/page/<n>.png` → pagina renderizzata
  - `POST /pdf/preview` → anonimizza + store, ritorna descrittore (anteprima dx + download usano stessa copia)

### Tag Disattivabili

Utente deseleziona tag in UI (icona 🏷️) → entità rilevate ma **NON sostituite** (restano in chiaro).
Serve per: confrontare importi (`AMOUNT`), tenere età/sesso in caso clinico (`AGE`, `GENDER`).

### Dizionario Reversibile On/Off

| Modalità | Comportamento |
|----------|---------------|
| **On (default)** | Placeholder `[TAG_N]` + dizionario locale `placeholder→valore`; tab "Ripristina" per rimettere valori da risposta LLM |
| **Off** | Anonimizzazione **definitiva**: nessun dizionario costruito/restituito/salvato; segmenti **perdono campo `t`** (altrimenti ricostruibile); UI nasconde download; numerazione placeholder resta (utile a LLM per coerenza) |

---

## API HTTP

### Endpoint Principali

| Endpoint | Metodo | Input | Output | Descrizione |
|----------|--------|-------|--------|-------------|
| `/health` | GET | — | JSON | Readiness senza inference (200=modello caricato, 503=no) |
| `/analyze` | POST | `{"text":...}` o multipart file | JSON | Entità + testo anonimizzato + dizionario |
| `/pdf` | POST | Come `/analyze` | **PDF binario** | Download PDF anonimizzato (redazione vera se input PDF) |
| `/preview` | POST | Multipart `.pdf` | JSON | Carica PDF per anteprima sx → `{doc_id, n_pages, text}` |
| `/pdf/preview` | POST | Come `/pdf` | JSON | Anonimizza + store in memoria → `{doc_id, n_pages, ...}` |
| `/doc/<id>/page/<n>.png` | GET | — | PNG | Pagina n (0-based) del documento in memoria |
| `/doc/<id>/file.pdf` | GET | — | PDF | Download documento anonimizzato in memoria |
| `/settings` | GET | — | JSON | Legenda tag, tag esclusi, stato dizionario |
| `/settings` | POST | `{"excluded_tags":[...], "mapping_enabled":bool}` | JSON | Salva preferenze in `prefs.json` |
| `/config` | GET | — | JSON | Host/porta correnti + path config |
| `/config` | POST | `{"host":"...", "port":N}` | JSON | Salva host/porta per prossimo avvio |
| `/port-check` | GET | `?host=...&port=...` | JSON | `{"available":bool}` pre-bind check |

### Formato Risposta `/analyze`

```json
{
  "segments": [
    {"t": "Il Sig. "},
    {"label": "FULLNAME", "ph": "[FULLNAME_1]", "src": "modello", "validated": false, "t": "Mario Rossi"},
    {"t": ", C.F. "},
    {"label": "CF", "ph": "[CF_1]", "src": "regex", "validated": true, "t": "RSSMRA85H12F205Y"},
    {"t": " è titolare..."}
  ],
  "anonymized_text": "Il Sig. [FULLNAME_1], C.F. [CF_1] è titolare...",
  "mapping": {"[FULLNAME_1]": "Mario Rossi", "[CF_1]": "RSSMRA85H12F205Y"},
  "mapping_enabled": true,
  "n_chunks": 1,
  "n_chars": 156,
  "n_entities": 2,
  "n_unique": 2,
  "by_label": {"FULLNAME": 1, "CF": 1},
  "by_source": {"modello": 1, "regex": 1},
  "excluded_tags": [],
  "source_text": "Il Sig. Mario Rossi, C.F. RSSMRA85H12F205Y..."  // solo se mapping_enabled
}
```

### Header Diagnostici `/pdf`

| Header | Significato |
|--------|-------------|
| `X-PII-Redactions` | Numero redazioni applicate |
| `X-PII-Residual` | Placeholder il cui valore è **ancora leggibile** nell'output (avviso UI) |
| `X-PII-Skipped` | Valori **troppo corti/ambigui** non redatti (es. "45", "C") — restano in chiaro, avviso UI |
| `X-PII-Notfound` | Placeholder cercati ma **senza occorrenze** nel PDF |

---

## Rete Regex + Checksum (`src/app/detectors.py`)

Modulo **standalone**, zero dipendenze ML, testabile in isolamento.

### Detectors Configurati

```python
DETECTORS = [
    ("EMAIL",      regex, None,          True),   # solo forma
    ("CF",         regex, cf_ok,         False),  # forma specifica, checksum opzionale
    ("CF",         regex_omocodia, cf_ok, True),  # omocodia: checksum OBBLIGATORIO
    ("IBAN",       regex_compact, iban_ok, True), # forma compatta
    ("CREDITCARD", regex,         luhn_ok, True), # Luhn obbligatorio
    ("PIVA",       regex,         piva_ok,  True), # mod-10 obbligatorio
    ("TELEPHONENUM", regex,       None,      True),
    ("AMOUNT",     regex,         None,      True),
    ("TARGA",      regex,         None,      True),
    ("URL",        regex_3forme,  None,      True), # schema / www / dominio nudo (TLD closed list)
    ("IPADDR",     regex_strict,  None,      True), # ottetti 0-255 + CIDR/range
    ("DOCID",      regex_con_sigla, None,      True), # sigla obbligatoria (RG, Prot, Rep)
    ("DATE",       regex,         None,      True),  # SOFT: modello può sovrascrivere
]
```

### Logica Priorità in `_merge()` (app.py)

```python
order = sorted(cands, key=lambda e: (
    1 if e["validated"] else 0,                    # 1. Checksum valido (top)
    1 if e["source"]=="regex" and e["label"] not in SOFT_REGEX_LABELS else 0,  # 2. Regex non-soft
    e["score"],                                     # 3. Score
    e["end"] - e["start"]                           # 4. Lunghezza (più lungo vince)
), reverse=True)
```

### Funzioni Chiave

- `detect_regex(text)` → lista entità con `validated=True/False`
- `detect_iban(text)` → scanner per IBAN formato stampa a gruppi (usa `IBAN_LEN` per paese)
- `complete_time(ents, text)` → estende span `TIME` del modello all'ora completa (modello taglia minuti)
- Validatori: `iban_ok` (mod-97), `piva_ok` (mod-10), `cf_ok` (algoritmo ufficiale + omocodia), `luhn_ok`

---

## Redazione PDF (`src/app/pdf_export.py`)

### Due Modalità

| Input | Output | Metodo |
|-------|--------|--------|
| **PDF** | PDF anonimizzato | **Redazione vera**: `page.apply_redactions()` rimuove testo da content stream, scrive placeholder al posto |
| Testo / `.md` / `.txt` | PDF ricostruito | `text_to_pdf()`: impagina da zero il testo anonimizzato |

### Matching Char-Preciso con Confini di Parola

```python
# 1. Indice char-preciso pagina
text, boxes = _page_char_index(page)  # ogni char + bbox (rawdict)

# 2. Pattern valore con confini parola + sillabazione tollerata
_value_pattern(value)  # (?<!\w)...(?!\w) + _HYPHEN_BREAK tra caratteri

# 3. Match → bbox uniti per riga → add_redact_annot + apply_redactions()
```

**Eccezione confini parola**: se valore ha punteggiatura ai bordi (modello ha tagliato parola spezzata a fine riga, es. `"Fran-\ncesco Cordella"` → modello etichetta `"-\ncesco Cordella"`), si toglie l'ancora **solo se** il core residuo ha ≥ 4 caratteri alfanumerici (altrimenti `.it` matcherebbe "it" ovunque).

### Pulizia Completa (Oltre apply_redactions)

| Elemento | Metodo | Perché |
|----------|--------|--------|
| Metadati + XMP | `doc.set_metadata({...})` + `del_xml_metadata()` | Possono contenere PII (autore, titolo...) |
| Annotazioni | `page.annots()` → `a.set_info()` | Commenti, FreeText, note: testo invisibile a `get_text()` |
| Campi modulo (AcroForm) | `page.widgets()` → `w.field_value` | Valore sopravvive a redazioni content stream |
| Segnalibri (TOC) | `doc.get_toc()` → `doc.set_toc()` | Spesso ricalcano intestazioni con nomi/numeri |
| Allegati incorporati | `doc.embfile_del(name)` | Non ispezionabili affidabilmente |

### Verifica Residui (Safety Net)

```python
def _verify_residuals(pdf_bytes, items):
    # Stesso pattern della redazione (sillabazione inclusa)
    text = _readable_text(doc)  # pagine + annot + widget + TOC
    for ph, val in items:
        if _value_pattern(val).search(text):
            residual.append(ph)
    return residual  # deve essere []; UI avvisa se non vuoto
```

### Valori Saltati (Troppo Corti/Ambigui)

```python
def _too_noisy(value):
    alnum = re.sub(r"[\W_]+", "", _norm(value))
    return len(alnum) < 2 or (len(alnum) == 2 and alnum.isdigit())
# Es: "1", "C", "05" → non redatti, finiscono in report["skipped"], UI DEVE avvisare
```

### Caso Scansione (Nessun Testo Estrattibile)

Se `report["occurrences"] == 0` → **errore 422**, non PDF "anonimizzato" che non lo è.
Messaggio: *"Nessuna occorrenza trovata nel PDF: se il documento è una scansione (testo dentro un'immagine) la redazione del layer testuale non può agire."*

---

## App Desktop Tauri

### Architettura

```
┌────────────────────────────────────────────┐
│           Tauri App (Rust)                  │
│  ┌──────────────────────────────────────┐  │
│  │ WebView (UI HTML/JS servita da Flask) │  │
│  └──────────────────────────────────────┘  │
│           │                                 │
│    Sidecar (bundled)                        │
│  ┌──────────────────────────────────────┐  │
│  │ serve.py (Flask headless, CPU only)   │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

### Avvio

1. Tauri legge `config.json` (host/porta)
2. Lancia sidecar `serve.py` con env `PII_HOST`/`PII_PORT`
3. Attende server su porta configurata
4. Mostra UI (splash con badge UE/GDPR + versione)
5. Se sidecar esce con **codice 76** (porta occupata) → splash mostra form configurazione → salva `config.json` → rilancia sidecar

### Comandi Tauri Esposti allo Splash

- `save_config(host, port)` — salva `config.json`
- `retry_backend` — rilancia sidecar

### Build

```bash
cd tauri
npm install
npx tauri build
```

Output: installer NSIS per-utente (Windows), `.dmg` firmato/notarizzato (macOS), `.AppImage`/`.deb` (Linux). Dettagli in `docs/BUILD.md`.

---

## Testing

### Test Suite (`tests/`)

| Test | Descrizione |
|------|-------------|
| `test_pdf_hard_split.py` | Split righe lunghe in `text_to_pdf` |
| `test_slot_pipeline.py` | Pipeline generazione sintetica |
| `test_docid_e_date.py` | DOCID e DATE regex |
| `test_confini_parola.py` | Confini parola in redazione |
| `test_detector_formats.py` | Formati detector (IBAN, CF, etc.) |
| `test_pdf_fillable_text.py` | Estrazione testo PDF fillable (issue #85) |
| `test_ripristino_placeholder.py` | Round-trip placeholder |
| `test_cf_omocodia.py` | CF omocodia checksum |
| `test_carta_scadenza.py` | Carta con scadenza attaccata |
| `test_pdf_protetto.py` | Gestione PDF protetti |
| `test_iban_formato_stampa.py` | IBAN formato stampa a gruppi |
| `test_log_backend.py` | Logging backend |
| `test_targa_generators.py` | Generatori targa |
| `test_tsv_anonymize.py` | Anonimizzazione TSV |
| `test_backend_llm_timeout.py` | Timeout LLM backend |
| `test_ora.py` | Completamento ore |
| `test_source_text.py` | Gestione source_text |
| `test_server_config.py` | Config server/prefs |
| `test_codifiche.py` | Gestione encoding (utf-16, cp1252, latin-1) |

### Esecuzione Test

```bash
# Tutti i test
python -m pytest tests/ -v

# Singolo test
python -m pytest tests/test_pdf_export.py -v

# Con coverage
python -m pytest tests/ --cov=src/app --cov=src/detectors
```

### Test Manuale Rapido

```bash
# CLI inference
PII_MODEL_DIR=models/rizzo-pii-0.3B-v1.5.0 \
  python src/training/test_pii.py "Mi chiamo Mario Rossi, IBAN IT60X0542811101000000123456"

# Server web
python src/app/app.py  # http://127.0.0.1:5005

# API test
curl localhost:5005/health
curl -X POST localhost:5005/analyze -H 'Content-Type: application/json' \
     -d '{"text": "Mario Rossi, CF RSSMRA85M01H501Q"}'
curl -F "file=@atto.pdf" localhost:5005/analyze
curl -F "file=@atto.pdf" localhost:5005/pdf -o atto_anonimizzato.pdf
```

---

## Build & Packaging

### Docker (CPU Only)

```bash
docker build -t rizzo-pii .                    # ~10 min, 2.65 GB image
docker run -d --name rizzo-pii -p 127.0.0.1:5005:5005 rizzo-pii
docker logs -f rizzo-pii
docker rm -f rizzo-pii
```

- Immagine **self-contained**: CPU PyTorch + modello baked in
- `docker run` **mai rete** (`HF_HUB_OFFLINE=1`)
- Gunicorn 1 worker + 4 thread (modello caricato una volta ~1.2 GB)
- `/health` risponde 503 finché modello non pronto → `HEALTHCHECK` polla

### Desktop (Tauri - Raccomandato)

```bash
cd tauri
npm install
npx tauri build
```

Vedi `docs/BUILD.md` per dettagli completi (firme, notarizzazione, AppImage, .deb).

### Legacy PyInstaller

```bash
# Sidecar per Tauri
pyinstaller build_sidecar.spec

# App standalone legacy
pyinstaller build.spec
```

---

## Estrazione Zone PII con Coordinate (Nuova Feature)

### Obiettivo

Permettere l'estrazione delle **zone del documento contenenti dati sensibili** tramite **coordinate geometriche** (bounding box), da dare in pasto a un motore OCR esterno per documenti scansionati (immagini raster senza layer testuale).

### Caso d'Uso

1. Utente carica PDF scansione (solo immagini, niente testo estraibile)
2. `/pdf` attualmente fallisce con 422 ("nessuna occorrenza trovata")
3. **Nuova feature**: endpoint che ritorna le coordinate delle zone PII rilevate nel layer testuale (se presente) **oppure** che permette di specificare coordinate manuali per far fare OCR solo su quelle zone

### Implementazione Richiesta

#### 1. Nuovo Endpoint: `/pii-zones`

```python
@app.route("/pii-zones", methods=["POST"])
def pii_zones_route():
    """
    Estrae le zone (bounding box) delle PII in un PDF.
    
    Input: multipart con file .pdf
    Output: JSON con lista zone per pagina:
    {
        "zones": [
            {
                "page": 0,
                "label": "FULLNAME",
                "placeholder": "[FULLNAME_1]",
                "value": "Mario Rossi",          # solo se mapping_enabled
                "bbox": [x0, y0, x1, y1],        # punti PDF (1/72 inch)
                "bbox_normalized": [0.1, 0.2, 0.3, 0.25],  # 0-1 rispetto pagina
                "confidence": 0.98
            },
            ...
        ],
        "page_sizes": [{"width": 595.28, "height": 841.89}, ...],  # punti PDF
        "dpi": 110
    }
    """
```

#### 2. Logica di Estrazione Coordinate

Riutilizzare `_page_char_index()` da `pdf_export.py` che già produce:
- `text`: stringa completa pagina
- `boxes`: lista `fitz.Rect` per carattere (None per newline)

```python
def extract_pii_zones(pdf_bytes, mapping, excluded_tags=None):
    """
    Ritorna lista zone PII con coordinate.
    Riutilizza la logica di matching char-preciso di pdf_export.redact_pdf.
    """
    excluded = set(excluded_tags or ())
    items = [(ph, v) for ph, v in mapping.items() 
             if ph not in excluded and isinstance(v, str) and v.strip()]
    
    zones = []
    page_sizes = []
    
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            page_sizes.append({"width": page.rect.width, "height": page.rect.height})
            text, boxes = _page_char_index(page)
            
            for ph, val in items:
                if _too_noisy(val):
                    continue
                pat = _value_pattern(val)
                if not pat:
                    continue
                    
                for m in pat.finditer(text):
                    rects = _match_rects(boxes, m)
                    for r in rects:
                        # Normalizza bbox 0-1
                        norm = [
                            r.x0 / page.rect.width,
                            r.y0 / page.rect.height,
                            r.x1 / page.rect.width,
                            r.y1 / page.rect.height
                        ]
                        zones.append({
                            "page": page_num,
                            "label": ph.split("_")[0].strip("[]"),
                            "placeholder": ph,
                            "value": val if mapping_enabled else None,
                            "bbox": [r.x0, r.y0, r.x1, r.y1],
                            "bbox_normalized": norm,
                            "confidence": 1.0 if validated else 0.9
                        })
    
    return {"zones": zones, "page_sizes": page_sizes, "dpi": PREVIEW_DPI}
```

#### 3. Endpoint OCR-Assisted: `/ocr-zones` (Implementato)

Per documenti scansionati, permette di inviare risultati OCR per zone specifiche e integrarli nell'anonimizzazione.

**Input JSON:**
```json
{
  "pdf_base64": "...",           // PDF originale (base64) OPPURE
  "doc_id": "...",               // documento già in memoria (da /preview)
  "zones": [
    {"page": 0, "bbox": [x0,y0,x1,y1], "ocr_text": "Mario Rossi"}
  ],
  "exclude_tags": [...],
  "include_mapping": true
}
```

**Flusso:**
1. Carica PDF (da base64 o da `doc_id` in memoria)
2. Analizza layer testuale esistente
3. Per ogni zona OCR: esegue `detect_model` + `detect_regex` su `ocr_text`
4. Restituisce entità OCR con metadati zona (`ocr_zone`: page, bbox, source_text)
5. **Nota**: le entità OCR hanno offset **relativi al testo OCR della zona**. Per fusione completa con entità layer testuale serve allineamento coordinate (lavoro futuro).

**Output:**
```json
{
  "base_analysis": { ... },      // risultato /analyze standard
  "ocr_entities": [ ... ],       // entità trovate nel testo OCR
  "ocr_zones_processed": 3,
  "note": "Le entità OCR hanno offset relativi al testo OCR della zona..."
}
```

#### 4. Endpoint Combinato: `/analyze-with-zones` (Implementato)

Combina `/analyze` + `/pii-zones` in una sola chiamata per evitare doppio processing del PDF.

**Input:** identico a `/pii-zones` (multipart PDF)  
**Output:** tutti i campi di `/analyze` **PIU'** `zones`, `page_sizes`, `dpi`

#### 5. Workflow Consigliato per Documenti Scansionati

```
1. POST /preview           → carica PDF, ottieni doc_id, n_pages, testo estratto
2. GET /doc/<id>/page/<n>.png  → renderizza pagine per visualizzazione
3. POST /pii-zones         → ottieni coordinate zone PII nel layer testuale (se presente)
4. [Client-side] Invia zone a motore OCR esterno (es. Tesseract, AWS Textract, Google Vision)
5. POST /ocr-zones         → invia risultati OCR (bbox + testo) per integrazione
6. POST /analyze-with-zones → analisi completa combinata (opzionale)
7. POST /pdf               → scarica PDF anonimizzato finale
```

### Limitazioni Note

1. **Solo layer testuale**: `/pii-zones` trova zone solo dove c'è testo estraibile. Per scansioni pure (immagini), il layer testuale è vuoto → `zones: []`.
2. **OCR fusion parziale**: `/ocr-zones` restituisce entità OCR con offset locali. Mappatura a offset globali richiede allineamento coordinate PDF ↔ testo OCR (lavoro futuro).
3. **Coordinate PDF**: Sistema di riferimento PyMuPDF (punti, origine bottom-left). Per conversione pixel immagine: `pixel = point * (DPI / 72)`.

### Estensioni Future

- [ ] Allineamento automatico coordinate OCR → offset globali testo
- [ ] Endpoint `/redact-zones` per redarre solo zone specifiche (bbox)
- [ ] Supporto coordinate in pixel (oltre che punti PDF normalizzati)
- [ ] Integrazione diretta con motori OCR popolari (Tesseract, PaddleOCR)

---

## Convenzioni di Codice

### Stile Python

- **Type hints** dove utile (non obbligatorio ovunque)
- **Docstring** in italiano per funzioni pubbliche/moduli
- **Naming**: `snake_case` funzioni/variabili, `PascalCase` classi, `UPPER_SNAKE` costanti
- **Import ordinati**: stdlib → third-party → local
- **Path risolti da `__file__`**: `Path(__file__).resolve().parents[N] / "..."`
- **UTF-8 forzato**: `sys.stdout.reconfigure(encoding="utf-8")` all'avvio script

### Gestione Errori

```python
# Eccezioni custom per errori controllati
class PdfError(ValueError): pass
class PdfLocked(ValueError): pass
class _ReqError(Exception):
    def __init__(self, msg, status=400):
        self.msg, self.status = msg, status

# In route: try/except → jsonify({"error": str(e)}), status
```

### Thread Safety

- `_DOCS` (LRU documenti anteprima) → `threading.Lock()`
- `EXCLUDED_TAGS`, `MAPPING_ENABLED` → `_PREFS_LOCK` per snapshot atomico
- `server_config.save_prefs()` / `load_prefs()` usano file locking

### Logging

- `serve.py` (Tauri backend): log su `%LOCALAPPDATA%\rizzo-pii\backend.log`
- `app.py` (standalone): stdout/stderr
- Nessun logging strutturato esterno (no structlog, ecc.)

---

## Troubleshooting Comune

| Problema | Causa | Soluzione |
|----------|-------|-----------|
| `torch.cuda.is_available()` False su RTX 50xx | torch non cu128 | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `operator torchvision::nms does not exist` | torchvision/torchaudio vecchie | `pip uninstall torchvision torchaudio` |
| `RuntimeError ... bootstrapping` (Windows) | `dataloader_num_workers>0` | Impostare `0` in Trainer |
| Modello non trovato all'avvio | `MODEL_DIR` sbagliato | Controlla `APP_MODEL_VERSION` e `models/registry.json` |
| Porta occupata all'avvio Tauri | Altra istanza gira | Tauri mostra form configurazione (codice uscita 76) |
| PDF protetto da password | `doc.needs_pass` | UI segnala errore "PDF protetto: rimuovi protezione" |
| Nessuna occorrenza in PDF scansione | Layer testuale vuoto | Errore 422: serve OCR → nuova feature `/pii-zones` |
| Valori residui in output | Matching fallito (sillabazione, encoding) | Controlla `X-PII-Residual` header; pattern `_value_pattern` |
| "DE" redatto dentro "CORDELLA" | Confini parola non rispettati | `_value_pattern` usa `(?<!\w)...(?!\w)` |
| IBAN troncato / coda in chiaro | Regex non copre formato stampa | Usa `detect_iban()` scanner (non regex) con `IBAN_LEN` |
| Carta credito con scadenza "12/26" in chiaro | Luhn fallisce su numero+scadenza | `_card_senza_scadenza()` riprova tagliando coda |

---

## Aggiungere Nuovi Tag PII

1. **Aggiungi a `TAG_MAP` in `train_pii.py`** (rimappa tag grezzi → 22 tag finali)
2. **Aggiorna `TAGS` in `app.py`** (legenda UI: tag, descrizione IT/EN, esempio)
3. **Se strutturato (checksum)**: aggiungi validatore in `detectors.py` + entry in `DETECTORS`
4. **Se solo regex**: aggiungi entry in `DETECTORS` con `validator=None`
5. **Se serve priorità regex**: NON metterlo in `SOFT_REGEX_LABELS`
6. **Test**: `python -m pytest tests/test_detector_formats.py -v`
7. **Rigenera dati** se tag nuovo richiede dati sintetici (`generate_synthetic_pii.py`)
8. **Riaddestra**: `python src/training/train_pii.py --type full --version X.Y.Z`

---

## Contribuire

Vedi `CONTRIBUTING.md` per:
- Code style
- Commit message convention
- Come contribuire dati (script `contribute_dataset.py`)
- Processo PR

---

## Licenza

- **Codice sorgente**: MIT License
- **Binari rilasciati**: AGPL-3.0 (bundlano PyMuPDF)
- Vedi `LICENSE` e `THIRD_PARTY_LICENSES.md`

---

*Manuale generato per rizzo-pii v2.0.0 — Aggiornare ad ogni cambio architetturale significativo.*