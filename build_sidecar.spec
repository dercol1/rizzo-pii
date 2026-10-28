# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec per il BACKEND SIDECAR dell'app Tauri (build CPU, windowed).
# Entry: src/app/serve.py (solo server Flask, nessun browser).
# Build:  build_env\Scripts\pyinstaller.exe build_sidecar.spec --noconfirm \
#           --distpath tauri/src-tauri/backend
# Output: tauri/src-tauri/backend/pii-backend/pii-backend.exe (+ _internal, modello, assets)
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ("transformers", "tokenizers", "safetensors", "huggingface_hub", "regex"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# modello + asset grafici impacchettati (app.py li risolve via _resource_path / _MEIPASS)
datas += [("models/rizzo-pii-0.3B-v1.5.0", "pii_model")]
datas += [("src/app/assets", "assets")]
hiddenimports += ["fitz", "flask", "sklearn.utils._typedefs"]

# OCR opzionale (serve solo per i PDF SCANNIONATI, senza layer testuale).
# Se rapidocr-onnxruntime e' installato nell'env di build viene impacchettato
# (modelli ONNX ~13 MB + onnxruntime ~68 MB). Se non e' installato la build
# resta identica a prima: l'app funziona e, sui documenti scansionati, dice
# che serve un motore OCR. Non e' un errore: e' una scelta.
try:
    import importlib.util as _ilu

    if _ilu.find_spec("rapidocr_onnxruntime") is not None:
        for pkg in ("rapidocr_onnxruntime", "onnxruntime"):
            d, b, h = collect_all(pkg)
            datas += d
            binaries += b
            hiddenimports += h
        print("[build] OCR: rapidocr-onnxruntime impacchettato nel sidecar")
except Exception as _e:
    print(f"[build] OCR non impacchettato ({_e})")

excludes = [
    "tensorflow", "tensorflow_intel", "tf_keras", "keras", "jax", "jaxlib", "flax",
    "vllm", "FlagEmbedding", "flagembedding", "torchvision", "torchaudio",
    "matplotlib", "pandas", "scipy", "IPython", "notebook", "PyQt5", "PySide2",
]

a = Analysis(
    ["src/app/serve.py"],
    pathex=["src/app"],          # cosi' PyInstaller trova il modulo 'app' importato da serve.py
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pii-backend",
    console=False,              # windowed: nessuna finestra console (i log vanno su file)
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="pii-backend",
)
