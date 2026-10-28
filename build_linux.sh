#!/usr/bin/env bash
# ============================================================================
# Build dell'app desktop Rizzo PII per LINUX (.deb/.rpm/.AppImage), CPU/offline.
#
# NON si compila da Windows: PyInstaller e i bundle Tauri Linux (webkit2gtk)
# vanno costruiti SU Linux. Lancia questo script su Ubuntu/Debian/Fedora/openSUSE o in WSL2.
#
# Speculare a docs/BUILD.md (Windows): stesso build_sidecar.spec, ma qui il
# sidecar esce come 'pii-backend' (senza .exe) e i bundle sono deb/rpm/appimage.
# Il Rust (lib.rs) sceglie il nome del binario in base al SO (cfg!(windows)).
#
# Uso:
#   bash build_linux.sh          # tutti e tre: deb, rpm, appimage
#   bash build_linux.sh deb      # solo .deb
#   bash build_linux.sh rpm      # solo .rpm
#   bash build_linux.sh appimage # solo .AppImage
#
# Prerequisiti di sistema (una volta):
#   Ubuntu/Debian:
#     sudo apt update && sudo apt install -y \
#       build-essential curl wget file libssl-dev libxdo-dev patchelf \
#       libwebkit2gtk-4.1-dev librsvg2-dev libayatana-appindicator3-dev rpm
#   Fedora/RHEL:
#     sudo dnf install -y gcc gcc-c++ make curl wget file openssl-devel patchelf \
#       webkit2gtk4.1-devel librsvg2-devel libappindicator-gtk3-devel rpm-build
#   openSUSE:
#     sudo zypper install -y gcc gcc-c++ make curl wget file libopenssl-devel patchelf \
#       webkit2gtk-4_1-devel librsvg-devel libappindicator3-devel rpm-build
#   # + Rust (https://rustup.rs) e Node.js 18+ (con npm)
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_DIR="models/rizzo-pii-0.3B-v1.5.0"   # deve combaciare con build_sidecar.spec
VENV="${VENV:-build_env_linux}"            # override: in Docker si usa il venv già nell'immagine
BUNDLES="${*:-deb rpm appimage}"           # bundle da produrre (es. "bash build_linux.sh deb")

# ---- 0) controlli ----------------------------------------------------------
[ -d "$MODEL_DIR" ] || { echo "ERRORE: modello mancante: $MODEL_DIR (copialo qui)"; exit 1; }
command -v cargo >/dev/null || { echo "ERRORE: Rust/cargo non trovato (installa da https://rustup.rs)"; exit 1; }
command -v npm   >/dev/null || { echo "ERRORE: npm non trovato (installa Node.js 18+)"; exit 1; }

# La CLI Tauri < 2.12 usa rpm-rs 0.16, che carica TUTTO il payload in RAM:
# con ~2,2 GB di risorse (sidecar PyInstaller + torch + modello) il bundle .rpm
# impazzisce (swap/thrashing) e non finisce mai -> niente .rpm e niente .AppImage.
# Il controllo definitivo gira dopo `npm install` (vedi passo 3).
check_tauri_cli() {
  local ver
  ver="$(node -p "require('./node_modules/@tauri-apps/cli/package.json').version" 2>/dev/null || echo 0)"
  if [ "$ver" = "0" ]; then
    echo "ERRORE: @tauri-apps/cli non installato in tauri/ (esegui: cd tauri && npm install)"; exit 1
  fi
  if [ "$(printf '%s\n2.12.0\n' "$ver" | sort -V | head -1)" != "2.12.0" ]; then
    echo "ERRORE: @tauri-apps/cli $ver troppo vecchia per il bundle .rpm."
    echo "       Serve >= 2.12 (rpm-rs 0.30, streaming): cd tauri && npm install -D @tauri-apps/cli@^2.12.4"
    exit 1
  fi
}

# ---- 1) venv CPU + dipendenze (come docs/BUILD.md, ma su Linux) ------------
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch
  "$VENV/bin/pip" install "transformers==4.57.3" tokenizers safetensors flask pymupdf pyinstaller
  # OCR opzionale per i PDF scansionati: se si installa, il pacchetto finisce
  # dentro il sidecar (build_sidecar.spec lo raccoglie solo se e' presente).
  # Se fallisce la build prosegue: l'app funziona, ma sui documenti scansionati
  # avvisa che serve un motore OCR.
  "$VENV/bin/pip" install rapidocr-onnxruntime \
    || echo "AVVERTENZA: OCR non impacchettato (rapidocr-onnxruntime non installato)"
fi

# ---- 2) sidecar PyInstaller -> tauri/src-tauri/backend/pii-backend/pii-backend ----
"$VENV/bin/pyinstaller" build_sidecar.spec --noconfirm \
  --distpath tauri/src-tauri/backend --workpath build/sidecar_work_linux
[ -f tauri/src-tauri/backend/pii-backend/pii-backend ] \
  || { echo "ERRORE: sidecar Linux non prodotto"; exit 1; }

# ---- 3) build Tauri: override bundle a deb + rpm + appimage (la conf di default è nsis) ----
cd tauri
npm install
check_tauri_cli
npx tauri build --bundles $BUNDLES

echo
echo "FATTO. Artefatti Linux in:"
for b in $BUNDLES; do
  case "$b" in
    deb)      echo "  tauri/src-tauri/target/release/bundle/deb/*.deb" ;;
    rpm)      echo "  tauri/src-tauri/target/release/bundle/rpm/*.rpm" ;;
    appimage) echo "  tauri/src-tauri/target/release/bundle/appimage/*.AppImage" ;;
  esac
done