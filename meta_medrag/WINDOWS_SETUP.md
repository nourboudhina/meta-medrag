# Meta-MedRAG — Windows Setup Guide

## The error you saw

```
ConnectionResetError: [WinError 10054]
```

This is **not a code problem**. pip was downloading a large package (~500 MB) and your internet connection dropped mid-download. Nothing is broken. The fix is simply to install packages in small batches with retry logic.

---

## Step 1 — Fix your Python version (IMPORTANT)

You are running **Python 3.13**. This version is too new for PyTorch and most ML libraries. You need Python **3.10 or 3.11**.

### Install Python 3.11

1. Download from https://www.python.org/downloads/release/python-3119/
2. Run the installer — **check "Add Python to PATH"**
3. Open a NEW terminal and verify:

```cmd
python --version
# Should print: Python 3.11.x
```

### Create a virtual environment (isolates your project)

```cmd
cd C:\Users\nourb\Downloads\meta_medrag

python -m venv venv
venv\Scripts\activate

# Your prompt should now show: (venv) C:\Users\nourb\Downloads\meta_medrag>
```

**Always activate the venv before working on the project:**
```cmd
venv\Scripts\activate
```

---

## Step 2 — Install dependencies (with retry logic)

Copy the `setup_windows.py` file into your project folder, then run:

```cmd
python setup_windows.py
```

This installs packages in 13 small groups with automatic retry on connection drop.

**If any group fails**, install it manually with:
```cmd
pip install --timeout 120 --retries 5 <package_name>
```

---

## Step 3 — Install PyTorch with GPU support (if you have NVIDIA GPU)

The setup script installs the CPU version of PyTorch (safe for all machines).
If you have an NVIDIA GPU, replace it with the CUDA version:

```cmd
# First uninstall CPU version
pip uninstall torch torchvision

# Install CUDA 12.1 version (for RTX 3000/4000 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

> **Note:** For training LLaVA-Med (7B), you need at least 24 GB VRAM.
> For inference only, 16 GB is sufficient with 4-bit quantisation.
> The Toulouse team's A100 GPUs (80 GB) handle everything.

---

## Step 4 — Prepare datasets

```cmd
# Create synthetic test data immediately (no download needed)
python scripts/prepare_datasets.py --dataset samples

# Download IU X-ray (free, ~1 GB, for real experiments)
python scripts/prepare_datasets.py --dataset iu_xray

# See MIMIC-CXR access instructions
python scripts/prepare_datasets.py --dataset mimic_cxr

# Check what is ready
python scripts/prepare_datasets.py --dataset verify
```

---

## Step 5 — Set environment variables

Create a file named `.env` in your project root:

```
OPENAI_API_KEY=sk-...your-key-here...
HF_TOKEN=hf_...your-token-here...
```

Then load it:
```cmd
pip install python-dotenv
```

Or set them directly in the terminal (for the current session):
```cmd
set OPENAI_API_KEY=sk-...
set HF_TOKEN=hf_...
```

- **OpenAI key**: https://platform.openai.com/api-keys  (needed for preference pair generation in Module 3)
- **HuggingFace token**: https://huggingface.co/settings/tokens  (needed to download LLaVA-Med)

---

## Step 6 — Run tests to verify everything works

```cmd
pytest tests/unit/ -v
```

All 3 test files should pass without a GPU.

---

## Step 7 — Launch the demo interface

```cmd
# Test the interface without loading the model (instant)
python -m src.interface.app --no-pipeline

# Full interface (requires model to be downloaded)
python -m src.interface.app
```

Open http://localhost:7860 in your browser.

---

## Quick reference — common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ConnectionResetError: [WinError 10054]` | Network dropped during pip download | Run `setup_windows.py` — it retries automatically |
| `ModuleNotFoundError: torch` | PyTorch not installed | `pip install torch --timeout 120` |
| `Python 3.13` compatibility errors | Wrong Python version | Install Python 3.11, create new venv |
| `CUDA not available` | CPU-only PyTorch | See Step 3 above |
| `Access denied` when running pip | Windows permissions | Right-click terminal → "Run as administrator" |
| `pip` timeout errors | Slow connection | `pip install --timeout 300 <package>` |
| `bitsandbytes` Windows error | Windows-specific build | `pip install bitsandbytes --prefer-binary` |

---

## Training workflow (for reference)

Once setup is complete, the full training flow is:

```cmd
# 1. Build contrastive pairs for probe training
python scripts/train_probe.py --step build_pairs

# 2. Extract activations (needs GPU — run on Toulouse A100)
python scripts/train_probe.py --step extract

# 3. Train the probe (CPU is fine, fast)
python scripts/train_probe.py --step train

# 4. Build FAISS vector stores
python scripts/build_vector_stores.py --domain radiology

# 5. Generate preference pairs (costs ~$5 OpenAI credits)
python scripts/run_dpo_training.py --step generate

# 6. DPO training (needs GPU — run on Toulouse A100)
python scripts/run_dpo_training.py --step train

# 7. Evaluate
python -m src.evaluation.run_evaluation --dataset iu_xray --task vqa
```

---

## File structure after setup

```
meta_medrag/
├── venv/                    ← virtual environment (do not commit)
├── .env                     ← API keys (do not commit)
├── data/
│   ├── raw/
│   │   ├── iu_xray/
│   │   │   ├── images/      ← X-ray images (.png)
│   │   │   ├── reports/     ← text reports (.txt)
│   │   │   └── splits.json  ← train/val/test split
│   │   └── mimic_cxr/       ← (same structure, after MIMIC approval)
│   ├── processed/           ← activations.pkl (after probe extraction)
│   └── vector_stores/       ← radiology.index etc. (after build)
├── checkpoints/             ← trained model weights
└── experiments/
    └── results/             ← evaluation results JSON files
```
