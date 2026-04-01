"""
setup_windows.py
================
Run this script to install ALL dependencies for Meta-MedRAG on Windows.
It handles the connection reset problem by installing packages in small
groups with automatic retry on failure.

Usage (run from inside your meta_medrag folder):
    python setup_windows.py

Requirements:
    - Python 3.10 or 3.11  (NOT 3.13 — see note below)
    - pip up to date
    - Stable internet connection
"""

import subprocess
import sys
import time
import os

# ─────────────────────────────────────────────────────────────────────────────
# PYTHON VERSION CHECK
# Python 3.13 has compatibility issues with PyTorch and several ML libraries.
# Use Python 3.10 or 3.11 for this project.
# ─────────────────────────────────────────────────────────────────────────────
major, minor = sys.version_info.major, sys.version_info.minor
print(f"Python version: {major}.{minor}")

if major == 3 and minor >= 13:
    print(
        "\n⚠️  WARNING: You are using Python 3.13.\n"
        "   PyTorch and many ML libraries do NOT fully support Python 3.13 yet.\n"
        "   Strongly recommended: install Python 3.10 or 3.11 from python.org\n"
        "   and create a virtual environment with that version.\n"
        "   See README instructions below.\n"
    )
    answer = input("Continue anyway? (y/N): ").strip().lower()
    if answer != "y":
        sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# PACKAGE GROUPS — installed in small batches to avoid download timeouts
# Each group is independent; if one fails, the others still install.
# ─────────────────────────────────────────────────────────────────────────────

PACKAGE_GROUPS = {

    "1. Core scientific stack": [
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "pandas>=2.2.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ],

    "2. Machine learning": [
        "scikit-learn>=1.4.0",
    ],

    "3. PyTorch (CPU version for setup — replace with GPU below)": [
        # CPU version first — safe for all machines.
        # For CUDA 12.1 GPU: replace with:
        #   torch torchvision --index-url https://download.pytorch.org/whl/cu121
        "torch",
        "torchvision",
    ],

    "4. HuggingFace ecosystem": [
        "transformers>=4.40.0",
        "tokenizers>=0.19.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "huggingface_hub>=0.22.0",
    ],

    "5. Parameter-efficient fine-tuning": [
        "peft>=0.10.0",
        "trl>=0.8.6",
    ],

    "6. Quantisation (for loading LLaVA-Med on consumer GPU)": [
        "bitsandbytes>=0.43.0",
    ],

    "7. Image processing": [
        "Pillow>=10.2.0",
        "einops>=0.7.0",
        "sentencepiece>=0.1.99",
    ],

    "8. FAISS (vector search) — CPU version": [
        # CPU version works on all machines.
        # On a GPU server: use   faiss-gpu   instead.
        "faiss-cpu>=1.7.4",
    ],

    "9. BiomedCLIP (open_clip)": [
        "open-clip-torch>=2.24.0",
    ],

    "10. Evaluation metrics": [
        "nltk>=3.8.1",
        "rouge-score>=0.1.2",
        "sacrebleu>=2.4.0",
        "evaluate>=0.4.1",
    ],

    "11. Gradio interface": [
        "gradio>=4.26.0",
    ],

    "12. OpenAI (for preference pair generation)": [
        "openai>=1.20.0",
    ],

    "13. Utilities": [
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.2",
        "rich>=13.7.0",
        "tqdm>=4.66.0",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# INSTALLER WITH RETRY LOGIC
# Retries up to 3 times on connection errors with increasing wait times.
# ─────────────────────────────────────────────────────────────────────────────

def install_group(group_name: str, packages: list, max_retries: int = 3):
    print(f"\n{'='*60}")
    print(f"Installing: {group_name}")
    print(f"{'='*60}")

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}...")

        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--timeout", "120",          # 2 min timeout per package
                "--retries", "5",            # pip internal retries
                "--progress-bar", "off",     # cleaner output
            ] + packages,
            capture_output=False,
        )

        if result.returncode == 0:
            print(f"✓ {group_name} installed successfully")
            return True
        else:
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"  Failed — waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"✗ {group_name} FAILED after {max_retries} attempts")
                print("  → Skip and continue. Install manually later with:")
                print(f"    pip install {' '.join(packages)}")
                return False


def main():
    print("\n" + "="*60)
    print("Meta-MedRAG — Windows Dependency Installer")
    print("="*60)

    # Update pip first (silently)
    print("\nUpdating pip...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True
    )

    failed = []
    for group_name, packages in PACKAGE_GROUPS.items():
        success = install_group(group_name, packages)
        if not success:
            failed.append(group_name)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)

    total  = len(PACKAGE_GROUPS)
    passed = total - len(failed)
    print(f"Installed: {passed}/{total} groups")

    if failed:
        print("\nFailed groups (install manually):")
        for f in failed:
            print(f"  - {f}")
    else:
        print("\n✓ All packages installed successfully!")

    # ── Post-install checks ───────────────────────────────────────────
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    checks = {
        "torch":        "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())",
        "transformers": "import transformers; print('Transformers:', transformers.__version__)",
        "sklearn":      "import sklearn; print('scikit-learn:', sklearn.__version__)",
        "faiss":        "import faiss; print('FAISS: OK, n_gpus =', faiss.get_num_gpus())",
        "gradio":       "import gradio; print('Gradio:', gradio.__version__)",
        "peft":         "import peft; print('PEFT:', peft.__version__)",
    }

    for pkg, code in checks.items():
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✓ {result.stdout.strip()}")
        else:
            print(f"  ✗ {pkg}: NOT available")


if __name__ == "__main__":
    main()
