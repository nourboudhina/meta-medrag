"""
scripts/prepare_datasets.py
============================
Downloads, extracts, and organises all datasets needed for Meta-MedRAG.

Datasets:
    1. IU X-ray       — Free, no account needed (~1 GB)
    2. MIMIC-CXR      — Requires PhysioNet account (guide included)
    3. Sample data    — Small test samples generated locally

Usage:
    python scripts/prepare_datasets.py --dataset iu_xray
    python scripts/prepare_datasets.py --dataset samples
    python scripts/prepare_datasets.py --dataset all
"""

import os
import json
import shutil
import argparse
import zipfile
import tarfile
import urllib.request
import urllib.error
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT = Path("data/raw")

DATASET_INFO = {
    "iu_xray": {
        "description": "Indiana University Chest X-Ray Collection (~1 GB)",
        "reports_url": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz",
        "images_url":  "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
        "target_dir":  DATA_ROOT / "iu_xray",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD HELPER WITH PROGRESS BAR
# ─────────────────────────────────────────────────────────────────────────────

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return True

    print(f"  Downloading {desc or dest.name}...")
    print(f"  URL: {url}")

    try:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=dest.name
        ) as t:
            urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
        print(f"  ✓ Downloaded: {dest}")
        return True

    except urllib.error.URLError as e:
        print(f"  ✗ Download failed: {e}")
        print("  Try manually downloading from the URL above and placing in:")
        print(f"  {dest}")
        return False

    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


def extract_archive(archive_path: Path, target_dir: Path):
    """Extract .tgz, .tar.gz, or .zip archive."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive_path.name} → {target_dir}")

    if str(archive_path).endswith((".tgz", ".tar.gz")):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(target_dir)
    elif str(archive_path).endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(target_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path.suffix}")

    print(f"  ✓ Extracted to {target_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# IU X-RAY DATASET
# ─────────────────────────────────────────────────────────────────────────────

def prepare_iu_xray():
    """Download and prepare the IU X-ray dataset."""
    print("\n" + "="*60)
    print("Dataset: Indiana University Chest X-Ray (IU X-ray)")
    print("="*60)
    print("Free dataset — no account required")
    print("Size: ~1 GB images + ~10 MB reports\n")

    info       = DATASET_INFO["iu_xray"]
    target_dir = info["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download reports
    reports_archive = target_dir / "NLMCXR_reports.tgz"
    ok1 = download_file(info["reports_url"], reports_archive, "IU X-ray reports")

    # Download images
    images_archive = target_dir / "NLMCXR_png.tgz"
    ok2 = download_file(info["images_url"], images_archive, "IU X-ray images")

    # Extract
    if ok1 and reports_archive.exists():
        extract_archive(reports_archive, target_dir / "reports_raw")

    if ok2 and images_archive.exists():
        extract_archive(images_archive, target_dir / "images")

    # Parse XML reports into plain text
    if (target_dir / "reports_raw").exists():
        print("\n  Parsing XML reports into plain text...")
        _parse_iu_xray_reports(
            xml_dir=target_dir / "reports_raw",
            out_dir=target_dir / "reports",
        )

    # Create split file
    _create_iu_xray_splits(target_dir)

    print("\n✓ IU X-ray dataset ready")
    print(f"  Images:  {target_dir}/images/")
    print(f"  Reports: {target_dir}/reports/")
    print(f"  Splits:  {target_dir}/splits.json")


def _parse_iu_xray_reports(xml_dir: Path, out_dir: Path):
    """Parse IU X-ray XML reports into plain text files."""
    import xml.etree.ElementTree as ET

    out_dir.mkdir(parents=True, exist_ok=True)
    xml_files = list(xml_dir.glob("**/*.xml"))
    print(f"  Found {len(xml_files)} XML report files")

    parsed = 0
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Extract text sections from IU X-ray XML format
            sections = {}
            for abstract in root.findall(".//AbstractText"):
                label = abstract.get("Label", "").upper()
                text  = (abstract.text or "").strip()
                if text:
                    sections[label] = text

            if not sections:
                continue

            # Build plain text report
            report_lines = []
            for section in ["FINDINGS", "IMPRESSION", "INDICATION", "COMPARISON"]:
                if section in sections:
                    report_lines.append(f"{section}: {sections[section]}")

            if report_lines:
                out_path = out_dir / (xml_file.stem + ".txt")
                out_path.write_text("\n".join(report_lines))
                parsed += 1

        except Exception as e:
            pass  # Skip malformed XML files

    print(f"  ✓ Parsed {parsed} reports")


def _create_iu_xray_splits(dataset_dir: Path):
    """Create train/val/test split file for IU X-ray."""
    report_dir = dataset_dir / "reports"
    image_dir  = dataset_dir / "images"

    if not report_dir.exists():
        print("  ✗ Reports directory not found — skipping split creation")
        return

    report_files = sorted(report_dir.glob("*.txt"))
    print(f"  Creating splits for {len(report_files)} reports...")

    # Find matching images
    items = []
    for rf in report_files:
        # IU X-ray images are named XXXXXX_*.png where XXXXXX = report ID
        stem = rf.stem
        img_matches = list(image_dir.glob(f"**/{stem}*.png"))

        if img_matches:
            with open(rf) as f:
                text = f.read().strip()

            items.append({
                "doc_id":     stem,
                "image":      str(img_matches[0].relative_to(".")),
                "text":       text,
                "question":   "Describe the key findings in this chest X-ray.",
                "answer":     _extract_impression(text),
                "report":     text,
                "domain":     "radiology",
                "split":      "",  # assigned below
            })

    # 70% train, 15% val, 15% test
    n = len(items)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    for i, item in enumerate(items):
        if i < n_train:
            item["split"] = "train"
        elif i < n_train + n_val:
            item["split"] = "val"
        else:
            item["split"] = "test"

    split_path = dataset_dir / "splits.json"
    with open(split_path, "w") as f:
        json.dump(items, f, indent=2)

    counts = {s: sum(1 for x in items if x["split"] == s)
              for s in ["train", "val", "test"]}
    print(f"  ✓ Split created: {counts}")


def _extract_impression(report_text: str) -> str:
    """Extract the IMPRESSION section as the VQA answer."""
    for line in report_text.split("\n"):
        if line.startswith("IMPRESSION:"):
            return line.replace("IMPRESSION:", "").strip()
    return report_text[:100]


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE DATA (for immediate testing without downloading)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_samples():
    """
    Create synthetic sample data so you can test the pipeline immediately
    without downloading any datasets.

    Generates:
        - 20 placeholder images (grey squares)
        - 20 text reports
        - Split file
    """
    print("\n" + "="*60)
    print("Creating synthetic sample data for testing")
    print("="*60)

    from PIL import Image
    import random

    random.seed(42)

    SAMPLE_REPORTS = [
        {
            "findings": "The heart size is within normal limits. The lungs are clear bilaterally. No pleural effusion is identified. Bony structures are intact.",
            "impression": "No acute cardiopulmonary abnormality.",
            "answer": "no",
            "question": "Is there evidence of cardiomegaly?",
        },
        {
            "findings": "Cardiomegaly is present. Mild interstitial pulmonary oedema is noted. Small bilateral pleural effusions are identified.",
            "impression": "Cardiomegaly with pulmonary oedema and bilateral pleural effusions.",
            "answer": "yes",
            "question": "Is there evidence of cardiomegaly?",
        },
        {
            "findings": "There is a right lower lobe consolidation consistent with pneumonia. The left lung is clear. Heart size is normal.",
            "impression": "Right lower lobe pneumonia.",
            "answer": "yes",
            "question": "Is there evidence of consolidation?",
        },
        {
            "findings": "Clear lung fields bilaterally. Normal cardiac silhouette. No pneumothorax. No pleural effusion. Bony thorax intact.",
            "impression": "Normal chest X-ray.",
            "answer": "no",
            "question": "Is there evidence of pleural effusion?",
        },
        {
            "findings": "Large right-sided pleural effusion with compressive atelectasis. The mediastinum is shifted to the left. Left lung appears clear.",
            "impression": "Large right pleural effusion with mediastinal shift.",
            "answer": "yes",
            "question": "Is there evidence of pleural effusion?",
        },
        {
            "findings": "Left-sided pneumothorax is present. Approximately 20% collapse of the left lung. Right lung appears clear. Heart size is normal.",
            "impression": "Left pneumothorax.",
            "answer": "yes",
            "question": "Is there evidence of pneumothorax?",
        },
        {
            "findings": "Increased interstitial markings bilaterally. Kerley B lines are present. Cardiomegaly noted. Findings consistent with pulmonary oedema.",
            "impression": "Pulmonary oedema with cardiomegaly.",
            "answer": "yes",
            "question": "Is there evidence of pulmonary oedema?",
        },
        {
            "findings": "No active disease. Lungs are clear. Cardiac size normal. No effusion. Old healed rib fractures noted on the left.",
            "impression": "No acute cardiopulmonary disease.",
            "answer": "no",
            "question": "Is there active pulmonary disease?",
        },
        {
            "findings": "Bilateral hilar lymphadenopathy is present. The lung parenchyma shows fine nodular pattern in both upper lobes. No pleural effusion.",
            "impression": "Findings consistent with sarcoidosis.",
            "answer": "yes",
            "question": "Is there hilar lymphadenopathy?",
        },
        {
            "findings": "A 2.5 cm mass is identified in the right upper lobe. No mediastinal lymphadenopathy. Left lung clear.",
            "impression": "Right upper lobe mass — further evaluation with CT recommended.",
            "answer": "yes",
            "question": "Is there a pulmonary mass?",
        },
    ] * 2  # duplicate to get 20 samples

    # Create directories
    for dataset in ["mimic_cxr", "iu_xray"]:
        img_dir = DATA_ROOT / dataset / "images" / "sample"
        rep_dir = DATA_ROOT / dataset / "reports"
        img_dir.mkdir(parents=True, exist_ok=True)
        rep_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for i, sample in enumerate(SAMPLE_REPORTS):
        doc_id = f"sample_{i:04d}"

        # Create a grey placeholder image (224x224)
        # In real usage these would be actual X-ray images
        brightness = random.randint(80, 180)
        img = Image.new("RGB", (224, 224), color=(brightness, brightness, brightness))

        # Add some variation to make them look different
        import struct
        for dataset in ["mimic_cxr", "iu_xray"]:
            img_path = DATA_ROOT / dataset / "images" / "sample" / f"{doc_id}.png"
            img.save(img_path)

            report_text = (
                f"FINDINGS: {sample['findings']}\n"
                f"IMPRESSION: {sample['impression']}"
            )
            rep_path = DATA_ROOT / dataset / "reports" / f"{doc_id}.txt"
            rep_path.write_text(report_text)

            split = "train" if i < 14 else ("val" if i < 17 else "test")

            items.append({
                "doc_id":   doc_id,
                "image":    str(DATA_ROOT / dataset / "images" / "sample" / f"{doc_id}.png"),
                "text":     report_text,
                "question": sample["question"],
                "answer":   sample["answer"],
                "report":   report_text,
                "domain":   "radiology",
                "split":    split,
            })

        print(f"  Created sample {i+1}/20: {doc_id}")

    # Save splits for both datasets
    for dataset in ["mimic_cxr", "iu_xray"]:
        dataset_items = [x for x in items if dataset in x["image"]]
        split_path = DATA_ROOT / dataset / "splits.json"
        with open(split_path, "w") as f:
            json.dump(dataset_items, f, indent=2)

    print(f"\n✓ Sample data created:")
    print(f"  data/raw/mimic_cxr/  — 20 placeholder images + reports")
    print(f"  data/raw/iu_xray/    — 20 placeholder images + reports")
    print(f"\n  ⚠️  These are grey placeholder images, NOT real X-rays.")
    print(f"  Use them only to verify the pipeline runs end-to-end.")
    print(f"  Download real datasets for actual training.")


# ─────────────────────────────────────────────────────────────────────────────
# MIMIC-CXR INSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def print_mimic_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║              MIMIC-CXR — Manual Download Required               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  MIMIC-CXR requires a free PhysioNet account and a              ║
║  data access agreement. Follow these steps:                      ║
║                                                                  ║
║  Step 1: Create account                                         ║
║    → https://physionet.org/register/                            ║
║                                                                  ║
║  Step 2: Complete CITI training (takes ~2 hours)                ║
║    → https://physionet.org/settings/training/                   ║
║                                                                  ║
║  Step 3: Request access to MIMIC-CXR-JPG                        ║
║    → https://physionet.org/content/mimic-cxr-jpg/2.0.0/        ║
║    Click "Request Access"                                        ║
║                                                                  ║
║  Step 4: Download (after approval, usually 1-3 days)            ║
║    Use the PhysioNet download tool:                              ║
║    pip install physionet-build                                   ║
║    physionet download mimic-cxr-jpg/2.0.0                       ║
║                                                                  ║
║  Step 5: Place files in:                                        ║
║    data/raw/mimic_cxr/images/   ← .jpg images                  ║
║    data/raw/mimic_cxr/reports/  ← .txt reports                 ║
║                                                                  ║
║  WHILE WAITING FOR APPROVAL:                                    ║
║    Use the IU X-ray dataset (free, immediate download)          ║
║    and sample data for pipeline testing.                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────
# VERIFY SETUP
# ─────────────────────────────────────────────────────────────────────────────

def verify_setup():
    """Check what is ready and what is missing."""
    print("\n" + "="*60)
    print("DATASET SETUP STATUS")
    print("="*60)

    checks = {
        "IU X-ray images":      DATA_ROOT / "iu_xray" / "images",
        "IU X-ray reports":     DATA_ROOT / "iu_xray" / "reports",
        "IU X-ray splits":      DATA_ROOT / "iu_xray" / "splits.json",
        "MIMIC-CXR images":     DATA_ROOT / "mimic_cxr" / "images",
        "MIMIC-CXR reports":    DATA_ROOT / "mimic_cxr" / "reports",
        "MIMIC-CXR splits":     DATA_ROOT / "mimic_cxr" / "splits.json",
        "Vector stores dir":    Path("data/vector_stores"),
        "Processed data dir":   Path("data/processed"),
        "Checkpoints dir":      Path("checkpoints"),
    }

    all_ok = True
    for label, path in checks.items():
        exists = path.exists()
        if exists and path.is_dir():
            count = len(list(path.iterdir()))
            status = f"✓  ({count} files)" if count > 0 else "⚠️  (empty)"
        elif exists:
            status = "✓"
        else:
            status = "✗  (missing)"
            all_ok = False

        print(f"  {label:<30} {status}")

    print()
    if all_ok:
        print("✓ All dataset directories are in place.")
    else:
        print("⚠️  Some directories are missing.")
        print("   Run: python scripts/prepare_datasets.py --dataset samples")
        print("   to create placeholder data for testing.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for Meta-MedRAG"
    )
    parser.add_argument(
        "--dataset",
        default="samples",
        choices=["iu_xray", "mimic_cxr", "samples", "verify", "all"],
        help=(
            "samples   = create fake data for immediate testing (recommended first)\n"
            "iu_xray   = download IU X-ray (~1 GB, free)\n"
            "mimic_cxr = show PhysioNet access instructions\n"
            "verify    = check what is ready\n"
            "all       = samples + iu_xray + mimic instructions"
        )
    )
    args = parser.parse_args()

    # Always ensure base directories exist
    for d in ["data/raw/mimic_cxr", "data/raw/iu_xray",
              "data/processed", "data/vector_stores",
              "data/preference_pairs", "checkpoints",
              "experiments/results", "experiments/ablations"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if args.dataset in ("samples", "all"):
        prepare_samples()

    if args.dataset in ("iu_xray", "all"):
        prepare_iu_xray()

    if args.dataset in ("mimic_cxr", "all"):
        print_mimic_instructions()

    if args.dataset in ("verify", "all"):
        verify_setup()


if __name__ == "__main__":
    main()
