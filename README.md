\# Meta-MedRAG

\### Meta-Cognitive Diagnostic Agents for Factual Medical Imaging Analysis



\*\*Candidate:\*\* Nour El Houda BOUDHINA | \*\*Supervisor:\*\* Prof. Lotfi Tlig | \*\*ISIMG 2025-2026\*\*



\## What is Meta-MedRAG?

Meta-MedRAG is a three-module system that makes medical image question-answering

more factual by deciding WHEN to retrieve external knowledge — not always, only when needed.



\*\*3 Innovations:\*\*

\- Module 1 — MeCo dual-threshold probe: reads LLaVA-Med hidden states to decide if retrieval is needed

\- Module 2 — Domain-aware FAISS retrieval: BiomedCLIP + adaptive-k filter

\- Module 3 — DPO alignment: fine-tuned to avoid ignoring the medical image



\## Quick Start 



\### Option A — Google Colab (recommended)

Open the notebook directly:

\[!\[Open In Colab](https://colab.research.google.com/github/nourboudhina/meta-medrag/blob/main/notebooks/Meta_MedRAG_POC_Supervisor.ipynb)



1\. Runtime → Change runtime type → \*\*GPU T4\*\*

2\. Run all cells in order

3\. Results saved automatically to Google Drive



\### Option B — Local installation

```bash

git clone https://github.com/nourboudhina/meta-medrag.git

cd meta-medrag

pip install -r requirements.txt

python -m src.interface.app --port 7860 --no-pipeline

```



\## Project Structure
meta_medrag/
├── src/
│   ├── module1_metacognition/   # MeCo probe (PCA + LogReg)
│   ├── module2_retrieval/       # FAISS + BiomedCLIP
│   ├── module3_alignment/       # DPO LoRA
│   ├── pipeline.py              # End-to-end pipeline
│   └── interface/app.py         # Gradio demo
├── scripts/
│   ├── train_probe.py           # Build pairs / extract / train / evaluate
│   ├── build_vector_stores.py   # Build FAISS indexes
│   └── run_poc_evaluation.py    # Baseline vs Meta-MedRAG
├── notebooks/
│   └── Meta_MedRAG_POC_Supervisor.ipynb  # Complete POC notebook
├── configs/config.yaml          # theta_low=0.35, theta_high=0.65
└── requirements.txt

## Datasets Used
| Dataset | Size | Role |
|---------|------|------|
| IU-Xray | 3,947 reports | FAISS radiology index + report generation |
| VQA-RAD | 2,244 Q/A | VQA evaluation (closed questions) |
| SLAKE | 7,033 Q/A EN | VQA evaluation (open questions) |
| MIMIC-CXR | 20 images | Radiology evaluation |
| PMC-OA | 500 items | FAISS pathology index |

## System Architecture
(Image, Question)
│
▼
Module 1 — MeCo Probe
LLaVA-Med hidden states → PCA → LogReg → score s ∈ [0,1]
│
├─ s < 0.35 ──→ DIRECT answer (no retrieval)
├─ 0.35 < s < 0.65 ──→ SOFT RAG (1 document)
└─ s > 0.65 ──→ FULL RAG (adaptive-k documents)
│
Module 2
BiomedCLIP domain classifier
FAISS similarity search
Adaptive-k filter (ratio=0.85)
│
LLaVA-Med
(+ Module 3 DPO weights)
│
Final Answer

## Current Status
- ✓ Module 1 code complete — awaiting GPU for activation extraction
- ✓ Module 2 fully tested — FAISS 3,947 vectors, adaptive-k working
- ✓ Gradio interface functional
- ⏳ Module 3 DPO — RunPod A100 scheduled

## Note on DPO (Module 3)
DPO fine-tuning requires A100 80GB GPU. This will be run on RunPod.
The POC notebook evaluates Module 1 + Module 2 without DPO — this is
scientifically valid and consistent with ablation study design.
