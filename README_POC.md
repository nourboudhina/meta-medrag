\# Meta-MedRAG — Proof of Concept



\## Reproduire les résultats en 5 commandes



\### 1. Installation

```

conda activate medrag

set PYTHONPATH=%CD%

```



\### 2. Préparer les datasets

```

python scripts/build\_iu\_xray\_split.py

python scripts/prepare\_vqa\_rad.py

```



\### 3. Construire les index FAISS

```

python scripts/build\_vector\_stores.py --domain radiology --device cuda

python scripts/build\_vector\_stores.py --domain pathology --device cuda

```



\### 4. Construire les paires contrastives

```

python scripts/train\_probe.py --step build\_pairs

```



\### 5. Lancer le demo Gradio

```

python -m src.interface.app --port 7860

```



\## Datasets utilisés

\- IU-Xray: 3947 rapports radiology

\- VQA-RAD: 2244 Q/A avec images

\- SLAKE: 7033 questions EN

\- MIMIC-CXR: 20 images (POC)

\- PMC-OA: 500 items synthétiques pathologie



\## Structure du projet

\- src/module1\_metacognition/ — Sonde MeCo (PCA + LogReg)

\- src/module2\_retrieval/ — FAISS + BiomedCLIP

\- src/module3\_alignment/ — DPO LoRA

\- src/pipeline.py — Pipeline complet

\- src/interface/app.py — Demo Gradio

