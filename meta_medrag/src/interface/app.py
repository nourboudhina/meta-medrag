"""
src/interface/app.py

Gradio demo interface for Meta-MedRAG.

Features:
    - Upload any medical image (X-ray, pathology slide, fundus)
    - Ask a clinical question in text
    - See MeCo score gauge (confidence meter)
    - See RAG decision (direct / soft_rag / full_rag)
    - See retrieved documents with similarity scores
    - See the final factual answer

Run:
    python -m src.interface.app
    or
    gradio src/interface/app.py
"""

import gradio as gr
import yaml
import time
from pathlib import Path
from loguru import logger

from src.pipeline import MetaMedRAGPipeline, PipelineOutput


# ── Global pipeline (loaded once at startup) ──────────────────────────
pipeline = None


def load_pipeline(config_path: str = "configs/config.yaml"):
    global pipeline
    logger.info("Loading Meta-MedRAG pipeline...")
    pipeline = MetaMedRAGPipeline(config_path=config_path)
    logger.info("Pipeline ready")


# ── Inference function called by Gradio ───────────────────────────────

def run_inference(image, question: str):
    """
    Main inference function.
    Called by Gradio when user clicks Submit.

    Args:
        image:    PIL Image from Gradio upload component
        question: string from textbox

    Returns:
        tuple of Gradio output values (mapped to interface outputs)
    """
    if pipeline is None:
        return (
            "Pipeline not loaded. Please restart the app.",
            "N/A", "N/A", "N/A", 0.0, "N/A", "N/A"
        )

    if image is None:
        return (
            "Please upload a medical image.",
            "N/A", "N/A", "N/A", 0.0, "N/A", "N/A"
        )

    if not question.strip():
        return (
            "Please enter a clinical question.",
            "N/A", "N/A", "N/A", 0.0, "N/A", "N/A"
        )

    try:
        output: PipelineOutput = pipeline.run(image=image, question=question)

        # ── Format MeCo decision badge ────────────────────────────────
        decision_map = {
            "direct":   ("No Retrieval (Direct)",   "#1D9E75"),
            "soft_rag": ("Soft Retrieval (1 doc)",  "#BA7517"),
            "full_rag": ("Full Retrieval",           "#993C1D"),
        }
        decision_label, _ = decision_map.get(
            output.decision, (output.decision, "#888780")
        )

        # ── Format retrieved documents ────────────────────────────────
        if output.retrieved_docs:
            docs_text = ""
            for i, doc in enumerate(output.retrieved_docs, 1):
                docs_text += (
                    f"**Document {i}** (similarity: {doc.score:.3f})\n"
                    f"{doc.text[:300]}...\n\n"
                )
        else:
            docs_text = "No documents retrieved (direct answer path)"

        # ── Format domain info ────────────────────────────────────────
        domain_info = (
            f"{output.domain} ({output.domain_confidence:.1%})"
            if output.domain else "N/A"
        )

        return (
            output.answer,                        # answer textbox
            f"{output.meco_score:.3f}",           # meco score
            decision_label,                       # decision label
            domain_info,                          # domain classification
            output.meco_score,                    # slider value for gauge
            docs_text,                            # retrieved docs markdown
            f"{output.latency_ms:.0f} ms",        # latency
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return (
            f"Error during inference: {str(e)}",
            "N/A", "N/A", "N/A", 0.0, "N/A", "N/A"
        )


# ── Example cases ──────────────────────────────────────────────────────
EXAMPLES = [
    ["data/raw/mimic_cxr/sample/sample1.jpg",
     "Is there evidence of cardiomegaly in this chest X-ray?"],
    ["data/raw/mimic_cxr/sample/sample2.jpg",
     "Are there signs of pleural effusion in this radiograph?"],
    ["data/raw/iu_xray/sample/sample1.jpg",
     "Describe the main findings in this chest X-ray."],
]


# ── Build Gradio interface ─────────────────────────────────────────────

def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="Meta-MedRAG — Factual Medical Imaging Analysis",
        theme=gr.themes.Soft(),
        css="""
        .meco-high { color: #993C1D; font-weight: bold; }
        .meco-low  { color: #1D9E75; font-weight: bold; }
        #answer-box textarea { font-size: 15px; line-height: 1.6; }
        """
    ) as demo:

        # ── Header ────────────────────────────────────────────────────
        gr.Markdown("""
        # Meta-MedRAG
        ### Meta-Cognitive Diagnostic Agents for Factual Medical Imaging Analysis
        *Master's Research Project — ISIMG 2025-2026 | Nour BOUDHINA*

        Upload a medical image, ask a clinical question, and see how the system
        decides whether to retrieve external knowledge before answering.
        """)

        # ── Main row ──────────────────────────────────────────────────
        with gr.Row():

            # Left column — inputs
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                image_input = gr.Image(
                    type="pil",
                    label="Medical image",
                    height=300,
                )
                question_input = gr.Textbox(
                    label="Clinical question",
                    placeholder="e.g. Is there evidence of cardiomegaly?",
                    lines=2,
                )
                submit_btn = gr.Button(
                    "Analyse", variant="primary", size="lg"
                )

                gr.Markdown("### Examples")
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[image_input, question_input],
                    label="Click to load an example",
                )

            # Right column — outputs
            with gr.Column(scale=1):
                gr.Markdown("### Module 1 — Meta-cognition")
                with gr.Row():
                    meco_score_box = gr.Textbox(
                        label="MeCo score", interactive=False, scale=1
                    )
                    decision_box = gr.Textbox(
                        label="Decision", interactive=False, scale=2
                    )

                meco_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0,
                    label="MeCo score gauge (0=confident → 1=uncertain)",
                    interactive=False,
                )
                with gr.Row():
                    gr.Markdown("◀ No retrieval needed", scale=1)
                    gr.Markdown("▶ Full retrieval triggered", scale=1)

                gr.Markdown("### Module 2 — Domain & Retrieval")
                domain_box = gr.Textbox(
                    label="Detected domain", interactive=False
                )
                retrieved_docs = gr.Markdown(
                    label="Retrieved documents",
                    value="Retrieved documents will appear here after analysis...",
                )

                gr.Markdown("### Final Answer")
                answer_box = gr.Textbox(
                    label="Factual clinical answer",
                    lines=6,
                    interactive=False,
                    elem_id="answer-box",
                )
                latency_box = gr.Textbox(
                    label="Latency", interactive=False
                )

        # ── Wire up ───────────────────────────────────────────────────
        submit_btn.click(
            fn=run_inference,
            inputs=[image_input, question_input],
            outputs=[
                answer_box,
                meco_score_box,
                decision_box,
                domain_box,
                meco_slider,
                retrieved_docs,
                latency_box,
            ],
        )

        # Also allow Enter key in question box
        question_input.submit(
            fn=run_inference,
            inputs=[image_input, question_input],
            outputs=[
                answer_box, meco_score_box, decision_box,
                domain_box, meco_slider, retrieved_docs, latency_box,
            ],
        )

        # ── Footer ────────────────────────────────────────────────────
        gr.Markdown("""
        ---
        **System info:**
        - Backbone: LLaVA-Med-1.5 (7B)
        - Module 1: RepE linear probe with dual-threshold (θ_low=0.35, θ_high=0.65)
        - Module 2: BiomedCLIP + FAISS adaptive-k retrieval
        - Module 3: DPO + LoRA fine-tuned for cross-modal alignment
        """)

    return demo


# ── Launch ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--port",   default=7860, type=int)
    parser.add_argument("--share",  action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="Skip pipeline loading (for UI testing)")
    args = parser.parse_args()

    if not args.no_pipeline:
        load_pipeline(args.config)

    demo = build_interface()
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="0.0.0.0",
    )
