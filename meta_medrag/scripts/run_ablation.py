"""
scripts/run_ablation.py

Ablation study runner.
Evaluates the contribution of each module by disabling them one at a time.

Ablation conditions:
    full:       complete Meta-MedRAG system (all 3 modules)
    no_probe:   always trigger RAG (no MeCo probe — Module 1 disabled)
    no_dpo:     use base LLaVA-Med without DPO fine-tuning (Module 3 disabled)
    no_filter:  retrieve fixed k=5 (no adaptive-k filter — part of Module 2 disabled)
    baseline:   vanilla LLaVA-Med with no RAG at all

Usage:
    python scripts/run_ablation.py --condition all --dataset mimic_cxr --task vqa
"""

import argparse
import yaml
import json
from pathlib import Path
from loguru import logger

from src.evaluation.metrics import evaluate_vqa, evaluate_report_generation, save_results


class AblatedPipeline:
    """
    A version of MetaMedRAGPipeline with specific modules disabled,
    used to measure each module's contribution.
    """

    def __init__(self, cfg: dict, condition: str):
        self.cfg       = cfg
        self.condition = condition
        logger.info(f"Loading ablation pipeline: condition={condition}")
        self._setup(condition)

    def _setup(self, condition: str):
        from src.backbone.llava_med import LLaVAMedBackbone
        from src.module1_metacognition.meco_probe import MeCoProbe
        from src.module2_retrieval.domain_classifier import DomainClassifier
        from src.module2_retrieval.vector_store import MultiDomainRetriever

        # Backbone is always the same
        model_path = (
            self.cfg["alignment"]["output_dir"] + "_merged"
            if condition != "no_dpo" and condition != "baseline"
            else self.cfg["backbone"]["model_name"]
        )
        backbone_cfg = {**self.cfg["backbone"], "model_name": model_path}
        self.backbone = LLaVAMedBackbone(backbone_cfg)

        # Load probe only when needed
        if condition not in ("no_probe", "baseline"):
            mc_cfg = self.cfg["metacognition"]
            self.probe = MeCoProbe(
                n_components=mc_cfg["pca_components"],
                theta_low=mc_cfg["theta_low"],
                theta_high=mc_cfg["theta_high"],
            )
            self.probe.load(mc_cfg["probe_checkpoint"])
        else:
            self.probe = None

        # Load retriever only when needed
        if condition != "baseline":
            self.domain_clf = DomainClassifier(
                model_name=self.cfg["retrieval"]["biomed_clip_model"]
            )
            self.retriever = MultiDomainRetriever(self.cfg)
        else:
            self.domain_clf = None
            self.retriever  = None

    def run(self, image, question: str) -> dict:
        from src.module1_metacognition.meco_probe import MeCoProbe

        result = {"question": question, "decision": "N/A", "meco_score": 0.0}

        # ── Baseline: no RAG ─────────────────────────────────────────
        if self.condition == "baseline":
            result["answer"]   = self.backbone.generate(image, question)
            result["decision"] = "direct"
            return result

        # ── No probe: always retrieve ────────────────────────────────
        if self.condition == "no_probe":
            decision   = MeCoProbe.FULL_RAG
            meco_score = 1.0
        else:
            hidden     = self.backbone.get_hidden_states(image, question,
                            layers=self.cfg["metacognition"]["probe_layers"])
            meco_score, decision = self.probe.score_and_decide(hidden)

        result["meco_score"] = meco_score
        result["decision"]   = decision

        # ── Direct path (no retrieval) ───────────────────────────────
        if decision == MeCoProbe.DIRECT:
            result["answer"] = self.backbone.generate(image, question)
            return result

        # ── Retrieval path ───────────────────────────────────────────
        domain, _, _ = self.domain_clf.classify(image)
        query_emb    = self.domain_clf.encode_image(image)

        if self.condition == "no_filter":
            # Fixed k=5, no adaptive filtering
            store = self.retriever.stores.get(domain)
            if store:
                docs = store.retrieve(query_emb, max_k=5, similarity_ratio_threshold=0.0)
                context = store.format_context(docs)
            else:
                context = ""
        else:
            _, context = self.retriever.retrieve(
                query_embedding=query_emb,
                domain=domain,
                decision=decision,
            )

        result["answer"] = self.backbone.generate(image, question, context or None)
        return result


def run_condition(condition: str, cfg: dict, test_data: list, task: str) -> dict:
    """Run one ablation condition and return metrics."""
    logger.info(f"Running ablation condition: {condition}")

    abl = AblatedPipeline(cfg, condition)
    predictions = []
    references  = []
    ref_key = "answer" if task == "vqa" else "report"

    from tqdm import tqdm
    for item in tqdm(test_data, desc=condition):
        out = abl.run(item["image"], item["question"])
        predictions.append(out["answer"])
        references.append(item.get(ref_key, ""))

    if task == "vqa":
        metrics = evaluate_vqa(predictions, references)
    else:
        metrics = evaluate_report_generation(predictions, references)

    logger.info(f"Condition={condition}: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="all",
                        choices=["full", "no_probe", "no_dpo",
                                 "no_filter", "baseline", "all"])
    parser.add_argument("--dataset", default="mimic_cxr",
                        choices=["mimic_cxr", "iu_xray"])
    parser.add_argument("--task",    default="vqa",
                        choices=["vqa", "report_gen"])
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--limit",   default=200, type=int,
                        help="Number of test items per condition (for speed)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load test data
    split_file = cfg["datasets"][args.dataset]["split_file"]
    with open(split_file) as f:
        all_data = json.load(f)
    test_data = [x for x in all_data if x.get("split") == "test"][:args.limit]
    logger.info(f"Test items: {len(test_data)}")

    conditions = (
        ["full", "no_probe", "no_dpo", "no_filter", "baseline"]
        if args.condition == "all"
        else [args.condition]
    )

    all_results = {}
    for cond in conditions:
        metrics = run_condition(cond, cfg, test_data, args.task)
        all_results[cond] = metrics

    # Save combined results
    out_path = f"experiments/ablations/{args.dataset}_{args.task}_ablation.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print("\n" + "="*70)
    print(f"ABLATION RESULTS — {args.dataset.upper()} / {args.task.upper()}")
    print("="*70)

    if all_results:
        metric_keys = list(list(all_results.values())[0].keys())
        header = f"{'Condition':<18} " + " ".join(f"{k:>12}" for k in metric_keys)
        print(header)
        print("-"*70)
        for cond, metrics in all_results.items():
            row = f"{cond:<18} " + " ".join(
                f"{metrics.get(k, 0.0):>12.4f}" for k in metric_keys
            )
            print(row)
    print("="*70)
    logger.info(f"Ablation results saved to {out_path}")


if __name__ == "__main__":
    main()
