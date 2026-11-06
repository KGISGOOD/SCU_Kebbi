"""Thin V2-only QA and training entrypoints; never duplicate inference logic."""
import sys
from experiments.benchmark_804.common import ROOT

def benchmark(architecture):
    if any(a.split("=", 1)[0] in {"--job", "--jobs"} for a in sys.argv[1:]):
        raise ValueError("this entry selects one architecture; use run_all for custom jobs")
    if any(a.split("=", 1)[0] == "--questions-file" for a in sys.argv[1:]):
        from experiments.qa.custom import main
        return main(architecture, sys.argv[1:])
    from experiments.benchmark_804.run_all import main
    return main(["--jobs", "qa_" + architecture] + sys.argv[1:])

def train(mode):
    # Delegate directly; native training CLI still enforces approvals/provenance.
    if mode in {"a2_1", "a2_2"}:
        if any(a == "--mode" or a.startswith("--mode=") for a in sys.argv[1:]):
            raise ValueError("architecture-specific training mode cannot be overridden")
        from experiments.a2_faq_finetuning.train_faq_embedding import main
        defaults = ["--mode", mode, "--v2-data-dir", str(ROOT / "experiments/datasets/training_v2/converted"),
                    "--config", str(ROOT / "experiments/a2_faq_finetuning/training_v2_config.json")]
    else:
        from experiments.a3_faq_sft.train import main
        defaults = ["--data-dir", str(ROOT / "experiments/datasets/training_v2/converted")]
    sys.argv[1:1] = defaults
    return main()
