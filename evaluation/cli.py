"""python -m evaluation <command>   (run from the repository root)

  stories                 metrics + timing from saved stories
  perplexity [--exp ID]   FINAL test perplexity (only after training is finished)
  human-sheet             make blind rating sheet + key
  human-summary --ratings FILE
  compare                 six-row comparison table
  plots                   all figures
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import config
from . import comparison, human, perplexity, runner, visualization
from .schemas import EXPERIMENT_IDS, RESULTS_DIR, VIS_DIR


def main():
    p = argparse.ArgumentParser(prog="evaluation")
    p.add_argument("cmd", choices=["stories", "perplexity", "human-sheet", "human-summary",
                                   "compare", "plots", "analysis"])
    p.add_argument("--force", action="store_true", help="human-sheet: overwrite key / allow missing runs")
    p.add_argument("--exp", choices=EXPERIMENT_IDS)
    p.add_argument("--outputs", default=str(config.OUTPUT_DIR))
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--ratings", default=str(RESULTS_DIR / "human_ratings.csv"))
    p.add_argument("--max-stories", type=int, help="debug only; never for final results")
    a = p.parse_args()
    res = Path(a.results)

    if a.cmd == "stories":
        rows, problems = runner.evaluate_stories(a.outputs, res)
        print(f"{len(rows)} story records written to {res}")
        for k, v in problems.items():
            print(f"  [{k}] {v}")
    elif a.cmd == "perplexity":
        path = res / "perplexity.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        res.mkdir(parents=True, exist_ok=True)
        for eid in ([a.exp] if a.exp else EXPERIMENT_IDS):
            run = Path(a.outputs) / eid
            if not (run / "best.pt").exists():
                print(f"{eid}: skipped, no best.pt (pending)")
                continue
            if eid in data and not a.exp:
                print(f"{eid}: already evaluated, skipped (use --exp {eid} to redo)")
                continue
            try:
                r = perplexity.evaluate_checkpoint(run, max_stories=a.max_stories)
            except Exception as e:
                print(f"{eid}: FAILED: {e}")
                continue
            if a.max_stories:
                print(f"{eid}: DEBUG run on {a.max_stories} stories, NOT saved")
            else:
                data[eid] = r
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")   # save after each run
            print(eid, f"CE={r['test_mean_cross_entropy']:.4f} PPL={r['test_perplexity']:.2f} "
                       f"tokens={r['test_token_count']} unk={r['test_unknown_token_rate']:.4f}")
    elif a.cmd == "human-sheet":
        print(human.make_blind_sheet(a.outputs, res, force=a.force), "stories in blind sheet")
    elif a.cmd == "human-summary":
        print(human.run_human_summary(a.ratings, res / "blind_key.csv", res).to_string(index=False))
    elif a.cmd == "compare":
        print(comparison.build_comparison(a.outputs, res).to_string(index=False))
    elif a.cmd == "plots":
        story = pd.read_csv(res / "automatic_evaluation.csv") if (res / "automatic_evaluation.csv").exists() else pd.DataFrame()
        comp = comparison.build_comparison(a.outputs, res)
        hs = pd.read_csv(res / "human_summary.csv") if (res / "human_summary.csv").exists() else None
        out = [*visualization.plot_story_metrics(story, VIS_DIR),
               visualization.plot_training_curves(a.outputs, VIS_DIR),
               visualization.plot_best_val(comp, VIS_DIR),
               visualization.plot_perplexity(comp, VIS_DIR),
               visualization.plot_human(hs, VIS_DIR)]
        for o in out:
            for x in (o if isinstance(o, list) else [o]):
                if x:
                    print("saved", x)
    elif a.cmd == "analysis":
        from .analysis import build_analysis
        print(build_analysis(res))