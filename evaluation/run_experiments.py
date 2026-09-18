"""

Runs a benchmark under one or more tool CONFIGURATIONS, repeated N times, and records
per-run pass/fail

Usage:
  python run_experiments.py --bench hyqbench.json          --runs 3 --configs baseline,full
  python run_experiments.py --bench hyqbench_holdout.json  --runs 3 --configs full,hardware,optim,physics

Config names -> active tools:
  baseline : ()                      (pure LLM, no detectors)
  full     : hardware+optim+physics  (the complete agent)
  hardware / optim / physics : that single detector only
"""

import argparse
import json
import os
import re
import subprocess
import time
import statistics
import warnings
import logging

warnings.filterwarnings("ignore")
for _lg in ("google_genai", "google.genai", "langchain_google_genai"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import hyqagent.agent as agent
from hyqagent.agent import fix_script

CONFIGS = {
    "baseline": ((), False, True),
    "full": (("hardware", "optim", "physics"), False, True),          # tools, NO intent
    "full_intent": (("hardware", "optim", "physics"), True, True),    # tools + gated Tier-2 intent
    "full_intent_nogate": (("hardware", "optim", "physics"), True, False),  # always-on intent
    "hardware": (("hardware",), False, True),
    "optim": (("optim",), False, True),
    "physics": (("physics",), False, True),
}


def run_asserts(fixed_code, asserts):
    combined = fixed_code + "\n\n# --- ASSERTS ---\n" + "\n".join(asserts)
    with open("temp_experiment_run.py", "w") as f:
        f.write(combined)
    try:
        r = subprocess.run(["python", "temp_experiment_run.py"], capture_output=True,
                           text=True, timeout=90, env=dict(os.environ, PYTHONWARNINGS="ignore"))
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        if os.path.exists("temp_experiment_run.py"):
            os.remove("temp_experiment_run.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="hyqbench.json")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--configs", default="baseline,full")
    ap.add_argument("--model", default="deepseek-chat",
                    help="Repair model; provider dispatched by prefix "
                         "(deepseek-*/gemini-*/gpt-*). Same tools+prompts run unchanged.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench = json.load(open(args.bench))
    configs = [c.strip() for c in args.configs.split(",")]

    bench_stem = os.path.splitext(os.path.basename(args.bench))[0]
    model_tag = re.sub(r"[^A-Za-z0-9.-]+", "-", args.model)
    out_path = args.out or f"results_{bench_stem}__{model_tag}.json"
    routes_path = out_path.replace('.json', '_routes.json')
    print(f"model={args.model}  ->  {out_path}")

    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    routes = json.load(open(routes_path)) if os.path.exists(routes_path) else {}
    for c in configs:
        results[c] = {t: [] for t in bench}
        routes[c] = {t: [] for t in bench}

    for run in range(args.runs):
        for config in configs:
            tools, use_intent, gate = CONFIGS[config]
            print(f"\n===== run {run+1}/{args.runs}  config={config}  =====")
            for test_name, data in bench.items():
                fixed = fix_script(data["code"], model_name=args.model,
                                   tools=tools, use_intent=use_intent, gate=gate)
                ok = bool(fixed) and run_asserts(fixed, data["asserts"])
                results[config][test_name].append(ok)
                routes[config][test_name].append(agent.LAST_META.get("route", "?"))
                print(f"  [{config}] {test_name:34s} {'PASS' if ok else 'fail'}  ({agent.LAST_META.get('route','?')})")
                json.dump(results, open(out_path, "w"), indent=2)
                json.dump(routes, open(routes_path, "w"), indent=2)
                time.sleep(3)

    print("\n==SUMMARY ==")
    n = len(bench)
    for config in configs:
        per_run = [sum(results[config][t][i] for t in bench) for i in range(args.runs)]
        mean = statistics.mean(per_run)
        sd = statistics.pstdev(per_run) if args.runs > 1 else 0.0
        print(f"  {config:12s}: {mean:.1f}/{n} = {mean/n*100:.1f}%  "
              f"(+/- {sd:.1f} over {args.runs} runs; per-run {per_run})")

    for config in configs:
        flat = [r for t in bench for r in routes[config][t]]
        if any(r in ("intent", "intent->fallback") for r in flat):
            from collections import Counter
            c = Counter(flat)
            intent_tests = sorted({t for t in bench if "intent" in routes[config][t][0]})
            intent_pass = sum(1 for t in intent_tests if all(results[config][t]))
            print(f"\n  [{config}] routing: {dict(c)}")
            print(f"  [{config}] intent-routed tests ({len(intent_tests)}), of which pass-all-runs: {intent_pass}")

    print(f"\nSaved: {out_path}  (+ routes)")


if __name__ == "__main__":
    main()
