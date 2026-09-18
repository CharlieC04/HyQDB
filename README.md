# HyQDB

A tool-augmented LLM framework for debugging **hybrid quantum-classical programs**. HyQDB injects deterministic hardware, optimization, and physics evidence into an LLM repair call, and escalates to an intent-reconstruction tier when the detectors find no evidence. This repository also contains **QFaultBench**, the benchmark used to evaluate the framework.

## Installation

Requires Python ≥ 3.10.

```bash
git clone <repo-url>
cd hyqdb
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Set the API key for whichever model provider you use in a `.env` file at the repo root:

```bash
DEEPSEEK_API_KEY=...   # deepseek-* models (default)
GOOGLE_API_KEY=...     # gemini-* models  (also: pip install langchain-google-genai)
OPENAI_API_KEY=...     # gpt-* / o-series (also: pip install langchain-openai)
```

The provider is dispatched automatically from the model name prefix.

## Usage

### Repair a single script

```python
from hyqagent.agent import fix_script

buggy = open("my_program.py").read()
fixed = fix_script(buggy, model_name="deepseek-chat")
print(fixed)
```

`fix_script` returns the corrected, executable script. Key arguments:

| Argument | Default | Meaning |
|---|---|---|
| `model_name` | `"deepseek-chat"` | Repair model (`deepseek-*`, `gemini-*`, `gpt-*`). |
| `tools` | `("hardware","optim","physics")` | Active detectors; `()` is the un-augmented baseline. |
| `use_intent` | `True` | Enable the intent-reconstruction tier. |
| `gate` | `True` | Escalate to intent only when detectors are silent; `False` runs it on every task. |

## QFaultBench

The benchmark lives in `evaluation/` as two files:

- **`hyqbench.json`** — development set (34 tasks)
- **`hyqbench_holdout.json`** — held-out set of human-authored programs (28 tasks)

Each entry maps a task name to `{"code": <buggy program>, "asserts": [<hidden checks>]}`.
A repair passes only if the fixed script satisfies all hidden asserts.

### Reproducing the paper results

```bash
cd evaluation

# Full sweep over both splits (three runs, all configurations)
python run_experiments.py --bench hyqbench.json         --configs baseline,full,full_intent --runs 3
python run_experiments.py --bench hyqbench_holdout.json  --configs baseline,full,full_intent --runs 3
```

Results are written per-model to `results_<bench>__<model>.json` (with routing decisions
in the matching `_routes.json`). Swap `--model gemini-3.6-flash` or `--model gpt-5-mini`
to reproduce the cross-model comparison.

