import os
import re
import json
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage

from hyqagent.tools.optim import profile
from hyqagent.tools.hardware import analyse_hardware_topology
from hyqagent.tools.physics import evaluate_symmetries

# Observability
LAST_META = {}

_ENV = """ENVIRONMENT (PennyLane 0.45 / JAX / PyTorch):
- Use only current, real APIs. Do NOT invent or guess API names.
- `qml.device(dev, wires=n, shots=N)` is valid. For analytic mode use shots=None.
- Standard optimizers: qml.AdamOptimizer, qml.GradientDescentOptimizer, qml.AdagradOptimizer, optax.adam.
- If your fix addresses the root cause but the model only converges marginally, also make sure the
  optimization is given enough steps and a robust step size so it actually reaches the target.
- Make the minimal change that fixes the bug. Do NOT rewrite working code, rename things, or add docstrings/comments for style.

OUTPUT FORMAT:
Return ONLY the complete, corrected, executable Python script inside a single ```python``` code block. No explanation before or after."""


def _msg_text(message) -> str:
    """
    Flatten response to a plain string.
    """

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
        return "".join(parts)
    return str(content or "")


def extract_code(response: str) -> str:
    """
    Extract python code from an LLM response
    """

    if not response:
        return ""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)(?:\n```|\Z)", response, re.DOTALL)
    code = match.group(1) if match else response
    code = code.replace("```", "")
    code = re.sub(r"^\s*(python|py)\s*\n", "", code)
    lines = code.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |#|def |class |@|dev |import\b)", line):
            start = i
            break
    return "\n".join(lines[start:]).strip()


def _insights(report_json: str):
    try:
        return json.loads(report_json).get("agent_insights", [])
    except Exception:
        return []


def _facts(*report_jsons) -> str:
    """
    Compact tool reports
    """

    keep = ("device", "wires", "detected_gates", "entangling_gate_count", "templates_used",
            "problem_domain", "measurements", "ansatz_templates", "state_preparation",
            "device_shots", "interface_violations", "qnode_executions")
    merged = {}
    for rj in report_jsons:
        try:
            d = json.loads(rj)
        except Exception:
            continue
        for k in keep:
            if k in d and d[k] not in (None, [], {}, "Unknown"):
                merged[k] = d[k]
    if not merged:
        return "(no structured facts available)"
    return "\n".join(f"- {k}: {v}" for k, v in merged.items())


def _llm(model_name, max_tokens):
    """
    LLM (3 providers)
    """

    name = model_name.lower()
    if name.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.0,
                                      max_output_tokens=max_tokens,
                                      google_api_key=os.environ.get("GOOGLE_API_KEY"))
    if name.startswith(("gpt", "o1", "o3", "o4")):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=0.0, max_tokens=max_tokens,
                          api_key=os.environ.get("OPENAI_API_KEY"))
    return ChatDeepSeek(api_key=os.environ.get("DEEPSEEK_API_KEY"),
                        model=model_name, temperature=0.0, max_tokens=max_tokens)


def _repair_call(llm, model_name, system_prompt, buggy_code):
    """
    Invoke a repair prompt, extract code, retry once at a larger budget if empty
    """

    messages = [SystemMessage(content=system_prompt),
                HumanMessage(content=f"Fix this buggy quantum script:\n\n```python\n{buggy_code}\n```")]
    fixed = extract_code(_msg_text(llm.invoke(messages)))
    if not fixed:
        fixed = extract_code(_msg_text(_llm(model_name, 16000).invoke(messages)))
    return fixed


def _standard_prompt(all_insights):
    diag = "Our static and dynamic analysers flagged the following (treat as strong hints, not gospel):\n"
    diag += "\n".join(f"- {s}" for s in all_insights)
    return f"""You are an expert Quantum Machine Learning Engineer. You are given a hybrid quantum-classical PennyLane script that runs but contains a silent bug: it fails to converge, diverges, crashes, or solves the wrong problem. Fix it.

        --- AUTOMATED DIAGNOSTICS ---
        {diag}

        HOW TO USE THE DIAGNOSTICS:
        - They are evidence from analysers, not commands. Weigh them, but your goal is code that RUNS and CONVERGES.
        - If a hint contradicts an obvious bug you can see in the code, trust the code.
        - Also fix any bug the analysers did NOT mention (shape mismatches, wrong reshape/transpose, indexing, missing data re-upload, wrong sign, etc.). Do not stop at the hinted bug.

    {_ENV}"""


def _thin_prompt():
    return f"""You are an expert Quantum Machine Learning Engineer. You are given a hybrid quantum-classical PennyLane script that runs but contains a silent semantic, logical, physical, or shape bug: it fails to converge or solves the wrong problem. Find and fix the bug.

    {_ENV}"""


# ---- Tier 2: intent reconstruction --------------------------------------------------

def _intent_infer(llm, buggy_code, facts):
    """
    Stage 2a: reconstruct the program intent
    """

    system = """You are a quantum software analyst. The PennyLane program below is KNOWN to contain a silent SEMANTIC bug: the circuit may not implement what it was intended to implement. Your job is NOT to fix it yet - it is to reconstruct the AUTHOR'S INTENT.

    Deliberately DISTRUST the circuit body. Infer intent only from: function names, variable names, docstrings/comments, the problem set-up, and the extracted facts.

    Produce a concise structured spec:
    1. ALGORITHM / PROBLEM: which quantum ML algorithm is this (QAOA MaxCut, VQE chemistry, data-reuploading classifier, variational regressor, quantum transfer learning, ...)?
    2. EXPECTED COMPONENTS: state initialisation; data encoding and whether data RE-UPLOADING per layer is expected; the variational ansatz and which parameters must be trainable; the cost/measurement observable AND its correct basis; the optimization DIRECTION (minimise/maximise what quantity).
    3. INTENT VIOLATIONS: concrete places where the implementation appears to CONTRADICT the intent - e.g. a helper unitary that is defined but never called; a trainable parameter passed to the QNode but unused; a loop variable that is ignored; a measurement in the wrong basis; a sign/optimization-direction inconsistency; data encoded once where re-uploading was intended.

    Be specific and brief. Do not output code."""

    human = f"EXTRACTED FACTS:\n{facts}\n\nPROGRAM:\n```python\n{buggy_code}\n```"
    try:
        spec = _msg_text(llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]))
        return (spec or "").strip()
    except Exception:
        return ""


def _intent_repair_prompt(intent_spec):
    """
    Stage 2b: reconcile implementation with the reconstructed intent
    """

    return f"""You are an expert Quantum Machine Learning Engineer fixing a silent semantic bug: the program runs but does not implement what it was intended to. A prior analysis reconstructed the AUTHOR'S INTENT from the code's names, structure and problem type:

    --- RECONSTRUCTED INTENT ---
    {intent_spec}

    YOUR TASK:
    - Make the implementation match this intent. Focus on the listed INTENT VIOLATIONS (unused operators/parameters, wrong measurement basis, wrong optimization direction/sign, missing data re-uploading, degenerate ansatz).
    - If the intent analysis is wrong about something the code clearly gets right, trust the code.
    - Also fix any ordinary bug (shape, indexing) needed for the program to run and converge.

    {_ENV}"""


# Router
def fix_script(buggy_code: str, model_name: str = "deepseek-chat",
               tools=("hardware", "optim", "physics"), use_intent: bool = True,
               gate: bool = True) -> str:
    """
    Repair a buggy hybrid quantum-classical script.
    """

    global LAST_META
    llm = _llm(model_name, 8000)
    tools = set(tools)

    try:
        hardware_report = analyse_hardware_topology(buggy_code) if "hardware" in tools else "{}"
        optim_report = profile(buggy_code) if "optim" in tools else "{}"
        physics_report = evaluate_symmetries(buggy_code) if "physics" in tools else "{}"

        all_insights = (_insights(hardware_report)
                        + _insights(optim_report)
                        + _insights(physics_report))

        # Tier 1: detectors find the fault
        if all_insights and gate:
            LAST_META = {"route": "standard", "n_warnings": len(all_insights)}
            print(f"route=STANDARD ({len(all_insights)} warning(s))")
            return _repair_call(llm, model_name, _standard_prompt(all_insights), buggy_code)

        # Tier 2: intent reconstruction
        if tools and use_intent:
            _why = "detectors silent" if all_insights == [] else "no gate"
            print(f"route=INTENT ({_why}) - reconstructing intent")
            facts = _facts(hardware_report, optim_report, physics_report)
            spec = _intent_infer(llm, buggy_code, facts)
            if spec:
                fixed = _repair_call(llm, model_name, _intent_repair_prompt(spec), buggy_code)
                if fixed:
                    LAST_META = {"route": "intent", "intent_spec_chars": len(spec)}
                    return fixed
            LAST_META = {"route": "intent->fallback"}
            print("   intent stage empty - falling back to thin repair")
            return _repair_call(llm, model_name, _thin_prompt(), buggy_code)

        # Baseline
        LAST_META = {"route": "baseline" if not tools else "thin"}
        return _repair_call(llm, model_name, _thin_prompt(), buggy_code)

    except Exception as e:
        LAST_META = {"route": "error", "error": str(e)}
        print(f"Agent Execution Error: {e}")
        return ""
