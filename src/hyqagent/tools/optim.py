import ast
import io
import json
import contextlib
import numpy as np
from unittest.mock import patch
import pennylane as qp


class _ProfilingBudgetReached(Exception):
    """ Stop a profiled script once the QNode-call budget is hit.
    """

# Names commonly bound to trainable variational parameters
_PARAM_NAMES = {"params", "init_params", "init_param", "weights", "weight",
                "theta", "thetas", "w", "phi", "var", "vqe_params"}
# Names commonly used for the optimizer step size / learning rate
_LR_NAMES = {"stepsize", "step_size", "learning_rate", "lr", "eta"}
# Names commonly used for ansatz depth / layer count
_DEPTH_NAMES = {"n_layers", "num_layers", "layers", "depth", "n_depth", "n_layer"}


def extract_numpy(val):
    try:
        return np.array(qp.math.unwrap(val), dtype=float).flatten()
    except Exception:
        if hasattr(val, "detach"):
            try:
                return val.detach().cpu().numpy().astype(float).flatten()
            except Exception:
                return np.array([])
        if hasattr(val, "_value"):
            try:
                return np.array(val._value, dtype=float).flatten()
            except Exception:
                return np.array([])
        try:
            return np.array(val, dtype=float).flatten()
        except Exception:
            return np.array([])


def _static_scan(script_code: str):
    """
    AST pass for optimization bugs that a dynamic profile cannot see
    """

    insights = []
    facts = {}
    try:
        tree = ast.parse(script_code)
    except SyntaxError:
        return insights, facts

    def _num(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = _num(node.operand)
            return -inner if inner is not None else None
        return None

    def _has_requires_grad_true(call):
        for kw in call.keywords:
            if kw.arg == "requires_grad" and isinstance(kw.value, ast.Constant):
                return kw.value.value is True
        return None  

    def _is_zeros_call(node):
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "zeros"

    depth_flagged = False

    # Map simple `name = <number>`
    assign_map = {}
    for n_ in ast.walk(tree):
        if isinstance(n_, ast.Assign) and len(n_.targets) == 1 and isinstance(n_.targets[0], ast.Name):
            v = _num(n_.value)
            if v is not None:
                assign_map[n_.targets[0].id] = v

    def _resolve(node):
        v = _num(node)
        if v is not None:
            return v
        if isinstance(node, ast.Name) and node.id in assign_map:
            return assign_map[node.id]
        return None

    for node in ast.walk(tree):
        # Zero-init parameters => symmetric saddle point
        if isinstance(node, ast.Assign):
            target_names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            val = node.value
            if _is_zeros_call(val):
                rg = _has_requires_grad_true(val)
                looks_trainable = bool(target_names & _PARAM_NAMES)
                if rg is True or (rg is None and looks_trainable):
                    insights.append(
                        "WARNING: Trainable parameters are initialised to all zeros. "
                        "This sits on a symmetric saddle point where every gradient is "
                        "identical (often exactly zero), so the optimizer cannot break "
                        "symmetry and never moves. Initialise with small random values "
                        "(e.g. np.random.normal(0, 0.1, shape) or 0.01 * np.random.randn).")

            # Learning-rate / step-size assigned to a literal
            if target_names & _LR_NAMES:
                n = _num(val)
                if n is not None:
                    facts.setdefault("learning_rate", n)

            # Ansatz depth assigned to a literal 
            if target_names & _DEPTH_NAMES and not depth_flagged:
                n = _num(val)
                if n is not None and n >= 10:
                    depth_flagged = True
                    insights.append(
                        f"WARNING: The ansatz depth is very large ({int(n)} layers). Deep "
                        "hardware-efficient ansaetze suffer from barren plateaus: the gradient "
                        "variance vanishes exponentially with depth and training stalls. Reduce "
                        "the number of layers to a shallow value (typically 2-4).")

        # Learning-rate / step-size passed as an optimizer keyword
        if isinstance(node, ast.keyword) and node.arg in _LR_NAMES:
            n = _resolve(node.value)
            if n is not None:
                facts.setdefault("learning_rate", n)

        # Ansatz depth passed as a keyword
        if isinstance(node, ast.keyword) and node.arg in _DEPTH_NAMES and not depth_flagged:
            n = _resolve(node.value)
            if n is not None and n >= 10:
                depth_flagged = True
                insights.append(
                    f"WARNING: The ansatz depth is very large ({int(n)} layers). Deep "
                    "hardware-efficient ansaetze suffer from barren plateaus: the gradient "
                    "variance vanishes exponentially with depth and training stalls. Reduce "
                    "the number of layers to a shallow value (typically 2-4).")

        # SPSA optimizer
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and "SPSA" in node.func.attr:
            for kw in node.keywords:
                if kw.arg == "a":
                    n = _num(kw.value)
                    if n is not None:
                        facts.setdefault("learning_rate", n)

    # Evaluate lr against bounds
    lr = facts.get("learning_rate")
    if lr is not None:
        if lr <= 1e-3:
            insights.append(
                f"WARNING: The optimizer step size / learning rate is extremely small ({lr}). "
                "With the limited number of training steps in this script the model cannot make "
                "meaningful progress and will fail to converge. Raise it to a normal range "
                "(roughly 1e-2 to 5e-1).")
        elif lr >= 2.0:
            insights.append(
                f"WARNING: The optimizer step size / learning rate is far too large ({lr}). "
                "Steps this big overshoot the minimum, causing the loss to diverge or oscillate "
                "(often to NaN). Lower it to a normal range (roughly 1e-2 to 5e-1).")

    return insights, facts


def profile(script_code: str) -> str:
    """
    Scan for optimization bugs, then dynamically execute the script
    """

    report = {
        "qnode_executions": 0,
        "dynamic_status": "not run",
        "output_trajectory_variance": None,
        "agent_insights": [],
    }

    # Static analysis
    static_insights, static_facts = _static_scan(script_code)
    if static_facts.get("learning_rate") is not None:
        report["detected_learning_rate"] = static_facts["learning_rate"]

    # Dynamic profiling
    output_history = []
    execs = [0]
    original_call = qp.QNode.__call__

    def intercept_call(self, *args, **kwargs):
        execs[0] += 1
        result = original_call(self, *args, **kwargs)
        try:
            arr = extract_numpy(result)
            if arr.size:
                output_history.append(float(np.mean(arr)))
        except Exception:
            pass
        if execs[0] >= 60:
            raise _ProfilingBudgetReached()
        return result

    try:
        with patch.object(qp.QNode, "__call__", intercept_call):
            exec_globals = {}
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                exec(script_code, exec_globals)
        report["dynamic_status"] = "completed"
    except _ProfilingBudgetReached:
        report["dynamic_status"] = "profiled (budget reached)"
    except Exception as e:
        report["dynamic_status"] = f"script raised {type(e).__name__}: {str(e)[:120]}"

    report["qnode_executions"] = execs[0]

    dynamic_insights = []
    outputs = np.array(output_history, dtype=float)

    if outputs.size >= 5:
        finite = outputs[np.isfinite(outputs)]
        if finite.size < outputs.size:
            dynamic_insights.append(
                "WARNING: The circuit output became NaN/inf during execution. The optimization "
                "is diverging - almost always an over-large step size or an unstable cost.")
        else:
            var = float(np.var(finite))
            report["output_trajectory_variance"] = var

            hybrid_classical = any(k in script_code for k in ("torch", "tensorflow", "keras"))
            if var < 1e-12 and not hybrid_classical:
                dynamic_insights.append(
                    "WARNING: The circuit output is completely frozen (zero variance across "
                    f"{outputs.size} executions). Nothing is changing during training. Three "
                    "families of cause: (1) a severed autograd chain (2) the update is computed but "
                    "never applied (3) the circuit is structurally unable to move the measured "
                    "observable. Check that the parameters reach the QNode, that "
                    "`opt.step` output is reassigned, AND that every parameter drives a gate that "
                    "changes the measured expectation.")
    elif execs[0] <= 2:
        report["dynamic_status"] += " | dynamic profiling unavailable (jitted or early exit); rely on static analysis and code reasoning."

    report["agent_insights"] = static_insights + dynamic_insights
    return json.dumps(report, indent=2)
