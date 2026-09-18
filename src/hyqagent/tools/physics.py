import ast
import json


def evaluate_symmetries(script_code: str) -> str:
    try:
        tree = ast.parse(script_code)
    except SyntaxError as e:
        return json.dumps({"error": f"Syntax error in code: {e}"})

    report = {
        "problem_domain": "Unknown",
        "state_preparation": set(),
        "ansatz_templates": set(),
        "measurements": set(),
        "interface_violations": set(),
        "device_shots": None,
        "agent_insights": [],
    }

    def _num(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        return None

    for node in ast.walk(tree):
        # Problem domain
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "qchem" in alias.name:
                    report["problem_domain"] = "Quantum Chemistry"
        elif isinstance(node, ast.ImportFrom):
            if getattr(node, "module", None) and "qchem" in node.module:
                report["problem_domain"] = "Quantum Chemistry"
            for alias in node.names:
                if "qchem" in alias.name:
                    report["problem_domain"] = "Quantum Chemistry"

        # Device shots
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "device":
            for kw in node.keywords:
                if kw.arg == "shots":
                    n = _num(kw.value)
                    if n is not None:
                        report["device_shots"] = n

        identifier = ""
        if isinstance(node, ast.Attribute):
            identifier = node.attr
        elif isinstance(node, ast.Name):
            identifier = node.id

        if identifier in ("molecular_hamiltonian", "Molecule"):
            report["problem_domain"] = "Quantum Chemistry"
        if identifier in ("BasisState", "StatePrep", "QubitStateVector"):
            report["state_preparation"].add(identifier)
        if identifier in ("StronglyEntanglingLayers", "BasicEntanglerLayers", "RandomLayers"):
            report["ansatz_templates"].add(identifier)
        if identifier in ("expval", "var", "probs", "sample", "state", "density_matrix"):
            report["measurements"].add(identifier)
        if identifier in ("detach", "stop_gradient"):
            report["interface_violations"].add(identifier)

    for key in ("state_preparation", "ansatz_templates", "measurements", "interface_violations"):
        report[key] = sorted(report[key])

    insights = []

    # Chemistry symmetry
    if report["problem_domain"] == "Quantum Chemistry":
        if "BasisState" not in report["state_preparation"]:
            insights.append(
                "WARNING: Chemistry problem detected but no 'BasisState' preparation found. The "
                "model must initialise the correct electron configuration (e.g. Hartree-Fock) to "
                "conserve particle number. Use the top-level op `qml.BasisState(hf_state, "
                "wires=range(qubits))` (in PennyLane 0.45 it is `qml.BasisState`, NOT "
                "`qml.templates.BasisState`, which no longer exists); build hf_state with "
                "`qml.qchem.hf_state(electrons, qubits)`.")
        if "StronglyEntanglingLayers" in report["ansatz_templates"]:
            insights.append(
                "WARNING: 'StronglyEntanglingLayers' breaks particle-number conservation "
                "(Z-symmetry). In chemistry problems this lets the optimizer explore unphysical "
                "sub-spaces. Use symmetry-preserving gates (DoubleExcitation / controlled "
                "rotations) instead.")

    # Missing measurement
    if not report["measurements"]:
        insights.append(
            "WARNING: No quantum measurement (expval/probs/...) detected. The Hamiltonian mapping "
            "or the QNode return may be broken or missing.")

    # Severed autograd chain
    if report["interface_violations"]:
        violations = ", ".join(report["interface_violations"])
        insights.append(
            f"WARNING: The code severs the automatic-differentiation chain using [{violations}]. "
            "Gradients cannot flow back to the trainable parameters, halting all learning. Remove "
            "these calls in the optimization path.")

    # Shot noise
    shots = report["device_shots"]
    if shots is not None and shots <= 20:
        insights.append(
            f"WARNING: The device uses only shots={shots}. So few measurement samples produce huge "
            "statistical (shot) noise in every expectation value, so the cost is dominated by "
            "noise and cannot converge. Increase shots substantially (e.g. 1000+) or use analytic "
            "mode with shots=None. In PennyLane 0.45 `qml.device(..., shots=N)` is still valid; if "
            "you prefer the modern API use the `qml.set_shots(qnode, shots=N)` transform. Do NOT "
            "invent other shot APIs.")

    report["agent_insights"] = insights
    return json.dumps(report, indent=2)
