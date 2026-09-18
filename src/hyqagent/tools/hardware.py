import ast
import json

# Two-qubit / entangling operations
_ENTANGLERS = {"CNOT", "CZ", "CY", "CH", "CSWAP", "Toffoli", "SWAP", "ISWAP",
               "IsingXX", "IsingYY", "IsingZZ", "IsingXY",
               "CRX", "CRY", "CRZ", "CRot", "ControlledPhaseShift", "MultiControlledX"}
# Templates whose class name should be reported
_TEMPLATE_HINTS = ("Layers", "Embedding", "Ansatz", "Prep", "Encoding")


def analyse_hardware_topology(script_code: str) -> str:
    """Report the device, wires, gate set and entangling-gate usage.
    """

    try:
        tree = ast.parse(script_code)
    except SyntaxError as e:
        return json.dumps({"error": f"Invalid syntax: {e}"})

    report = {
        "device": "Unknown",
        "wires": "Unknown",
        "detected_gates": set(),
        "templates_used": set(),
        "entangling_gate_count": 0,
        "agent_insights": [],
    }
    entangler_counts = {}

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node):

            # Device declaration
            if isinstance(node.func, ast.Attribute) and node.func.attr == "device":
                if node.args and isinstance(node.args[0], ast.Constant):
                    report["device"] = node.args[0].value
                for kw in node.keywords:
                    if kw.arg == "wires" and isinstance(kw.value, ast.Constant):
                        report["wires"] = kw.value.value

            # Gate/template usage
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id in ("qp", "qml", "pnp"):
                op_name = node.func.attr
                if any(h in op_name for h in _TEMPLATE_HINTS):
                    report["templates_used"].add(op_name)
                elif op_name[:1].isupper():
                    report["detected_gates"].add(op_name)
                    if op_name in _ENTANGLERS:
                        entangler_counts[op_name] = entangler_counts.get(op_name, 0) + 1
            self.generic_visit(node)

    Visitor().visit(tree)

    report["detected_gates"] = sorted(report["detected_gates"])
    report["templates_used"] = sorted(report["templates_used"])
    report["entangling_gate_count"] = sum(entangler_counts.values())
    report["entangling_gate_breakdown"] = entangler_counts

    insights = []

    # Native-gate directive
    lowered = script_code.lower()
    wants_native_ising = ("isingxx" in lowered) and (
        "trapped-ion" in lowered or "trapped ion" in lowered
        or "ion trap" in lowered or "native" in lowered)
    if wants_native_ising and "CNOT" in entangler_counts:
        insights.append(
            "WARNING: The target hardware requires the native IsingXX entangler, but the circuit "
            "uses CNOT. Replace each CNOT with an IsingXX-based entangler so the transpiled circuit "
            "contains IsingXX and no CNOT.")

    # Factual gate-set note
    elif "CNOT" in entangler_counts and any(g.startswith("Ising") for g in entangler_counts):
        insights.append(
            "NOTE: The circuit mixes CNOT with native Ising-type entanglers. If the target "
            "architecture requires a specific native gate set, make the entangling gates "
            "consistent with it.")

    report["agent_insights"] = insights
    return json.dumps(report, indent=2)
