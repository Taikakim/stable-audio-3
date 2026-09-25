"""train_lora_modular.py passes the melody-subspace loss flags through to the training wrapper.

The loss itself lives in DiffusionCondTrainingWrapper (which ModularTrainingWrapper inherits); the
modular trainer used to drop it on the floor. The parser is built inline in main(), so this checks
the source: both flags are declared, and the wrapper call forwards both to the matching kwargs.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "train_lora_modular.py"


def _tree():
    return ast.parse(SRC.read_text())


def test_flags_declared_with_the_train_lora_dests():
    dests = {}
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
            for kw in node.keywords:
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    for f in flags:
                        dests[f] = kw.value.value
    assert dests.get("--subspace-loss-basis") == "subspace_loss_basis"
    assert dests.get("--subspace-loss-weight") == "subspace_loss_weight"
    # the underscore spellings used by train_lora.py / the LUMI sbatch keep working
    assert dests.get("--subspace_loss_basis") == "subspace_loss_basis"
    assert dests.get("--subspace_loss_weight") == "subspace_loss_weight"


def test_wrapper_call_forwards_both():
    calls = [n for n in ast.walk(_tree()) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "ModularTrainingWrapper"]
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kw.get("subspace_loss_basis") == "args.subspace_loss_basis"
    assert kw.get("subspace_loss_weight") == "args.subspace_loss_weight"
