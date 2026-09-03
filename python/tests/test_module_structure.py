"""Structural guards against refactor regressions.

The 0.7.0 refactor of InferenceClient into module-level codec functions
shipped three latent bugs that unit tests missed because each parse path was
only exercised with an empty response:

1. Module-level codec functions kept a stray ``self`` parameter
   (``_parse_infer_response`` shipped in 0.7.0/0.7.1, ``_tensor_to_numpy``
   and ``_parse_post_result`` still present in 0.7.2).
2. The class kept BOTH the full method implementations and the thin codec
   delegates, with the delegates defined later — so the dead earlier copies
   silently shadowed nothing while drifting from the codec.

These tests parse the package source so none of those patterns can ship again.
"""

import ast
import pathlib

import neoruntime_ipc_sdk

PACKAGE_ROOT = pathlib.Path(neoruntime_ipc_sdk.__file__).parent


def _is_property_setter(node):
    return any(
        isinstance(decorator, ast.Attribute) and decorator.attr == "setter"
        for decorator in node.decorator_list
    )


def _collect_problems():
    problems = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first_arg = node.args.args[0].arg if node.args.args else None
                if first_arg in ("self", "cls"):
                    problems.append(
                        f"{path.name}:{node.lineno} module-level def "
                        f"{node.name}() takes '{first_arg}'"
                    )

        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            seen = {}
            for item in cls.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if item.name in seen and not _is_property_setter(item):
                    problems.append(
                        f"{path.name}:{item.lineno} class {cls.name} redefines "
                        f"{item.name}() (first definition at line {seen[item.name]})"
                    )
                seen[item.name] = item.lineno
    return problems


def test_no_module_level_function_takes_self_or_cls():
    # A module-level function taking `self` only works when called as a
    # (unbound) method — every direct codec call raises TypeError.
    problems = [p for p in _collect_problems() if "module-level def" in p]
    assert problems == []


def test_no_class_shadows_its_own_methods():
    # Two same-named defs in one class body: the later silently wins, so the
    # earlier copy is dead code that drifts (the exact 0.7.x failure mode).
    # Property setters are the one legitimate repeat.
    problems = [p for p in _collect_problems() if "redefines" in p]
    assert problems == []
