"""Test the trainer's WSD function without importing its training entrypoint."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('power', [1.0, 1.2])
def test_power_warmdown_preserves_warmup_plateau_and_endpoint(power):
    source = Path(__file__).resolve().parents[1] / 'scripts/base_train.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'get_lr_multiplier')
    scope = {'args': SimpleNamespace(warmup_steps=125, warmdown_ratio=.25,
                                    final_lr_frac=.05, warmdown_power=power),
             'num_iterations': 3000}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
    lr = scope['get_lr_multiplier']
    assert lr(0) == pytest.approx(1 / 125)
    assert lr(124) == lr(2250) == 1.0
    assert lr(2625) == pytest.approx(.05 + .95 * .5 ** power)
    assert lr(3000) == .05
    assert all(lr(step) >= lr(step + 1) for step in range(2250, 3000))
