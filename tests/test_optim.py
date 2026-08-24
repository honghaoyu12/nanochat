"""
Test the combined MuonAdamW optimizer (single rank, no process group).
The AdamW half is checked against torch.optim.AdamW as a reference;
the Muon half is checked behaviorally (determinism + convergence).

Requires a GPU (the fused kernels are compiled on first use, which
dominates the runtime of this file).

python -m pytest tests/test_optim.py -v
"""

import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="optimizer tests require CUDA")

if cuda_available:
    from nanochat.optim import KELLER_NS_COEFFICIENTS, MuonAdamW

DEVICE = "cuda"

# keep shapes small and reuse them across tests so the fused kernels compile once
ADAMW_SMALL_SHAPE = (48,)        # numel < 1024 path
ADAMW_LARGE_SHAPE = (1024, 4)    # numel >= 1024 path
MUON_WIDE_SHAPE = (32, 64)
MUON_TALL_SHAPE = (64, 32)


def make_params_and_groups(seed=1337, offset=0.0, targets=None):
    """The menagerie: small/large AdamW params, wide/tall Muon stacks."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    def rand(shape):
        base = torch.randn(shape, generator=gen, device=DEVICE) * 0.05
        return torch.nn.Parameter(base + offset)
    params = [rand(ADAMW_SMALL_SHAPE), rand(ADAMW_LARGE_SHAPE)]
    params += [rand(MUON_WIDE_SHAPE) for _ in range(3)]
    params += [rand(MUON_TALL_SHAPE) for _ in range(2)]
    groups = [
        dict(kind="adamw", params=params[0:1], lr=0.02, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", params=params[1:2], lr=0.02, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.0),
        dict(kind="muon", params=params[2:5], lr=0.02, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0),
        dict(kind="muon", params=params[5:7], lr=0.02, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0),
    ]
    return params, groups


def test_adamw_matches_torch_reference():
    """Our fused AdamW must agree with torch.optim.AdamW (both decoupled wd)."""
    hypers = dict(lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    p_ours = torch.nn.Parameter(torch.randn(64, 32, device=DEVICE))
    p_ref = torch.nn.Parameter(p_ours.detach().clone())
    opt_ours = MuonAdamW([dict(kind="adamw", params=[p_ours], **hypers)])
    opt_ref = torch.optim.AdamW([p_ref], **hypers)
    for step in range(10):
        grad = torch.randn(p_ours.shape, generator=torch.Generator(device=DEVICE).manual_seed(step), device=DEVICE)
        p_ours.grad = grad.clone()
        p_ref.grad = grad.clone()
        opt_ours.step()
        opt_ref.step()
    torch.testing.assert_close(p_ours, p_ref, rtol=1e-5, atol=1e-6)


def test_determinism():
    """Two identical runs must produce bitwise identical parameters."""
    results = []
    for _ in range(2):
        params, groups = make_params_and_groups()
        opt = MuonAdamW(groups)
        for step in range(5):
            gen = torch.Generator(device=DEVICE).manual_seed(step)
            for p in params:
                p.grad = torch.randn(p.shape, generator=gen, device=DEVICE) * 0.01
            opt.step()
        results.append([p.detach().clone() for p in params])
    for pa, pb in zip(*results):
        assert torch.equal(pa, pb)


def test_convergence():
    """Optimizing distance-to-target must actually approach the target."""
    targets_params, _ = make_params_and_groups(seed=999)
    targets = [p.detach().clone() for p in targets_params]
    params, groups = make_params_and_groups(seed=999, offset=0.1) # start offset from the targets
    opt = MuonAdamW(groups)
    def distances():
        return [(p.detach() - t).norm().item() for p, t in zip(params, targets)]
    initial = distances()
    for _ in range(50):
        for p, t in zip(params, targets):
            p.grad = 2 * (p.detach() - t) # gradient of ||p - t||^2
        opt.step()
    final = distances()
    for i, (d0, d1) in enumerate(zip(initial, final)):
        assert d1 < 0.5 * d0, f"param {i} did not converge: {d0:.4f} -> {d1:.4f}"
        assert torch.isfinite(params[i]).all()


def test_muon_update_is_orthogonalized():
    """
    The very first Muon update (zero momentum state) of a full-rank gradient
    should be (near) semi-orthogonal after the polar iteration: its nonzero
    singular values land in a band around 1, rather than being spread out.
    """
    p = torch.nn.Parameter(torch.zeros(MUON_WIDE_SHAPE, device=DEVICE))
    group = dict(kind="muon", params=[p], lr=1.0, momentum=0.0, ns_steps=5, beta2=1.0, weight_decay=0.0)
    opt = MuonAdamW([group])
    p.grad = torch.randn(p.shape, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)
    opt.step()
    # with lr=1, wd=0: p_new = -update, so the update is just -p
    svals = torch.linalg.svdvals(-p.detach().float())
    # NorMuon variance reduction rescales the magnitude, so normalize by the mean
    svals = svals / svals.mean()
    assert svals.max() / svals.min() < 4.0, f"update far from semi-orthogonal: {svals}"
@torch.no_grad()
def keller_reference_step(p, grad, momentum_buffer, lr, momentum, weight_decay, ns_steps):
    momentum_buffer.lerp_(grad, 1 - momentum)
    g = grad.lerp(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    a, b, c = KELLER_NS_COEFFICIENTS
    for _ in range(ns_steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    p.mul_(1 - lr * weight_decay)
    shape_scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
    p.sub_(lr * shape_scale * X.to(p.dtype))


@pytest.mark.parametrize("shape", [MUON_WIDE_SHAPE, MUON_TALL_SHAPE])
def test_keller_backend_matches_pinned_reference(shape):
    """The opt-in backend must match the pinned Keller update on both orientations."""
    gen = torch.Generator(device=DEVICE).manual_seed(20260820 + sum(shape))
    p = torch.nn.Parameter(torch.randn(shape, generator=gen, device=DEVICE) * 0.05)
    p_ref = torch.nn.Parameter(p.detach().clone())
    group = dict(kind="muon", params=[p], lr=0.02, momentum=0.9, ns_steps=5, beta2=0.9, weight_decay=0.07)
    opt = MuonAdamW([group], muon_backend="keller_original")
    momentum_ref = torch.zeros_like(p)
    for step in range(4):
        lr = 0.02 / (step + 1)
        momentum = 0.90 + 0.01 * step
        weight_decay = 0.07 + 0.01 * step
        group["lr"] = lr
        group["momentum"] = momentum
        group["weight_decay"] = weight_decay
        grad = torch.randn(shape, generator=torch.Generator(device=DEVICE).manual_seed(step), device=DEVICE)
        p.grad = grad.clone()
        opt.step()
        keller_reference_step(p_ref, grad, momentum_ref, lr, momentum, weight_decay, 5)
    torch.testing.assert_close(p, p_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(opt.state[p]["momentum_buffer"][0], momentum_ref, rtol=2e-6, atol=2e-6)
    assert "second_momentum_buffer" not in opt.state[p]


def test_unknown_muon_backend_rejected():
    p = torch.nn.Parameter(torch.zeros(MUON_WIDE_SHAPE, device=DEVICE))
    group = dict(kind="muon", params=[p], lr=0.02, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0)
    with pytest.raises(ValueError, match="unknown Muon backend"):
        MuonAdamW([group], muon_backend="not-a-backend")


def test_adamw_path_is_identical_across_muon_backends():
    """Keller selection must not alter the shared AdamW implementation."""
    p_enhanced = torch.nn.Parameter(torch.randn(48, device=DEVICE, generator=torch.Generator(device=DEVICE).manual_seed(7)))
    p_keller = torch.nn.Parameter(p_enhanced.detach().clone())
    group_enhanced = dict(kind="adamw", params=[p_enhanced], lr=0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.1)
    group_keller = dict(kind="adamw", params=[p_keller], lr=0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.1)
    opt_enhanced = MuonAdamW([group_enhanced], muon_backend="nanochat_enhanced")
    opt_keller = MuonAdamW([group_keller], muon_backend="keller_original")
    for step in range(4):
        grad = torch.randn(48, generator=torch.Generator(device=DEVICE).manual_seed(900 + step), device=DEVICE)
        p_enhanced.grad = grad.clone()
        p_keller.grad = grad.clone()
        opt_enhanced.step()
        opt_keller.step()
    torch.testing.assert_close(p_enhanced, p_keller, rtol=1e-5, atol=1e-6)
