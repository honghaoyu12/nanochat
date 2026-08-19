"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
import csv
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.controlled_muon import (
    ABSOLUTE_GROUP_SCALING_MODES,
    CONTROL_ACTION_POLICIES,
    CONTROL_FEEDBACK_SCOPES,
    PHASE_HOLD_CRUISE_POLICIES,
    NanochatMuonController,
    LossProgressRhoReference,
    ThreeStageLossProgressRhoReference,
    absolute_group_lr_scale,
    control_feedback_state_dict,
    select_control_feedback,
    validate_absolute_group_scaling,
    validate_control_feedback_configuration,
    validate_resumed_control_feedback,
)
from nanochat.control_governors import (
    AUTONOMOUS_COOLDOWN_VARIANTS,
    AlphaCeilingGovernor,
    GovernedMuonController,
    ValidationProgressDetector,
)
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

BASE_CONTROLLED_MUON_VARIANTS = set(NanochatMuonController.VALID_VARIANTS)
AUTONOMOUS_COOLDOWN_MUON_VARIANTS = set(AUTONOMOUS_COOLDOWN_VARIANTS)
CONTROLLED_MUON_VARIANTS = sorted(
    BASE_CONTROLLED_MUON_VARIANTS | AUTONOMOUS_COOLDOWN_MUON_VARIANTS
)
OPTIMIZER_VARIANTS = ["native_muon", "torch_muon"] + CONTROLLED_MUON_VARIANTS

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--seed", type=int, default=42, help="global model and CUDA RNG seed")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
parser.add_argument("--max-runtime-minutes", type=float, default=-1.0, help="gracefully stop after this many wallclock minutes (-1 = disable)")
parser.add_argument("--stop-after-steps", type=int, default=-1, help="cleanly stop at this optimizer step while keeping num_iterations as the scheduler horizon (-1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Controlled Muon research baseline
parser.add_argument("--optimizer-variant", type=str, default="native_muon", choices=OPTIMIZER_VARIANTS, help="optimizer/controller variant")
parser.add_argument("--controlled-muon", action="store_true", help="enable controlled Muon every-step probe path")
parser.add_argument("--control-alpha-mode", type=str, default="absolute", choices=["absolute", "multiplier"], help="absolute alpha replaces controlled group lr; multiplier scales native scheduled controlled group lr")
parser.add_argument("--control-alpha-reference", type=str, default="none", choices=["none", "native_wsd"], help="alpha reference mode; none means alpha is an absolute actuator")
parser.add_argument("--control-reference-factor-init", type=float, default=1.0, help="initial multiplier for native_wsd ablations")
parser.add_argument("--control-reference-factor-min", type=float, default=0.8, help="minimum multiplier for native_wsd ablations")
parser.add_argument("--control-reference-factor-max", type=float, default=1.2, help="maximum multiplier for native_wsd ablations")
parser.add_argument("--control-start-step", type=int, default=-1, help="first controller probe/update step; -1 uses the alpha warmup boundary")
parser.add_argument("--control-scope", type=str, default="muon_only", choices=["muon_only", "all_groups"], help="scope for alpha control: Muon matrix groups only, or all optimizer groups")
parser.add_argument("--control-feedback-scope", type=str, default="total", choices=sorted(CONTROL_FEEDBACK_SCOPES), help="feedback observation used by the alpha controller")
parser.add_argument("--control-absolute-group-scaling", type=str, default="uniform", choices=sorted(ABSOLUTE_GROUP_SCALING_MODES), help="uniform assigns alpha directly; initial_lr_ratio preserves native group LR ratios around Muon alpha")
parser.add_argument("--control-period-steps", type=int, default=1, help="controller probe period; use 1 for every-step baseline")
parser.add_argument("--control-probe-scope", type=str, default="full_accum_batch", choices=["full_accum_batch", "last_microbatch"], help="tokens used for before/after probe loss")
parser.add_argument("--control-alpha-warmup-steps", type=int, default=0, help="linearly warm up applied absolute alpha over this many steps")
parser.add_argument("--control-alpha-init", type=float, default=-1.0, help="initial alpha; -1 uses matrix_lr after batch scaling")
parser.add_argument("--control-alpha-min", type=float, default=-1.0, help="minimum alpha; -1 uses 0.25 * alpha_init")
parser.add_argument("--control-alpha-max", type=float, default=-1.0, help="maximum alpha; -1 uses 2.0 * alpha_init")
parser.add_argument("--control-alpha-replay-file", type=str, default="", help="reserved absolute alpha replay source")
parser.add_argument("--control-residual-validity-gate", action="store_true", help="reject low-observability Muon residual probes before rho EMA")
parser.add_argument("--control-residual-min-muon-predicted-fraction", type=float, default=0.10, help="minimum predicted Muon fraction for residual feedback validity")
parser.add_argument("--control-residual-max-adamw-predicted-fraction", type=float, default=0.90, help="maximum predicted AdamW fraction for residual feedback validity")
parser.add_argument("--control-startup-alpha-reference-ratio", type=float, default=1.0, help="opt-in autonomous startup alpha target as a multiple of alpha init; 1 disables")
parser.add_argument("--control-startup-alpha-reference-gain", type=float, default=0.5, help="startup alpha reference gain in log-alpha space")
parser.add_argument("--control-startup-one-sided-safety", action="store_true", help="during autonomous startup, suppress non-emergency downward corrections")
parser.add_argument("--control-startup-monotone", action="store_true", help="during autonomous startup, enforce a monotone alpha envelope")
parser.add_argument("--control-startup-emergency-rho", type=float, default=-0.25, help="trusted rho below which startup safety may reduce alpha")
parser.add_argument("--control-startup-kp", type=float, default=-1.0, help="startup proportional gain for three-stage rho reference; -1 uses control-kp")
parser.add_argument("--control-startup-factor-max", type=float, default=-1.0, help="startup factor ceiling for three-stage rho reference; -1 uses control-factor-max")
parser.add_argument("--control-action-policy", type=str, default="legacy", choices=sorted(CONTROL_ACTION_POLICIES), help="legacy signed controller or opt-in phase-specific action authority")
parser.add_argument("--control-phase-hold-cruise-policy", type=str, default="hold", choices=sorted(PHASE_HOLD_CRUISE_POLICIES), help="exact alpha hold or low-authority rho deadband during cruise")
parser.add_argument("--control-phase-hold-cruise-kp", type=float, default=0.0, help="cruise rho-deadband gain for phase_hold")
parser.add_argument("--control-phase-hold-cruise-deadband", type=float, default=0.05, help="half-width of the phase_hold cruise rho deadband")
parser.add_argument("--control-phase-hold-late-kp", type=float, default=0.03, help="one-sided downward late gain for phase_hold")
parser.add_argument("--control-phase-hold-late-exponent", type=float, default=2.0, help="exponent applied to autonomous late-phase authority")
parser.add_argument("--control-recovery-terminal-ratio", type=float, default=1.0, help="terminal alpha cap as a fraction of the frozen pre-late peak")
parser.add_argument("--control-recovery-exponent", type=float, default=4.0, help="exponent shaping monotone late-phase recovery progress")
parser.add_argument("--control-rho-star", type=float, default=0.7, help="target rho")
parser.add_argument("--control-rho-reference", type=str, default="fixed", choices=["fixed", "loss_progress", "loss_progress_three_stage"], help="fixed, two-stage, or three-stage loss-progress rho target")
parser.add_argument("--control-rho-start", type=float, default=None, help="startup rho target for loss_progress_three_stage reference")
parser.add_argument("--control-rho-cruise", type=float, default=None, help="cruise rho target for loss_progress_three_stage reference")
parser.add_argument("--control-rho-middle", type=float, default=None, help="middle-phase rho target for loss_progress reference")
parser.add_argument("--control-rho-late", type=float, default=None, help="late-phase rho target for loss_progress reference")
parser.add_argument("--control-rho-startup-fast-beta", type=float, default=0.70, help="fast loss EMA beta for three-stage startup transition")
parser.add_argument("--control-rho-startup-slow-beta", type=float, default=0.95, help="slow loss EMA beta for three-stage startup transition")
parser.add_argument("--control-rho-startup-reference-beta", type=float, default=0.995, help="progress-reference beta for three-stage startup transition")
parser.add_argument("--control-rho-startup-phase-beta", type=float, default=0.90, help="phase smoothing beta for three-stage startup transition")
parser.add_argument("--control-rho-startup-ratio-high", type=float, default=0.80, help="progress ratio at startup phase zero")
parser.add_argument("--control-rho-startup-ratio-low", type=float, default=0.40, help="progress ratio at startup phase one")
parser.add_argument("--control-rho-startup-min-observations", type=int, default=10, help="loss observations required before startup phase can advance")
parser.add_argument("--control-rho-progress-fast-beta", type=float, default=0.90, help="fast loss EMA beta for phased rho reference")
parser.add_argument("--control-rho-progress-slow-beta", type=float, default=0.99, help="slow loss EMA beta for phased rho reference")
parser.add_argument("--control-rho-progress-reference-beta", type=float, default=0.999, help="decay beta for observed progress scale")
parser.add_argument("--control-rho-phase-beta", type=float, default=0.99, help="smoothing beta for phased rho target")
parser.add_argument("--control-rho-progress-ratio-high", type=float, default=0.50, help="progress ratio at phase zero")
parser.add_argument("--control-rho-progress-ratio-low", type=float, default=0.10, help="progress ratio at phase one")
parser.add_argument("--control-rho-progress-min-observations", type=int, default=50, help="loss observations required before phase can advance")
parser.add_argument("--control-kp", type=float, default=1.0, help="P-controller gain")
parser.add_argument("--control-ki", type=float, default=0.0, help="I-controller gain for PI/PID variants; must be >0 for PI/PID")
parser.add_argument("--control-kd", type=float, default=0.0, help="D-controller gain for PID variants; must be >0 for PID")
parser.add_argument("--control-rho-beta", type=float, default=0.9, help="rho EMA beta")
parser.add_argument("--control-integral-beta", type=float, default=0.95, help="EMA decay for PI/PID integral state")
parser.add_argument("--control-integral-clip", type=float, default=10.0, help="absolute clamp for PI/PID integral state")
parser.add_argument("--control-derivative-beta", type=float, default=0.0, help="EMA decay for PID derivative state")
parser.add_argument("--control-factor-min", type=float, default=0.9, help="minimum per-step alpha factor")
parser.add_argument("--control-factor-max", type=float, default=1.1, help="maximum per-step alpha factor")
parser.add_argument("--control-rho-clip-min", type=float, default=-1.0, help="minimum clipped rho for controller")
parser.add_argument("--control-rho-clip-max", type=float, default=3.0, help="maximum clipped rho for controller")
parser.add_argument("--control-trust-rho-threshold", type=float, default=0.9, help="rho threshold for trust-region expansion")
parser.add_argument("--control-trust-alpha-threshold", type=float, default=1e-4, help="alpha threshold for trust-region expansion")
parser.add_argument("--control-trust-expand-factor", type=float, default=1.5, help="trust-region expansion factor")
parser.add_argument("--control-trust-max-factor", type=float, default=1.5, help="hard maximum factor when trust expansion is active")
parser.add_argument("--control-trust-patience", type=int, default=2, help="good-rho count before trust-region expansion")
parser.add_argument("--control-alignment-aware", action="store_true", help="apply alignment-aware safety to the selected P/PI/PID controller")
parser.add_argument("--control-alignment-c-min", type=float, default=0.02, help="minimum alignment for trust-region expansion")
parser.add_argument("--control-alignment-c-bad", type=float, default=-0.01, help="alignment below this threshold triggers bad-step shrinkage")
parser.add_argument("--control-alignment-penalty", type=float, default=0.10, help="log-alpha penalty per unit below c-min")
parser.add_argument("--control-alignment-bad-step-shrink", type=float, default=0.5, help="next-alpha factor for invalid or strongly bad aligned steps")
parser.add_argument("--control-alignment-max-log-alpha-change", type=float, default=0.05, help="absolute log-alpha change bound for valid alignment-aware updates")
parser.add_argument("--control-alignment-eps", type=float, default=1e-12, help="alignment denominator epsilon")
parser.add_argument("--control-reject-bad-steps", action="store_true", help="reserved; not supported in first nanochat implementation")
parser.add_argument("--control-log-every", type=int, default=1, help="write controller CSV rows every N controller updates")
parser.add_argument("--control-output-dir", type=str, default="", help="directory for local controlled optimizer metrics")
parser.add_argument("--local-output-dir", type=str, default="", help="directory for local train/eval metrics for any optimizer variant")
parser.add_argument("--control-cooldown-window-evals", type=int, default=5, help="validation observations in the robust cooldown progress window")
parser.add_argument("--control-cooldown-patience-windows", type=int, default=2, help="consecutive low-progress windows required for cooldown")
parser.add_argument("--control-cooldown-min-relative-progress-per-billion", type=float, default=0.05, help="diminishing-progress threshold as relative validation-BPB improvement per billion tokens")
parser.add_argument("--control-cooldown-event-log-reduction", type=float, default=math.log(2.0), help="finite log alpha-cap reduction per confirmed cooldown event")
parser.add_argument("--control-cooldown-transition-steps", type=int, default=200, help="optimization steps used to reach each new alpha cap")
parser.add_argument("--control-cooldown-holdoff-evals", type=int, default=2, help="validation observations ignored after a cap transition")
parser.add_argument("--control-cooldown-min-cap-ratio", type=float, default=0.10, help="minimum cap relative to initial controller alpha")
parser.add_argument("--control-cooldown-max-events", type=int, default=3, help="maximum autonomous cooldown events")
parser.add_argument("--local-log-every", type=int, default=1, help="write train CSV rows every N optimization steps")
parser.add_argument("--component-timing-mode", type=str, default="off", choices=["off", "cuda_events", "synchronized"], help="accepted for experiment configs; detailed component timing is disabled in this local patch")
parser.add_argument("--timing-warmup-steps", type=int, default=10, help="component timing warmup steps")
parser.add_argument("--timing-probe-warmup-count", type=int, default=1, help="component timing probe warmup count")
parser.add_argument("--timing-log-every", type=int, default=1, help="component timing log period")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
parser.add_argument("--skip-final-checkpoint", action="store_true", help="skip the final checkpoint for disposable benchmark jobs")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
autonomous_cooldown_enabled = args.optimizer_variant in AUTONOMOUS_COOLDOWN_MUON_VARIANTS
cooldown_defaults = {
    "control_cooldown_window_evals": 5,
    "control_cooldown_patience_windows": 2,
    "control_cooldown_min_relative_progress_per_billion": 0.05,
    "control_cooldown_event_log_reduction": math.log(2.0),
    "control_cooldown_transition_steps": 200,
    "control_cooldown_holdoff_evals": 2,
    "control_cooldown_min_cap_ratio": 0.10,
    "control_cooldown_max_events": 3,
}
if not autonomous_cooldown_enabled:
    changed_cooldown_args = [
        name for name, default in cooldown_defaults.items()
        if getattr(args, name) != default
    ]
    if changed_cooldown_args:
        raise ValueError("cooldown flags require an _autocooldown optimizer variant")
if args.control_reject_bad_steps:
    raise ValueError("--control-reject-bad-steps is reserved for a later transactional implementation")
if args.optimizer_variant in CONTROLLED_MUON_VARIANTS and not args.controlled_muon:
    args.controlled_muon = True
if args.controlled_muon and args.optimizer_variant not in CONTROLLED_MUON_VARIANTS:
    raise ValueError("--controlled-muon requires a controlled_muon_* optimizer variant")
control_rho_middle = args.control_rho_star if args.control_rho_middle is None else args.control_rho_middle
control_rho_cruise = control_rho_middle if args.control_rho_cruise is None else args.control_rho_cruise
control_rho_late = args.control_rho_late
if args.control_rho_reference in {"loss_progress", "loss_progress_three_stage"}:
    if args.optimizer_variant not in {"controlled_muon_raw", "controlled_muon_ema", "controlled_muon_ema_trust"}:
        raise ValueError("loss_progress rho reference is currently supported only for P controlled_muon variants")
    if control_rho_late is None:
        raise ValueError("loss_progress rho reference requires --control-rho-late")
    if args.control_alpha_warmup_steps != 0 or args.control_start_step not in {-1, 0}:
        raise ValueError("loss_progress rho reference requires zero controller alpha warmup and control-start-step=0")
if args.control_rho_reference == "loss_progress_three_stage":
    if args.control_rho_start is None:
        raise ValueError("loss_progress_three_stage requires --control-rho-start")
    if not args.control_rho_start < control_rho_cruise < control_rho_late:
        raise ValueError("three-stage rho targets must satisfy start < cruise < late")
    if args.control_startup_alpha_reference_ratio != 1.0:
        raise ValueError("three-stage rho reference does not use a startup alpha reference")
if args.control_startup_kp >= 0 and args.control_rho_reference != "loss_progress_three_stage":
    raise ValueError("--control-startup-kp requires loss_progress_three_stage")
if args.control_startup_factor_max >= 0 and args.control_rho_reference != "loss_progress_three_stage":
    raise ValueError("--control-startup-factor-max requires loss_progress_three_stage")
phase_hold_enabled = args.control_action_policy in {"phase_hold", "phase_hold_recovery"}
if phase_hold_enabled:
    if args.control_rho_reference != "loss_progress_three_stage":
        raise ValueError("phase-hold action policies require loss_progress_three_stage")
    if args.control_alpha_mode not in {"absolute", "multiplier"} or args.control_scope != "muon_only":
        raise ValueError("phase-hold action policies require absolute or multiplier Muon-only control")
    if args.optimizer_variant not in {"controlled_muon_raw", "controlled_muon_ema", "controlled_muon_ema_trust"}:
        raise ValueError("phase-hold action policies are supported only for P controlled_muon variants")
    if args.control_startup_kp < 0:
        raise ValueError("phase-hold action policies require an explicit non-negative startup kp")
    if args.control_phase_hold_cruise_policy == "hold" and args.control_phase_hold_cruise_kp != 0.0:
        raise ValueError("phase_hold exact cruise requires zero cruise kp")
    if args.control_phase_hold_cruise_policy == "rho_deadband" and args.control_phase_hold_cruise_kp <= 0.0:
        raise ValueError("phase_hold rho deadband requires positive cruise kp")
    if args.control_phase_hold_cruise_deadband < 0.0:
        raise ValueError("phase_hold cruise deadband must be non-negative")
    if args.control_phase_hold_late_kp < 0.0:
        raise ValueError("phase_hold late kp must be non-negative")
    if args.control_phase_hold_late_exponent < 1.0:
        raise ValueError("phase_hold late exponent must be at least 1")
if args.control_action_policy == "phase_hold_recovery":
    if not math.isfinite(args.control_recovery_terminal_ratio) or not 0 < args.control_recovery_terminal_ratio <= 1:
        raise ValueError("phase_hold_recovery terminal ratio must be finite and in (0, 1]")
    if not math.isfinite(args.control_recovery_exponent) or args.control_recovery_exponent < 1:
        raise ValueError("phase_hold_recovery exponent must be finite and at least 1")
elif (
    args.control_recovery_terminal_ratio != 1.0
    or args.control_recovery_exponent != 4.0
):
    raise ValueError("recovery flags require --control-action-policy=phase_hold_recovery")
if args.control_alignment_aware and args.optimizer_variant not in CONTROLLED_MUON_VARIANTS:
    raise ValueError("--control-alignment-aware requires a controlled P/PI/PID Muon variant")
if args.seed < 0:
    raise ValueError("--seed must be non-negative")
validate_absolute_group_scaling(
    mode=args.control_absolute_group_scaling,
    controlled=args.optimizer_variant in CONTROLLED_MUON_VARIANTS,
    control_scope=args.control_scope,
    alpha_mode=args.control_alpha_mode,
    alpha_reference=args.control_alpha_reference,
)
if args.control_alpha_reference != "none":
    raise ValueError("--control-alpha-reference=native_wsd is not enabled for the direct absolute-alpha launch path")
validate_control_feedback_configuration(
    scope=args.control_feedback_scope,
    controlled=args.optimizer_variant in CONTROLLED_MUON_VARIANTS,
    control_scope=args.control_scope,
)
if args.control_alpha_replay_file:
    raise ValueError("--control-alpha-replay-file is reserved and not enabled for this launch path")
if args.control_period_steps <= 0:
    raise ValueError("--control-period-steps must be positive")
if args.control_log_every <= 0:
    raise ValueError("--control-log-every must be positive")
if args.local_log_every <= 0:
    raise ValueError("--local-log-every must be positive")
if args.control_residual_validity_gate and args.control_feedback_scope != "muon_residual_proxy":
    raise ValueError("--control-residual-validity-gate requires muon_residual_proxy feedback")
if args.control_residual_min_muon_predicted_fraction < 0 or args.control_residual_min_muon_predicted_fraction > 1:
    raise ValueError("--control-residual-min-muon-predicted-fraction must be in [0, 1]")
if args.control_residual_max_adamw_predicted_fraction < 0 or args.control_residual_max_adamw_predicted_fraction > 1:
    raise ValueError("--control-residual-max-adamw-predicted-fraction must be in [0, 1]")
if args.control_startup_alpha_reference_ratio < 1:
    raise ValueError("--control-startup-alpha-reference-ratio must be at least 1")
if args.control_startup_alpha_reference_ratio > 1 and args.control_rho_reference != "loss_progress":
    raise ValueError("startup alpha reference requires --control-rho-reference=loss_progress")
if args.control_startup_alpha_reference_ratio > 1 and (args.control_alpha_warmup_steps != 0 or args.control_start_step not in {-1, 0}):
    raise ValueError("startup alpha reference requires zero controller warmup and control-start-step=0")
if (args.control_startup_one_sided_safety or args.control_startup_monotone) and args.control_startup_alpha_reference_ratio <= 1:
    raise ValueError("startup safety options require an enabled startup alpha reference")
if args.control_alpha_warmup_steps < 0:
    raise ValueError("--control-alpha-warmup-steps must be non-negative")
if args.control_start_step < -1:
    raise ValueError("--control-start-step must be -1 or non-negative")
if args.control_start_step == -1:
    args.control_start_step = args.control_alpha_warmup_steps if args.control_alpha_mode == "absolute" else 0
if args.timing_warmup_steps < 0 or args.timing_probe_warmup_count < 0:
    raise ValueError("timing warmup counts must be non-negative")
if args.timing_log_every <= 0:
    raise ValueError("--timing-log-every must be positive")
if autonomous_cooldown_enabled and args.eval_every <= 0:
    raise ValueError("autonomous cooldown requires --eval-every > 0")
if autonomous_cooldown_enabled and args.eval_tokens <= 0:
    raise ValueError("autonomous cooldown requires --eval-tokens > 0")
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type, seed=args.seed)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
if args.optimizer_variant == "torch_muon" and ddp_world_size > 1:
    raise ValueError("torch_muon reference baseline is single-rank only; run one process per GPU instead of torchrun")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Project-local metrics for controlled-optimizer experiments.
def _csv_writer(path, fieldnames):
    if not master_process:
        return None, None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer

def _write_csv_row(writer, file_handle, row):
    if writer is None:
        return
    writer.writerow(row)
    file_handle.flush()

def _all_reduce_mean_scalar(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if is_ddp_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())

def _average_probe_loss(model, probe_batches):
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in probe_batches:
            loss = model(xb, yb)
            loss_sum += float(loss.detach().item())
            count += 1
    if count == 0:
        raise RuntimeError("probe loss requested with no stored probe batches")
    return loss_sum / count

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer_backend = "torch_muon" if args.optimizer_variant == "torch_muon" else "native"
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    backend=optimizer_backend,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

controlled_muon_enabled = args.optimizer_variant in CONTROLLED_MUON_VARIANTS
muon_initial_lrs = [float(group["initial_lr"]) for group in optimizer.param_groups if group["kind"] == "muon"]
native_muon_reference_peak_lr = max(muon_initial_lrs) if muon_initial_lrs else float(args.matrix_lr * batch_lr_scale)
scaled_absolute_all_groups_enabled = (
    controlled_muon_enabled
    and args.control_scope == "all_groups"
    and args.control_alpha_mode == "absolute"
    and args.control_alpha_reference == "none"
    and args.control_absolute_group_scaling == "initial_lr_ratio"
)
for group in optimizer.param_groups:
    group["control_absolute_lr_scale"] = absolute_group_lr_scale(
        mode=args.control_absolute_group_scaling,
        initial_lr=float(group["initial_lr"]),
        anchor_lr=native_muon_reference_peak_lr,
    )
rho_reference = None
if controlled_muon_enabled and args.control_rho_reference == "loss_progress":
    rho_reference = LossProgressRhoReference(
        rho_middle=control_rho_middle,
        rho_late=control_rho_late,
        beta_fast=args.control_rho_progress_fast_beta,
        beta_slow=args.control_rho_progress_slow_beta,
        beta_reference=args.control_rho_progress_reference_beta,
        beta_phase=args.control_rho_phase_beta,
        progress_ratio_high=args.control_rho_progress_ratio_high,
        progress_ratio_low=args.control_rho_progress_ratio_low,
        minimum_observations=args.control_rho_progress_min_observations,
    )
elif controlled_muon_enabled and args.control_rho_reference == "loss_progress_three_stage":
    rho_reference = ThreeStageLossProgressRhoReference(
        rho_start=args.control_rho_start,
        rho_cruise=control_rho_cruise,
        rho_late=control_rho_late,
        startup_beta_fast=args.control_rho_startup_fast_beta,
        startup_beta_slow=args.control_rho_startup_slow_beta,
        startup_beta_reference=args.control_rho_startup_reference_beta,
        startup_beta_phase=args.control_rho_startup_phase_beta,
        startup_progress_ratio_high=args.control_rho_startup_ratio_high,
        startup_progress_ratio_low=args.control_rho_startup_ratio_low,
        startup_minimum_observations=args.control_rho_startup_min_observations,
        late_beta_fast=args.control_rho_progress_fast_beta,
        late_beta_slow=args.control_rho_progress_slow_beta,
        late_beta_reference=args.control_rho_progress_reference_beta,
        late_beta_phase=args.control_rho_phase_beta,
        late_progress_ratio_high=args.control_rho_progress_ratio_high,
        late_progress_ratio_low=args.control_rho_progress_ratio_low,
        late_minimum_observations=args.control_rho_progress_min_observations,
    )
controller = None
control_output_dir = args.control_output_dir
if controlled_muon_enabled:
    if args.control_alpha_init > 0:
        alpha_init = args.control_alpha_init
    elif args.control_alpha_mode == "multiplier":
        alpha_init = 1.0
    else:
        alpha_init = args.matrix_lr * batch_lr_scale
    alpha_min = args.control_alpha_min if args.control_alpha_min > 0 else 0.25 * alpha_init
    alpha_max = args.control_alpha_max if args.control_alpha_max > 0 else 2.0 * alpha_init
    base_controller_variant = AUTONOMOUS_COOLDOWN_VARIANTS.get(args.optimizer_variant, args.optimizer_variant)
    base_controller = NanochatMuonController(
        variant=base_controller_variant,
        alpha_init=alpha_init,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        rho_star=rho_reference.rho_star if rho_reference is not None else args.control_rho_star,
        kp=args.control_kp,
        ki=args.control_ki,
        kd=args.control_kd,
        rho_beta=args.control_rho_beta,
        integral_beta=args.control_integral_beta,
        integral_clip=args.control_integral_clip,
        derivative_beta=args.control_derivative_beta,
        factor_min=args.control_factor_min,
        factor_max=args.control_factor_max,
        rho_clip_min=args.control_rho_clip_min,
        rho_clip_max=args.control_rho_clip_max,
        trust_region_rho_threshold=args.control_trust_rho_threshold,
        trust_region_alpha_threshold=args.control_trust_alpha_threshold,
        trust_region_expand_factor=args.control_trust_expand_factor,
        trust_region_max_factor=args.control_trust_max_factor,
        trust_region_patience=args.control_trust_patience,
        alignment_aware=args.control_alignment_aware,
        alignment_c_min=args.control_alignment_c_min,
        alignment_c_bad=args.control_alignment_c_bad,
        alignment_penalty=args.control_alignment_penalty,
        alignment_bad_step_shrink=args.control_alignment_bad_step_shrink,
        alignment_max_log_alpha_change=args.control_alignment_max_log_alpha_change,
        alignment_eps=args.control_alignment_eps,
        startup_alpha_reference_ratio=args.control_startup_alpha_reference_ratio,
        startup_alpha_reference_gain=args.control_startup_alpha_reference_gain,
        startup_one_sided_safety=args.control_startup_one_sided_safety,
        startup_monotone=args.control_startup_monotone,
        startup_emergency_rho=args.control_startup_emergency_rho,
        startup_kp=args.control_startup_kp if args.control_startup_kp >= 0 else None,
        startup_factor_max=(
            args.control_startup_factor_max
            if args.control_startup_factor_max >= 0
            else None
        ),
        action_policy=args.control_action_policy,
        phase_hold_cruise_policy=args.control_phase_hold_cruise_policy,
        phase_hold_start_rho=(
            args.control_rho_start
            if args.control_rho_start is not None
            else args.control_rho_star
        ),
        phase_hold_cruise_rho=control_rho_cruise,
        phase_hold_cruise_kp=args.control_phase_hold_cruise_kp,
        phase_hold_cruise_deadband=args.control_phase_hold_cruise_deadband,
        phase_hold_late_rho=(
            control_rho_late
            if control_rho_late is not None
            else args.control_rho_star
        ),
        phase_hold_late_kp=args.control_phase_hold_late_kp,
        phase_hold_late_exponent=args.control_phase_hold_late_exponent,
        recovery_terminal_ratio=args.control_recovery_terminal_ratio,
        recovery_exponent=args.control_recovery_exponent,
    )
    if not control_output_dir:
        control_output_dir = os.path.join(base_dir, "controlled_optimizer_outputs", output_dirname, args.optimizer_variant)
    if autonomous_cooldown_enabled:
        cooldown_detector = ValidationProgressDetector(
            window_evals=args.control_cooldown_window_evals,
            patience_windows=args.control_cooldown_patience_windows,
            min_relative_progress_per_billion=args.control_cooldown_min_relative_progress_per_billion,
        )
        cooldown_governor = AlphaCeilingGovernor(
            reference_alpha=alpha_init,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            detector=cooldown_detector,
            event_log_reduction=args.control_cooldown_event_log_reduction,
            transition_steps=args.control_cooldown_transition_steps,
            holdoff_evals=args.control_cooldown_holdoff_evals,
            minimum_cap_ratio=args.control_cooldown_min_cap_ratio,
            maximum_events=args.control_cooldown_max_events,
        )
        controller = GovernedMuonController(
            variant=args.optimizer_variant,
            base_controller=base_controller,
            governor=cooldown_governor,
        )
    else:
        controller = base_controller
    if resuming:
        controller_state = meta_data.get("loop_state", {}).get("controlled_muon_controller")
        if controller_state is not None:
            controller.load_state_dict(controller_state)
        saved_reference_mode = meta_data.get("user_config", {}).get("control_rho_reference", "fixed")
        if saved_reference_mode != args.control_rho_reference:
            raise ValueError(f"cannot resume {saved_reference_mode} rho reference with {args.control_rho_reference} configured")
        saved_reference_state = meta_data.get("loop_state", {}).get("controlled_muon_rho_reference")
        if rho_reference is None:
            if saved_reference_state is not None:
                raise ValueError("dynamic rho reference checkpoint requires a loss-progress rho reference")
        elif saved_reference_state is None:
            raise ValueError("loss-progress rho reference checkpoint state is missing")
        else:
            rho_reference.load_state_dict(saved_reference_state)
        validate_resumed_control_feedback(
            configured_scope=args.control_feedback_scope,
            saved_state=meta_data.get("loop_state", {}).get(
                "controlled_muon_feedback"
            ),
        )

metrics_output_dir = args.local_output_dir
if controlled_muon_enabled and not metrics_output_dir:
    metrics_output_dir = control_output_dir
local_metrics_enabled = bool(metrics_output_dir)
if local_metrics_enabled and master_process:
    os.makedirs(metrics_output_dir, exist_ok=True)

if local_metrics_enabled and master_process:
    metadata_path = os.path.join(metrics_output_dir, "run_metadata.json")
    metadata = {
        "nanochat_commit": "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd",
        "optimizer_variant": args.optimizer_variant,
        "optimizer_backend": optimizer_backend,
        "controlled_muon_enabled": controlled_muon_enabled,
        "control_output_dir": control_output_dir,
        "local_output_dir": metrics_output_dir,
        "user_config": user_config,
        "model_config": model_config_kwargs,
        "param_counts": param_counts,
        "batch_lr_scale": batch_lr_scale,
        "weight_decay_scaled": weight_decay_scaled,
        "controller": None if controller is None else controller.state_dict(),
        "control_alpha_reference": args.control_alpha_reference,
        "control_absolute_group_scaling": args.control_absolute_group_scaling,
        "native_muon_reference_peak_lr": native_muon_reference_peak_lr,
        "scaled_absolute_all_groups_enabled": scaled_absolute_all_groups_enabled,
        "control_feedback": control_feedback_state_dict(args.control_feedback_scope),
        "control_rho_reference": args.control_rho_reference,
        "rho_reference": None if rho_reference is None else rho_reference.state_dict(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

if controlled_muon_enabled and master_process:
    os.makedirs(control_output_dir, exist_ok=True)

controller_csv_fields = [
    "step", "tokens", "alpha", "alpha_next", "alpha_update_factor", "alpha_mode",
    "control_scope", "feedback_scope",
    "native_lrm", "native_muon_lr_min", "native_muon_lr_max", "muon_lr_min",
    "muon_lr_max", "adamw_lr_min", "adamw_lr_max",
    "effective_muon_lr_min", "effective_muon_lr_max",
    "effective_muon_lr_mean", "rho", "rho_clipped", "rho_ema", "rho_control",
    "rho_star_applied", "rho_reference_mode",
    "feedback_observation_valid", "feedback_invalid_reason",
    "startup_active", "startup_alpha_reference", "startup_log_term",
    "startup_emergency", "startup_monotone_clamped", "startup_weight",
    "kp_applied", "factor_max_applied",
    "action_policy", "phase_start_weight", "phase_cruise_weight",
    "phase_late_weight", "phase_start_action", "phase_cruise_action",
    "phase_late_action", "phase_cruise_deadband_active",
    "phase_late_exponent",
    "recovery_enabled", "recovery_terminal_ratio", "recovery_exponent",
    "recovery_alpha_peak", "recovery_peak_frozen",
    "recovery_late_phase_raw", "recovery_late_phase_monotone",
    "recovery_progress", "recovery_alpha_uncapped", "recovery_alpha_cap",
    "recovery_cap_binding", "recovery_cap_binding_count",
    "rho_phase_observation_count", "rho_phase_loss_fast",
    "rho_phase_loss_slow", "rho_phase_relative_progress", "rho_phase_progress_reference",
    "rho_phase_progress_ratio", "rho_phase_candidate", "rho_phase",
    "rho_startup_phase", "rho_startup_weight", "rho_startup_progress_ratio",
    "rho_startup_phase_candidate", "rho_late_phase",
    "loss_before_probe", "loss_after_probe", "actual_decrease",
    "feedback_actual_decrease", "feedback_predicted_decrease",
    "rho_total", "rho_muon_residual_proxy", "adamw_predicted_fraction",
    "predicted_decrease_total", "predicted_decrease_muon", "predicted_decrease_adamw",
    "predicted_decrease_safe", "control_error", "p_term", "i_term", "d_term",
    "control_log_factor", "integral_state", "derivative_state",
    "muon_update_norm", "adamw_update_norm", "total_update_norm", "muon_grad_norm",
    "adamw_grad_norm", "total_grad_norm", "num_muon_params", "num_adamw_params",
    "num_muon_tensors", "num_adamw_tensors", "factor_applied",
    "trust_region_expanded", "trust_good_count", "probe_scope", "probe_num_microbatches",
    "alignment_c", "alignment_penalty_term", "alignment_allows_trust_expansion",
    "alignment_bad_step", "integral_accumulation_frozen",
    "cooldown_alpha_proposed", "cooldown_alpha_cap_target",
    "cooldown_alpha_cap", "cooldown_alpha_applied",
    "cooldown_cap_is_binding", "cooldown_governor_state",
    "cooldown_event_count",
    "predicted_was_floored", "rho_was_clipped", "skipped_reason", "dt_seconds",
]
train_csv_fields = [
    "step", "tokens", "train_loss", "smooth_train_loss", "lrm", "dt_seconds",
    "tok_per_sec", "mfu", "total_training_time_seconds", "alpha",
    "native_muon_lr_min", "native_muon_lr_max", "muon_lr_min", "muon_lr_max",
    "muon_lr_mean", "effective_muon_lr_mean",
]
eval_csv_fields = [
    "step", "tokens", "val_bpb", "min_val_bpb", "eval_steps", "eval_tokens",
    "total_training_time_seconds", "wallclock_time_seconds",
]
cooldown_csv_fields = [
    "step", "tokens", "val_bpb", "window_progress_per_billion",
    "plateau_threshold", "plateau_candidate", "plateau_streak",
    "plateau_confirmed", "window_observations", "governor_state",
    "holdoff_evals_remaining", "cooldown_event", "cooldown_event_count",
    "alpha_proposed", "alpha_cap_target", "alpha_cap", "alpha_applied",
    "cap_is_binding",
]
controller_csv_file, controller_csv_writer = (None, None)
train_csv_file, train_csv_writer = (None, None)
eval_csv_file, eval_csv_writer = (None, None)
cooldown_csv_file, cooldown_csv_writer = (None, None)
if controlled_muon_enabled:
    controller_csv_file, controller_csv_writer = _csv_writer(os.path.join(control_output_dir, "controller_metrics.csv"), controller_csv_fields)
if autonomous_cooldown_enabled:
    cooldown_csv_file, cooldown_csv_writer = _csv_writer(os.path.join(control_output_dir, "cooldown_metrics.csv"), cooldown_csv_fields)
if local_metrics_enabled:
    train_csv_file, train_csv_writer = _csv_writer(os.path.join(metrics_output_dir, "train_metrics.csv"), train_csv_fields)
    eval_csv_file, eval_csv_writer = _csv_writer(os.path.join(metrics_output_dir, "eval_metrics.csv"), eval_csv_fields)
    if master_process:
        group_summary = []
        for group_idx, group in enumerate(optimizer.param_groups):
            params = group["params"]
            num_params = sum(p.numel() for p in params)
            shapes = {}
            for p in params:
                key = "x".join(str(dim) for dim in p.shape)
                shapes[key] = shapes.get(key, 0) + 1
            group_summary.append({
                "group_idx": group_idx,
                "kind": group["kind"],
                "initial_lr": group["initial_lr"],
                "control_absolute_lr_scale": group.get("control_absolute_lr_scale"),
                "num_tensors": len(params),
                "num_params": num_params,
                "shapes": shapes,
            })
        with open(os.path.join(metrics_output_dir, "optimizer_group_summary.json"), "w", encoding="utf-8") as f:
            json.dump(group_summary, f, indent=2)

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")

if args.stop_after_steps > 0:
    if args.stop_after_steps > num_iterations:
        raise ValueError("--stop-after-steps must be <= --num-iterations / computed scheduler horizon")
    stop_step = args.stop_after_steps
    print0(f"Will stop after {stop_step:,} optimizer steps while using {num_iterations:,} as scheduler horizon")
else:
    stop_step = num_iterations

schedule_total_tokens = total_batch_size * num_iterations
total_tokens = total_batch_size * stop_step # the actual number of tokens we will train for
print0(f"Scheduler horizon tokens: {schedule_total_tokens:,}")
print0(f"Planned training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_tokens / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

def get_control_alpha_warmup_multiplier(it):
    if (
        not controlled_muon_enabled
        or args.control_alpha_mode != "absolute"
        or args.control_alpha_warmup_steps <= 0
    ):
        return 1.0
    return min(1.0, (it + 1) / args.control_alpha_warmup_steps)

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
run_wallclock_start = time.time()
runtime_stop_announced = False
while True:
    runtime_limit_reached = (
        args.max_runtime_minutes > 0
        and step > 0
        and (time.time() - run_wallclock_start) >= args.max_runtime_minutes * 60.0
    )
    if runtime_limit_reached and not runtime_stop_announced:
        print0(
            f"Stopping because max runtime {args.max_runtime_minutes:.2f} minutes "
            f"was reached at step {step:,}"
        )
        runtime_stop_announced = True
    last_step = step == stop_step or runtime_limit_reached # loop runs stop_step+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        if autonomous_cooldown_enabled and eval_steps < 1:
            raise ValueError("autonomous cooldown requires --eval-tokens to cover at least one distributed evaluation batch")
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        _write_csv_row(eval_csv_writer, eval_csv_file, {
            "step": step,
            "tokens": total_batch_size * step,
            "val_bpb": val_bpb,
            "min_val_bpb": min_val_bpb,
            "eval_steps": eval_steps,
            "eval_tokens": args.eval_tokens,
            "total_training_time_seconds": total_training_time,
            "wallclock_time_seconds": time.time() - run_wallclock_start,
        })
        if autonomous_cooldown_enabled:
            cooldown_stats = None
            duplicate_resume_validation = False
            if master_process:
                duplicate_resume_validation = (
                    controller.governor.last_validation_step == step
                )
                if duplicate_resume_validation:
                    cooldown_stats = controller.governor.last_validation_stats
                else:
                    cooldown_stats = controller.observe_validation(
                        step=step,
                        tokens=total_batch_size * step,
                        val_bpb=val_bpb,
                        allow_event=not last_step,
                    )
            if is_ddp_initialized():
                cooldown_state_payload = [
                    controller.state_dict() if master_process else None
                ]
                dist.broadcast_object_list(cooldown_state_payload, src=0)
                if not master_process:
                    controller.load_state_dict(cooldown_state_payload[0])
                    cooldown_stats = controller.governor.last_validation_stats
            if master_process and not duplicate_resume_validation:
                _write_csv_row(cooldown_csv_writer, cooldown_csv_file, {
                    "step": cooldown_stats.step,
                    "tokens": cooldown_stats.tokens,
                    "val_bpb": cooldown_stats.val_bpb,
                    "window_progress_per_billion": cooldown_stats.window_progress_per_billion,
                    "plateau_threshold": args.control_cooldown_min_relative_progress_per_billion,
                    "plateau_candidate": int(cooldown_stats.plateau_candidate),
                    "plateau_streak": cooldown_stats.plateau_streak,
                    "plateau_confirmed": int(cooldown_stats.plateau_confirmed),
                    "window_observations": cooldown_stats.window_observations,
                    "governor_state": cooldown_stats.governor_state,
                    "holdoff_evals_remaining": cooldown_stats.holdoff_evals_remaining,
                    "cooldown_event": int(cooldown_stats.cooldown_event),
                    "cooldown_event_count": cooldown_stats.cooldown_event_count,
                    "alpha_proposed": cooldown_stats.alpha_proposed,
                    "alpha_cap_target": cooldown_stats.alpha_cap_target,
                    "alpha_cap": cooldown_stats.alpha_cap,
                    "alpha_applied": cooldown_stats.alpha_applied,
                    "cap_is_binding": int(cooldown_stats.cap_is_binding),
                })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if (last_step and not args.skip_final_checkpoint) or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            optimizer.state_dict(), # optimizer state
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                    "controlled_muon_controller": None if controller is None else controller.state_dict(),
                    "controlled_muon_feedback": None if controller is None else control_feedback_state_dict(args.control_feedback_scope),
                    "controlled_muon_rho_reference": None if rho_reference is None else rho_reference.state_dict(),
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    control_active = controlled_muon_enabled and step >= args.control_start_step
    probe_this_step = control_active and ((step - args.control_start_step) % args.control_period_steps == 0)
    probe_batches = []
    loss_before_probe_sum = 0.0
    loss_before_probe_count = 0
    control_stats = None
    control_diagnostics = None
    control_feedback = None
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach() # for logging
        if probe_this_step:
            if args.control_probe_scope == "full_accum_batch":
                probe_batches.append((x.detach().clone(), y.detach().clone()))
                loss_before_probe_sum += float(loss.detach().item())
                loss_before_probe_count += 1
            elif args.control_probe_scope == "last_microbatch" and micro_step == grad_accum_steps - 1:
                probe_batches = [(x.detach().clone(), y.detach().clone())]
                loss_before_probe_sum = float(loss.detach().item())
                loss_before_probe_count = 1
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    if autonomous_cooldown_enabled:
        controller.prepare_step(step)
    control_alpha_warmup_multiplier = get_control_alpha_warmup_multiplier(step)
    control_alpha_applied = (
        controller.alpha * control_alpha_warmup_multiplier
        if controlled_muon_enabled and controller is not None
        else None
    )
    native_muon_lr_values = []
    muon_lr_values = []
    adamw_lr_values = []
    effective_muon_lr_values = []
    for group in optimizer.param_groups:
        base_lr = group["initial_lr"] * lrm
        group["lr"] = base_lr
        if controlled_muon_enabled and args.control_scope == "all_groups":
            if args.control_alpha_mode == "absolute":
                scale = group.get("control_absolute_lr_scale", 1.0) if scaled_absolute_all_groups_enabled else 1.0
                group["lr"] = control_alpha_applied * scale
            elif args.control_alpha_mode == "multiplier":
                group["lr"] = base_lr * controller.alpha
            else:
                raise ValueError(f"unknown control alpha mode: {args.control_alpha_mode}")
        if group["kind"] == "muon":
            native_muon_lr_values.append(float(base_lr))
            if controlled_muon_enabled and args.control_scope == "muon_only":
                if args.control_alpha_mode == "absolute":
                    group["lr"] = control_alpha_applied
                elif args.control_alpha_mode == "multiplier":
                    group["lr"] = base_lr * controller.alpha
                else:
                    raise ValueError(f"unknown control alpha mode: {args.control_alpha_mode}")
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
            shape = group["params"][0].shape
            shape_scale = max(1.0, shape[-2] / shape[-1])**0.5
            muon_lr_values.append(float(group["lr"]))
            effective_muon_lr_values.append(float(group["lr"] * shape_scale))
        elif group["kind"] == "adamw":
            adamw_lr_values.append(float(group["lr"]))
    native_muon_lr_min = min(native_muon_lr_values) if native_muon_lr_values else ""
    native_muon_lr_max = max(native_muon_lr_values) if native_muon_lr_values else ""
    muon_lr_min = min(muon_lr_values) if muon_lr_values else ""
    muon_lr_max = max(muon_lr_values) if muon_lr_values else ""
    muon_lr_mean = sum(muon_lr_values) / len(muon_lr_values) if muon_lr_values else ""
    adamw_lr_min = min(adamw_lr_values) if adamw_lr_values else ""
    adamw_lr_max = max(adamw_lr_values) if adamw_lr_values else ""
    effective_muon_lr_min = min(effective_muon_lr_values) if effective_muon_lr_values else ""
    effective_muon_lr_max = max(effective_muon_lr_values) if effective_muon_lr_values else ""
    effective_muon_lr_mean = sum(effective_muon_lr_values) / len(effective_muon_lr_values) if effective_muon_lr_values else ""
    if rho_reference is not None:
        train_loss_f = train_loss.item()
        rho_reference.observe(_all_reduce_mean_scalar(train_loss_f, device))
    if probe_this_step:
        optimizer.set_control_diagnostics(True, include_adamw=True)
    step_skipped_by_scaler = False
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        found_inf_values = list(scaler._found_inf_per_device(optimizer).values())
        if is_ddp_initialized():
            for v in found_inf_values:
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        step_skipped_by_scaler = any(float(v.item()) != 0.0 for v in found_inf_values)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if probe_this_step:
        control_diagnostics = optimizer.consume_control_diagnostics(all_reduce=True)
        if not step_skipped_by_scaler:
            loss_before_probe = loss_before_probe_sum / max(1, loss_before_probe_count)
            loss_after_probe = _average_probe_loss(model, probe_batches)
            loss_before_probe = _all_reduce_mean_scalar(loss_before_probe, device)
            loss_after_probe = _all_reduce_mean_scalar(loss_after_probe, device)
            actual_total = loss_before_probe - loss_after_probe
            control_feedback = select_control_feedback(
                scope=args.control_feedback_scope,
                actual_total=actual_total,
                predicted_total=control_diagnostics["predicted_decrease_total"],
                predicted_muon=control_diagnostics["predicted_decrease_muon"],
                predicted_adamw=control_diagnostics["predicted_decrease_adamw"],
                total_grad_norm=control_diagnostics["total_grad_norm"],
                muon_grad_norm=control_diagnostics["muon_grad_norm"],
                total_update_norm=control_diagnostics["total_update_norm"],
                muon_update_norm=control_diagnostics["muon_update_norm"],
            )
            feedback_observation_valid = True
            feedback_invalid_reason = None
            if args.control_residual_validity_gate:
                predicted_total = float(control_feedback.predicted_total)
                predicted_muon = float(control_feedback.predicted_for_control)
                actual_muon = float(control_feedback.actual_for_control)
                if not math.isfinite(actual_muon):
                    feedback_observation_valid = False
                    feedback_invalid_reason = "residual_actual_nonfinite"
                elif not math.isfinite(predicted_total) or predicted_total <= 0.0:
                    feedback_observation_valid = False
                    feedback_invalid_reason = "total_predicted_decrease_invalid"
                elif not math.isfinite(predicted_muon) or predicted_muon <= 0.0:
                    feedback_observation_valid = False
                    feedback_invalid_reason = "muon_predicted_decrease_invalid"
                else:
                    raw_residual_rho = actual_muon / predicted_muon
                    if (
                        not math.isfinite(raw_residual_rho)
                        or raw_residual_rho < args.control_rho_clip_min
                        or raw_residual_rho > args.control_rho_clip_max
                    ):
                        feedback_observation_valid = False
                        feedback_invalid_reason = "residual_rho_clipped"
                    else:
                        muon_fraction = predicted_muon / predicted_total
                        adamw_fraction = 1.0 - muon_fraction
                        if not math.isfinite(muon_fraction) or muon_fraction < args.control_residual_min_muon_predicted_fraction:
                            feedback_observation_valid = False
                            feedback_invalid_reason = "muon_predicted_fraction_too_small"
                        elif not math.isfinite(adamw_fraction) or adamw_fraction > args.control_residual_max_adamw_predicted_fraction:
                            feedback_observation_valid = False
                            feedback_invalid_reason = "adamw_predicted_fraction_too_large"
            startup_active = (
                args.control_startup_alpha_reference_ratio > 1.0
                and rho_reference is not None
                and rho_reference.phase <= 0.0
            )
            startup_weight = float(getattr(rho_reference, "startup_weight", 0.0))
            feedback_actual_override = (
                None
                if args.control_feedback_scope == "total"
                else control_feedback.actual_for_control
            )
            control_stats = controller.update(
                step=step,
                loss_before=loss_before_probe,
                loss_after=loss_after_probe,
                predicted_decrease=control_feedback.predicted_for_control,
                grad_norm=control_feedback.grad_norm_for_control,
                update_norm=control_feedback.update_norm_for_control,
                feedback_actual_decrease=feedback_actual_override,
                feedback_observation_valid=feedback_observation_valid,
                feedback_invalid_reason=feedback_invalid_reason,
                startup_active=startup_active,
                startup_weight=startup_weight,
                late_phase=float(getattr(rho_reference, "late_phase", 0.0)),
                rho_star_override=None if rho_reference is None else rho_reference.rho_star,
            )
    model.zero_grad(set_to_none=True)
    if rho_reference is None:
        train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    log_local_metrics = local_metrics_enabled and (step % args.local_log_every == 0)
    governed_stats = (
        controller.last_governed_stats if autonomous_cooldown_enabled else None
    )
    if control_stats is not None and (controller.num_updates % args.control_log_every == 0):
        assert control_feedback is not None
        predicted_total = control_diagnostics["predicted_decrease_total"]
        adamw_predicted_fraction = (
            control_diagnostics["predicted_decrease_adamw"] / predicted_total
            if math.isfinite(predicted_total) and predicted_total > 0.0
            else ""
        )
        row = {
            "step": step,
            "tokens": total_batch_size * step,
            "alpha": control_stats.alpha,
            "alpha_next": control_stats.alpha_next,
            "alpha_update_factor": control_stats.alpha_update_factor,
            "alpha_mode": args.control_alpha_mode,
            "control_scope": args.control_scope,
            "feedback_scope": args.control_feedback_scope,
            "native_lrm": lrm,
            "native_muon_lr_min": native_muon_lr_min,
            "native_muon_lr_max": native_muon_lr_max,
            "muon_lr_min": muon_lr_min,
            "muon_lr_max": muon_lr_max,
            "adamw_lr_min": adamw_lr_min,
            "adamw_lr_max": adamw_lr_max,
            "effective_muon_lr_min": effective_muon_lr_min,
            "effective_muon_lr_max": effective_muon_lr_max,
            "effective_muon_lr_mean": effective_muon_lr_mean,
            "rho": control_stats.rho,
            "rho_clipped": control_stats.rho_clipped,
            "rho_ema": control_stats.rho_ema,
            "rho_control": control_stats.rho_control,
            "rho_star_applied": control_stats.rho_star,
            "feedback_observation_valid": int(control_stats.feedback_observation_valid),
            "feedback_invalid_reason": control_stats.feedback_invalid_reason,
            "startup_active": int(control_stats.startup_active),
            "startup_alpha_reference": control_stats.startup_alpha_reference,
            "startup_log_term": control_stats.startup_log_term,
            "startup_emergency": int(control_stats.startup_emergency),
            "startup_monotone_clamped": int(control_stats.startup_monotone_clamped),
            "startup_weight": control_stats.startup_weight,
            "kp_applied": control_stats.kp_applied,
            "factor_max_applied": control_stats.factor_max_applied,
            "action_policy": control_stats.action_policy,
            "phase_start_weight": control_stats.phase_start_weight,
            "phase_cruise_weight": control_stats.phase_cruise_weight,
            "phase_late_weight": control_stats.phase_late_weight,
            "phase_start_action": control_stats.phase_start_action,
            "phase_cruise_action": control_stats.phase_cruise_action,
            "phase_late_action": control_stats.phase_late_action,
            "phase_cruise_deadband_active": int(control_stats.phase_cruise_deadband_active),
            "phase_late_exponent": control_stats.phase_late_exponent,
            "recovery_enabled": int(control_stats.recovery_enabled),
            "recovery_terminal_ratio": control_stats.recovery_terminal_ratio,
            "recovery_exponent": control_stats.recovery_exponent,
            "recovery_alpha_peak": control_stats.recovery_alpha_peak,
            "recovery_peak_frozen": int(control_stats.recovery_peak_frozen),
            "recovery_late_phase_raw": control_stats.recovery_late_phase_raw,
            "recovery_late_phase_monotone": control_stats.recovery_late_phase_monotone,
            "recovery_progress": control_stats.recovery_progress,
            "recovery_alpha_uncapped": control_stats.recovery_alpha_uncapped,
            "recovery_alpha_cap": control_stats.recovery_alpha_cap,
            "recovery_cap_binding": int(control_stats.recovery_cap_binding),
            "recovery_cap_binding_count": control_stats.recovery_cap_binding_count,
            "rho_reference_mode": args.control_rho_reference,
            "rho_phase_observation_count": "" if rho_reference is None else rho_reference.observation_count,
            "rho_phase_loss_fast": "" if rho_reference is None else rho_reference.loss_fast,
            "rho_phase_loss_slow": "" if rho_reference is None else rho_reference.loss_slow,
            "rho_phase_relative_progress": "" if rho_reference is None else rho_reference.relative_progress,
            "rho_phase_progress_reference": "" if rho_reference is None else rho_reference.progress_reference,
            "rho_phase_progress_ratio": "" if rho_reference is None else rho_reference.progress_ratio,
            "rho_phase_candidate": "" if rho_reference is None else rho_reference.phase_candidate,
            "rho_phase": "" if rho_reference is None else rho_reference.phase,
            "rho_startup_phase": "" if rho_reference is None else getattr(rho_reference, "startup_phase", ""),
            "rho_startup_weight": "" if rho_reference is None else getattr(rho_reference, "startup_weight", ""),
            "rho_startup_progress_ratio": "" if rho_reference is None else getattr(getattr(rho_reference, "startup_reference", None), "progress_ratio", ""),
            "rho_startup_phase_candidate": "" if rho_reference is None else getattr(getattr(rho_reference, "startup_reference", None), "phase_candidate", ""),
            "rho_late_phase": "" if rho_reference is None else getattr(rho_reference, "late_phase", rho_reference.phase),
            "loss_before_probe": control_stats.loss_before,
            "loss_after_probe": control_stats.loss_after,
            "actual_decrease": control_stats.actual_decrease,
            "feedback_actual_decrease": control_feedback.actual_for_control,
            "feedback_predicted_decrease": control_feedback.predicted_for_control,
            "rho_total": control_feedback.rho_total,
            "rho_muon_residual_proxy": control_feedback.rho_muon_residual_proxy,
            "adamw_predicted_fraction": adamw_predicted_fraction,
            "predicted_decrease_total": control_diagnostics["predicted_decrease_total"],
            "predicted_decrease_muon": control_diagnostics["predicted_decrease_muon"],
            "predicted_decrease_adamw": control_diagnostics["predicted_decrease_adamw"],
            "predicted_decrease_safe": control_stats.predicted_decrease_safe,
            "control_error": control_stats.control_error,
            "p_term": control_stats.p_term,
            "i_term": control_stats.i_term,
            "d_term": control_stats.d_term,
            "control_log_factor": control_stats.control_log_factor,
            "integral_state": control_stats.integral_state,
            "derivative_state": control_stats.derivative_state,
            "muon_update_norm": control_diagnostics["muon_update_norm"],
            "adamw_update_norm": control_diagnostics["adamw_update_norm"],
            "total_update_norm": control_diagnostics["total_update_norm"],
            "muon_grad_norm": control_diagnostics["muon_grad_norm"],
            "adamw_grad_norm": control_diagnostics["adamw_grad_norm"],
            "total_grad_norm": control_diagnostics["total_grad_norm"],
            "num_muon_params": int(round(control_diagnostics["num_muon_params"])),
            "num_adamw_params": int(round(control_diagnostics["num_adamw_params"])),
            "num_muon_tensors": int(round(control_diagnostics["num_muon_tensors"])),
            "num_adamw_tensors": int(round(control_diagnostics["num_adamw_tensors"])),
            "factor_applied": control_stats.factor_applied,
            "trust_region_expanded": int(control_stats.trust_region_expanded),
            "trust_good_count": control_stats.trust_good_count,
            "alignment_c": control_stats.alignment_c,
            "alignment_penalty_term": control_stats.alignment_penalty_term,
            "alignment_allows_trust_expansion": int(control_stats.alignment_allows_trust_expansion),
            "alignment_bad_step": int(control_stats.alignment_bad_step),
            "probe_scope": args.control_probe_scope,
            "probe_num_microbatches": loss_before_probe_count,
            "predicted_was_floored": int(control_stats.predicted_was_floored),
            "rho_was_clipped": int(control_stats.rho_was_clipped),
            "integral_accumulation_frozen": int(control_stats.integral_accumulation_frozen),
            "cooldown_alpha_proposed": "" if governed_stats is None else governed_stats.alpha_proposed,
            "cooldown_alpha_cap_target": "" if governed_stats is None else governed_stats.alpha_cap_target,
            "cooldown_alpha_cap": "" if governed_stats is None else governed_stats.alpha_cap,
            "cooldown_alpha_applied": "" if governed_stats is None else governed_stats.alpha_applied,
            "cooldown_cap_is_binding": "" if governed_stats is None else int(governed_stats.cap_is_binding),
            "cooldown_governor_state": "" if governed_stats is None else governed_stats.governor_state,
            "cooldown_event_count": "" if governed_stats is None else governed_stats.cooldown_event_count,
            "skipped_reason": control_stats.skipped_reason or ("step_skipped_by_scaler" if step_skipped_by_scaler else ""),
            "dt_seconds": dt,
        }
        _write_csv_row(controller_csv_writer, controller_csv_file, row)
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / stop_step
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    if log_local_metrics:
        _write_csv_row(train_csv_writer, train_csv_file, {
            "step": step,
            "tokens": total_batch_size * step,
            "train_loss": train_loss_f,
            "smooth_train_loss": debiased_smooth_loss,
            "lrm": lrm,
            "dt_seconds": dt,
            "tok_per_sec": tok_per_sec,
            "mfu": mfu,
            "total_training_time_seconds": total_training_time,
            "alpha": "" if controller is None else control_alpha_applied,
            "native_muon_lr_min": native_muon_lr_min,
            "native_muon_lr_max": native_muon_lr_max,
            "muon_lr_min": muon_lr_min,
            "muon_lr_max": muon_lr_max,
            "muon_lr_mean": muon_lr_mean,
            "effective_muon_lr_mean": effective_muon_lr_mean,
        })
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = stop_step - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{stop_step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        if control_stats is not None:
            log_data.update({
                "control/alpha": control_stats.alpha,
                "control/alpha_next": control_stats.alpha_next,
                "control/rho": control_stats.rho,
                "control/rho_ema": control_stats.rho_ema,
                "control/loss_before_probe": control_stats.loss_before,
                "control/loss_after_probe": control_stats.loss_after,
                "control/predicted_decrease_total": control_diagnostics["predicted_decrease_total"],
                "control/predicted_decrease_muon": control_diagnostics["predicted_decrease_muon"],
                "control/predicted_decrease_adamw": control_diagnostics["predicted_decrease_adamw"],
                "control/feedback_actual_decrease": control_feedback.actual_for_control,
                "control/feedback_predicted_decrease": control_feedback.predicted_for_control,
                "control/rho_total": control_feedback.rho_total,
                "control/rho_muon_residual_proxy": control_feedback.rho_muon_residual_proxy,
            })
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect() # manually collect a lot of garbage from setup
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0: # every 5000 steps...
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
if controller_csv_file is not None:
    controller_csv_file.close()
if train_csv_file is not None:
    train_csv_file.close()
if eval_csv_file is not None:
    eval_csv_file.close()
if cooldown_csv_file is not None:
    cooldown_csv_file.close()

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
