"""Leakage-safe training and prediction primitives for Stage-7.

The module deliberately separates three optimisation stages:

* each Agent learns a valid local Known/open-set decision;
* the fair B2 parent learns only from public :class:`LocalDecision` values;
* semantic responders and the route policy use exact counterfactual targets.

Formal unknowns are accepted only by :func:`collect_predictions` when the
caller explicitly marks the call as final evaluation.  In particular, an
outer-LCO test proxy cannot silently enter any training helper.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .contracts import DialogueOutput, LocalDecision, RouteAction
from .router import public_team_features
from .splits import assert_no_formal_unknown


ACTION_ORDER = (RouteAction.STOP, RouteAction.W_FIRST, RouteAction.P_FIRST)
LOCAL_AGENT_NAMES = ("waveform", "prototype")
B2_TRANSFER_FORMAT = "maros.stage7.b2_transfer"
B2_TRANSFER_VERSION = 1


def _b2_tensor_specs(
    state: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    """Return a JSON-safe structural manifest for a B2 state mapping."""

    specs: dict[str, dict[str, Any]] = {}
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("B2 transfer state must map string keys to tensors")
        if value.layout != torch.strided:
            raise TypeError("B2 transfer accepts only dense strided tensors")
        specs[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return specs


def _b2_state_checksum(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, tensor metadata and bytes in a deterministic order."""

    digest = hashlib.sha256()
    digest.update(b"maros-stage7-b2-transfer-v1\0")
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        header = (
            f"{name}\0{value.dtype}\0"
            f"{','.join(str(size) for size in value.shape)}\0"
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        payload = value.numpy().tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def export_b2_transfer_bundle(system: nn.Module) -> dict[str, Any]:
    """Export the sole class-count-invariant Stage-7 public fusion service.

    The returned mapping deliberately contains *local* ``b2_fusion`` keys,
    never a whole-system state dict.  Consequently Agent encoders, private
    observations, response heads, the adjudicator and the router cannot be
    transferred accidentally during an inner-to-outer LCO hand-off.
    """

    try:
        fusion = system.dialogue.b2_fusion
        source_num_classes = int(system.num_classes)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("system must expose dialogue.b2_fusion and num_classes") from error
    state = {
        name: value.detach().cpu().clone()
        for name, value in fusion.state_dict().items()
    }
    specs = _b2_tensor_specs(state)
    checksum = _b2_state_checksum(state)
    return {
        "format": B2_TRANSFER_FORMAT,
        "version": B2_TRANSFER_VERSION,
        "manifest": {
            "module_path": "dialogue.b2_fusion",
            "source_num_classes": source_num_classes,
            "agent_order": list(fusion.AGENT_NAMES),
            "max_adaptive_mass": float(fusion.max_adaptive_mass),
            "tensor_specs": specs,
            "state_sha256": checksum,
        },
        "state_dict": state,
    }


def import_b2_transfer_bundle(
    system: nn.Module,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and atomically import an inner-fold B2 into an outer system.

    Exact key whitelisting is intentional: callers cannot pass a general
    checkpoint or smuggle Agent/router/responder/private parameters into this
    operation.  Tensor shape and dtype validation happens before
    ``load_state_dict`` mutates the target module.
    """

    if not isinstance(bundle, Mapping):
        raise TypeError("B2 transfer bundle must be a mapping")
    required_top_level = {"format", "version", "manifest", "state_dict"}
    if set(bundle) != required_top_level:
        raise ValueError(
            "B2 transfer bundle must contain exactly format, version, "
            "manifest and state_dict")
    if bundle["format"] != B2_TRANSFER_FORMAT:
        raise ValueError("unsupported B2 transfer format")
    if bundle["version"] != B2_TRANSFER_VERSION:
        raise ValueError("unsupported B2 transfer version")
    manifest = bundle["manifest"]
    incoming = bundle["state_dict"]
    if not isinstance(manifest, Mapping) or not isinstance(incoming, Mapping):
        raise TypeError("B2 transfer manifest and state_dict must be mappings")
    if manifest.get("module_path") != "dialogue.b2_fusion":
        raise ValueError("B2 transfer module_path is not dialogue.b2_fusion")

    try:
        fusion = system.dialogue.b2_fusion
        target_num_classes = int(system.num_classes)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError("system must expose dialogue.b2_fusion and num_classes") from error
    expected = fusion.state_dict()
    incoming_keys = set(incoming)
    forbidden_fragments = (
        "agents.", "router.", "adjudicator.", "response", "private",
    )
    forbidden = sorted(
        key for key in incoming_keys
        if not isinstance(key, str)
        or any(fragment in key.lower() for fragment in forbidden_fragments)
    )
    if forbidden:
        raise ValueError(
            "B2 transfer contains forbidden Agent/router/response/private "
            f"state keys: {forbidden}")
    expected_keys = set(expected)
    if incoming_keys != expected_keys:
        missing = sorted(expected_keys - incoming_keys)
        unexpected = sorted(incoming_keys - expected_keys)
        raise ValueError(
            "B2 transfer state is not the exact b2_fusion whitelist; "
            f"missing={missing}, unexpected={unexpected}")

    actual_specs = _b2_tensor_specs(incoming)
    if manifest.get("tensor_specs") != actual_specs:
        raise ValueError("B2 transfer tensor manifest does not match its payload")
    expected_specs = _b2_tensor_specs(expected)
    if actual_specs != expected_specs:
        mismatched = sorted(
            key for key in expected_keys
            if actual_specs[key] != expected_specs[key])
        raise ValueError(
            "B2 transfer tensor shape/dtype does not match target: "
            f"{mismatched}")
    checksum = _b2_state_checksum(incoming)
    if manifest.get("state_sha256") != checksum:
        raise ValueError("B2 transfer checksum mismatch")
    if list(manifest.get("agent_order", ())) != list(fusion.AGENT_NAMES):
        raise ValueError("B2 transfer Agent order does not match target")
    if float(manifest.get("max_adaptive_mass", float("nan"))) != float(
            fusion.max_adaptive_mass):
        raise ValueError("B2 transfer max_adaptive_mass does not match target")
    source_num_classes = manifest.get("source_num_classes")
    if (not isinstance(source_num_classes, int)
            or isinstance(source_num_classes, bool)
            or source_num_classes < 2):
        raise ValueError("B2 transfer source_num_classes is invalid")

    # Every rejection condition above precedes this sole mutation point.
    fusion.load_state_dict(dict(incoming), strict=True)
    loaded_checksum = _b2_state_checksum(fusion.state_dict())
    if loaded_checksum != checksum:  # pragma: no cover - defensive device copy audit
        raise RuntimeError("loaded B2 state differs from the validated payload")
    return {
        "format": B2_TRANSFER_FORMAT,
        "version": B2_TRANSFER_VERSION,
        "module_path": "dialogue.b2_fusion",
        "source_num_classes": source_num_classes,
        "target_num_classes": target_num_classes,
        "class_count_changed": bool(source_num_classes != target_num_classes),
        "transferred_keys": sorted(expected_keys),
        "state_sha256": checksum,
        "shape_dtype_validated": True,
        "strict_public_b2_only": True,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _capture_training_modes(module: nn.Module) -> tuple[tuple[nn.Module, bool], ...]:
    """Capture every module flag, including mixed frozen/trainable trees."""

    return tuple((child, bool(child.training)) for child in module.modules())


def _restore_training_modes(
    snapshot: Sequence[tuple[nn.Module, bool]],
) -> None:
    # Assign the flag directly: ``Module.train`` recurses and would overwrite
    # the distinct child modes that open-head-only fitting intentionally uses.
    for module, training in snapshot:
        module.training = bool(training)


class CompetitionPUGDataset(Dataset):
    """Auxiliary input-space boundary proposals derived from train-Known only.

    Stage-5 generated proposals in a learned feature space.  Stage-7 has two
    isolated encoders and intentionally exposes no shared private feature
    space, so this dataset mirrors the same outward/competitor geometry in the
    normalised I/Q observation space.  It is auxiliary evidence only; inner
    class-held-out episodes remain the source of genuine proxy unknowns.
    """

    purpose = "competition_pug_train"
    source_split = "train"
    formal_unknown = False
    force_unknown = True

    def __init__(
        self,
        samples: torch.Tensor,
        source_indices: np.ndarray,
        source_classes: np.ndarray,
    ) -> None:
        samples = torch.as_tensor(samples, dtype=torch.float32).contiguous()
        source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
        source_classes = np.asarray(source_classes, dtype=np.int64).reshape(-1)
        if samples.ndim != 3 or samples.shape[1] != 2:
            raise ValueError("PUG samples must have shape [N,2,L]")
        if not (len(samples) == len(source_indices) == len(source_classes)):
            raise ValueError("PUG samples and provenance vectors must align")
        if not len(samples) or not torch.isfinite(samples).all():
            raise ValueError("PUG samples must be non-empty and finite")
        self.x = samples
        self.y = np.full(len(samples), -1, dtype=np.int64)
        self.source_indices = source_indices
        self.source_classes = source_classes

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], torch.tensor(-1, dtype=torch.long)


def _dataset_of(value):
    """Return a loader's dataset without unwrapping provenance subsets.

    ``ProvenanceSubset`` deliberately exposes its source as ``.dataset``.
    The old duck-typed implementation therefore replaced every registered
    class/sample subset by the full source split during prediction and during
    provenance checks.  Only a real ``DataLoader`` should be unwrapped here;
    ordinary Dataset wrappers must remain intact so their labels, purpose and
    formal-unknown flags continue to guard the protocol.
    """
    return value.dataset if isinstance(value, DataLoader) else value


def _dataset_labels(dataset) -> np.ndarray | None:
    labels = getattr(dataset, "y", None)
    if labels is None:
        return None
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def _assert_training_dataset(dataset, role: str) -> None:
    """Validate the provenance and label role of one training dataset.

    ``role='proxy'`` is intentionally stricter than checking for negative
    labels: only an inner episode's ``train_proxy_unknown`` partition is
    legal.  This catches the easy-to-make but invalid use of
    ``outer.test_proxy_unknown`` when an outer model is trained.
    """

    dataset = _dataset_of(dataset)
    assert_no_formal_unknown(dataset)
    if bool(getattr(dataset, "formal_unknown", False)):
        raise RuntimeError("formal unknown data entered Stage-7 training")
    source_split = str(getattr(dataset, "source_split", "")).lower()
    purpose = str(getattr(dataset, "purpose", "")).lower()
    if source_split.startswith("test") or "test_proxy_unknown" in purpose:
        raise RuntimeError("test/proxy-test data cannot enter Stage-7 training")
    labels = _dataset_labels(dataset)
    if role == "known":
        if labels is not None and (not len(labels) or np.any(labels < 0)):
            raise ValueError("train_known must contain only non-negative labels")
    elif role == "proxy":
        if "train_proxy_unknown" not in purpose:
            raise RuntimeError(
                "proxy-unknown training requires an inner train_proxy_unknown partition")
        if labels is not None and (not len(labels) or np.any(labels >= 0)):
            raise ValueError("train_proxy_unknown must expose only label -1")
    elif role == "pug":
        if "pug" not in purpose:
            raise RuntimeError("PUG training data require explicit PUG provenance")
        if labels is not None and (not len(labels) or np.any(labels >= 0)):
            raise ValueError("PUG data must expose only unknown labels")
    else:
        raise ValueError(f"unsupported training dataset role: {role}")


def _require_batch_role(labels: torch.Tensor, role: str) -> None:
    if labels.ndim != 1:
        raise ValueError("training labels must be a vector")
    if role == "known" and bool(labels.lt(0).any()):
        raise ValueError("Known batch contains an unknown label")
    if role != "known" and bool(labels.ge(0).any()):
        raise ValueError(f"{role} batch contains a Known label")


def local_agent_objective(
    decisions: Mapping[str, LocalDecision],
    labels: torch.Tensor,
    *,
    class_weight: float = 1.0,
    open_weight: float = 1.0,
    competence_weight: float = 0.0,
    competence_target_mode: str = "identity_or_zero",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum independent Agent losses for a Known or proxy-unknown batch.

    Every Agent receives its own BCE rejection loss and, for Known rows, its
    own classification loss.  There is no fused/team term here, so local
    capability cannot be delegated to B2 or to communication.
    """

    if competence_weight < 0:
        raise ValueError("competence_weight must be non-negative")
    if competence_target_mode not in {
            "identity_or_zero", "local_decision_success"}:
        raise ValueError(
            f"unsupported competence_target_mode: {competence_target_mode}")
    labels = labels.long()
    unknown = labels.lt(0)
    targets = unknown.to(dtype=torch.float32)
    losses: dict[str, torch.Tensor] = {}
    for name in LOCAL_AGENT_NAMES:
        if name not in decisions:
            raise KeyError(f"missing local Agent decision: {name}")
        decision = decisions[name]
        if len(decision.unknown_score) != len(labels):
            raise ValueError("local decisions and labels are not aligned")
        loss = float(open_weight) * F.binary_cross_entropy_with_logits(
            decision.unknown_score.reshape(-1), targets, reduction="mean")
        known = ~unknown
        if bool(known.any()):
            loss = loss + float(class_weight) * F.cross_entropy(
                decision.class_logits[known], labels[known])
        if competence_weight:
            # Reliability must answer a different question from ordinary
            # Known/Unknown risk: "is this Agent's own identity decision
            # trustworthy?"  The true-class probability is a smooth,
            # stop-gradient competence target on Known rows.  In the
            # ``local_decision_success`` mode proxy/PUG rows use rejection
            # probability, so correctly rejecting an Unknown means *high*
            # competence rather than low competence.  This keeps the loss
            # local to each Agent and lets
            # a public selector compare meaningful self-assessments without
            # exposing private state.
            competence_target = torch.zeros_like(decision.unknown_score)
            if bool(known.any()):
                probability = F.softmax(
                    decision.class_logits[known].detach(), dim=1)
                identity_success = probability.gather(
                    1, labels[known, None]).squeeze(1)
                if competence_target_mode == "local_decision_success":
                    identity_success = identity_success * torch.sigmoid(
                        -decision.unknown_score[known].detach())
                competence_target[known] = identity_success
            if (competence_target_mode == "local_decision_success"
                    and bool(unknown.any())):
                competence_target[unknown] = torch.sigmoid(
                    decision.unknown_score[unknown].detach())
            competence = F.binary_cross_entropy(
                decision.reliability.reshape(-1).clamp(1e-6, 1.0 - 1e-6),
                competence_target, reduction="mean")
            loss = loss + float(competence_weight) * competence
        losses[name] = loss
    # Keep each local Agent at unit task-loss scale.  Averaging the two losses
    # halves both classification gradients while leaving SupCon/view/metric
    # regularisers unchanged, which made a nominally identical 30-epoch run
    # optimise a materially different objective from the proven independent
    # baselines.  A sum also reflects the protocol: neither Agent may delegate
    # its local capability to its peer.
    return torch.stack(tuple(losses.values())).sum(), losses


def supervised_contrastive_loss(
    state: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Numerically stable supervised contrastive loss for Known batches."""

    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    state = F.normalize(state, dim=-1)
    similarity = state @ state.T / float(temperature)
    similarity = similarity - similarity.max(1, keepdim=True).values.detach()
    diagonal = torch.eye(len(state), dtype=torch.bool, device=state.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    exp_similarity = similarity.exp() * (~diagonal)
    log_probability = similarity - torch.log(
        exp_similarity.sum(1, keepdim=True).clamp_min(1e-8))
    positive_count = positive.sum(1)
    valid = positive_count.gt(0)
    if not bool(valid.any()):
        return similarity.new_zeros(())
    return -(
        (log_probability * positive).sum(1)[valid]
        / positive_count[valid]
    ).mean()


def local_metric_objective(
    system: nn.Module,
    context,
    labels: torch.Tensor,
    *,
    supcon_temperature: float = 0.1,
    waveform_supcon_weight: float = 0.1,
    prototype_supcon_weight: float = 0.1,
    view_aux_weight: float = 0.25,
    prototype_compact_weight: float = 0.1,
    prototype_margin_weight: float = 0.1,
    prototype_margin: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train strong, distinct private representations on Known labels only.

    Nothing from this objective is copied into the public message contract.
    The Waveform role receives a supervised-contrastive constraint, while the
    Enrollment role receives independent per-view supervision plus compact and
    rival-margin metric constraints.  This is deliberately dataset agnostic.
    """

    if bool(labels.lt(0).any()):
        raise ValueError("metric representation loss accepts Known labels only")
    private = system.agents.training_private(context)
    waveform = private["waveform"]
    prototype = private["prototype"]
    waveform_supcon = supervised_contrastive_loss(
        waveform.state, labels, supcon_temperature)
    prototype_supcon = supervised_contrastive_loss(
        prototype.state, labels, supcon_temperature)

    view_count = int(prototype.view_logits.shape[1])
    expanded_labels = labels[:, None].expand(-1, view_count).reshape(-1)
    view_aux = F.cross_entropy(
        prototype.view_logits.flatten(0, 1), expanded_labels)
    distances = prototype.distances
    own = distances.gather(1, labels[:, None]).squeeze(1)
    rival = distances.clone()
    rival.scatter_(1, labels[:, None], float("inf"))
    nearest_rival = rival.min(1).values
    compact = own.mean()
    margin = F.relu(float(prototype_margin) + own - nearest_rival).mean()
    pieces = {
        "waveform_supcon": waveform_supcon,
        "prototype_supcon": prototype_supcon,
        "prototype_view_aux": view_aux,
        "prototype_compact": compact,
        "prototype_margin": margin,
    }
    total = (
        float(waveform_supcon_weight) * waveform_supcon
        + float(prototype_supcon_weight) * prototype_supcon
        + float(view_aux_weight) * view_aux
        + float(prototype_compact_weight) * compact
        + float(prototype_margin_weight) * margin)
    return total, pieces


def _next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _set_agent_training_mode_for_trainable_parameters(system: nn.Module) -> None:
    """Enable stochastic/train-time behaviour only for trainable Agent parts.

    Stage-7 first trains both complete local Agents and then freezes their
    encoders while fitting only the bounded rejection residuals.  Calling
    ``system.agents.train()`` unconditionally in that second phase used to
    mutate frozen BatchNorm running statistics and enable frozen dropout,
    despite every backbone parameter having ``requires_grad=False``.  Besides
    violating the frozen-backbone protocol, this made checkpoint resume
    dependent on how many open-head batches had already been consumed.

    When every Agent parameter is trainable we retain normal module-wide
    training mode.  For a partially frozen Agent tree we start in evaluation
    mode and re-enable only leaf modules which directly own trainable
    parameters.  The current open heads contain LayerNorm/Linear layers, so
    their gradients remain enabled while all backbone BatchNorm/dropout state
    is exactly frozen.
    """

    parameters = tuple(system.agents.parameters())
    if parameters and all(parameter.requires_grad for parameter in parameters):
        system.agents.train()
        return
    system.agents.eval()
    for module in system.agents.modules():
        if any(parameter.requires_grad
               for parameter in module.parameters(recurse=False)):
            module.train()


def train_local_agents_epoch(
    system: nn.Module,
    known_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    proxy_unknown_loader=None,
    pug_loader=None,
    class_weight: float = 1.0,
    open_weight: float = 1.0,
    competence_weight: float = 0.0,
    competence_target_mode: str = "identity_or_zero",
    proxy_unknown_weight: float = 1.0,
    pug_auxiliary_weight: float = 0.25,
    supcon_temperature: float = 0.1,
    waveform_supcon_weight: float = 0.1,
    prototype_supcon_weight: float = 0.1,
    view_aux_weight: float = 0.25,
    prototype_compact_weight: float = 0.1,
    prototype_margin_weight: float = 0.1,
    prototype_margin: float = 0.5,
    clip_grad_norm: float = 5.0,
) -> dict[str, float]:
    """Run one balanced epoch of independent local-Agent optimisation."""

    _assert_training_dataset(known_loader, "known")
    if proxy_unknown_loader is not None:
        _assert_training_dataset(proxy_unknown_loader, "proxy")
    if pug_loader is not None:
        _assert_training_dataset(pug_loader, "pug")
    if len(known_loader) == 0:
        raise ValueError("known_loader must not be empty")
    if proxy_unknown_weight < 0 or pug_auxiliary_weight < 0:
        raise ValueError("open-set loss weights must be non-negative")

    _set_agent_training_mode_for_trainable_parameters(system)
    proxy_iterator = iter(proxy_unknown_loader) if proxy_unknown_loader is not None else None
    pug_iterator = iter(pug_loader) if pug_loader is not None else None
    totals = {
        "loss": 0.0, "known_loss": 0.0,
        "proxy_open_loss": 0.0, "pug_open_loss": 0.0,
        "metric_loss": 0.0, "waveform_supcon": 0.0,
        "prototype_supcon": 0.0, "prototype_view_aux": 0.0,
        "prototype_compact": 0.0, "prototype_margin": 0.0,
        "waveform_known_accuracy": 0.0,
        "prototype_known_accuracy": 0.0,
    }
    seen = 0
    steps = 0
    parameters = [parameter for parameter in system.agents.parameters()
                  if parameter.requires_grad]
    for x_known, y_known in known_loader:
        x_known, y_known = x_known.to(device), y_known.to(device)
        _require_batch_role(y_known, "known")
        known_context = system.local(x_known)
        known_loss, _ = local_agent_objective(
            known_context, y_known, class_weight=class_weight,
            open_weight=open_weight, competence_weight=competence_weight,
            competence_target_mode=competence_target_mode)
        metric_loss, metric_pieces = local_metric_objective(
            system, known_context, y_known,
            supcon_temperature=supcon_temperature,
            waveform_supcon_weight=waveform_supcon_weight,
            prototype_supcon_weight=prototype_supcon_weight,
            view_aux_weight=view_aux_weight,
            prototype_compact_weight=prototype_compact_weight,
            prototype_margin_weight=prototype_margin_weight,
            prototype_margin=prototype_margin,
        )
        weighted_losses = [known_loss + metric_loss]
        weights = [1.0]
        proxy_value = known_loss.new_zeros(())
        pug_value = known_loss.new_zeros(())

        if proxy_unknown_loader is not None:
            (x_proxy, y_proxy), proxy_iterator = _next_or_restart(
                proxy_iterator, proxy_unknown_loader)
            x_proxy, y_proxy = x_proxy.to(device), y_proxy.to(device)
            _require_batch_role(y_proxy, "proxy")
            proxy_value, _ = local_agent_objective(
                system.local(x_proxy), y_proxy, class_weight=class_weight,
                open_weight=open_weight, competence_weight=competence_weight,
                competence_target_mode=competence_target_mode)
            if proxy_unknown_weight:
                weighted_losses.append(proxy_value * float(proxy_unknown_weight))
                weights.append(float(proxy_unknown_weight))

        if pug_loader is not None:
            (x_pug, y_pug), pug_iterator = _next_or_restart(pug_iterator, pug_loader)
            x_pug, y_pug = x_pug.to(device), y_pug.to(device)
            _require_batch_role(y_pug, "pug")
            pug_value, _ = local_agent_objective(
                system.local(x_pug), y_pug, class_weight=class_weight,
                open_weight=open_weight, competence_weight=competence_weight,
                competence_target_mode=competence_target_mode)
            if pug_auxiliary_weight:
                weighted_losses.append(pug_value * float(pug_auxiliary_weight))
                weights.append(float(pug_auxiliary_weight))

        loss = torch.stack(weighted_losses).sum() / max(sum(weights), 1e-12)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
        optimizer.step()

        batch = len(y_known)
        totals["loss"] += float(loss.detach()) * batch
        totals["known_loss"] += float(known_loss.detach()) * batch
        totals["metric_loss"] += float(metric_loss.detach()) * batch
        for name, value in metric_pieces.items():
            totals[name] += float(value.detach()) * batch
        totals["proxy_open_loss"] += float(proxy_value.detach()) * batch
        totals["pug_open_loss"] += float(pug_value.detach()) * batch
        for name in LOCAL_AGENT_NAMES:
            totals[f"{name}_known_accuracy"] += int(
                (known_context[name].class_logits.argmax(1) == y_known).sum())
        seen += batch
        steps += 1
    return {
        key: (value / max(seen, 1)) for key, value in totals.items()
    } | {"steps": float(steps)}


def _loader(
    dataset,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=generator, num_workers=0, drop_last=False)


def train_local_agents(
    system: nn.Module,
    train_known,
    proxy_unknown,
    cfg: Mapping[str, Any],
    device: torch.device,
    seed: int,
    *,
    pug_dataset=None,
) -> tuple[list[dict[str, float]], int]:
    """Train both local Agents with one dataset-independent recipe.

    ``proxy_unknown`` may be ``None`` for an outer model.  When present it
    must be an inner episode's training proxy partition; evaluation proxies
    and formal unknowns are rejected before the optimiser is constructed.
    """

    _assert_training_dataset(train_known, "known")
    if proxy_unknown is not None:
        _assert_training_dataset(proxy_unknown, "proxy")
    if pug_dataset is not None:
        _assert_training_dataset(pug_dataset, "pug")
    set_seed(seed)
    system.to(device)
    epochs = int(cfg.get("local_pretrain_epochs", 30))
    batch_size = int(cfg.get("batch_size", 256))
    if epochs < 1 or batch_size < 1:
        raise ValueError("local training requires positive epochs and batch_size")
    optimizer = torch.optim.AdamW(
        system.agents.parameters(), lr=float(cfg.get("local_lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)))
    generator = torch.Generator().manual_seed(int(seed))
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        known_loader = _loader(
            train_known, batch_size, shuffle=True, generator=generator)
        proxy_loader = (_loader(
            proxy_unknown, batch_size, shuffle=True, generator=generator)
            if proxy_unknown is not None else None)
        pug_loader = (_loader(
            pug_dataset, batch_size, shuffle=True, generator=generator)
            if pug_dataset is not None else None)
        row = train_local_agents_epoch(
            system, known_loader, optimizer, device,
            proxy_unknown_loader=proxy_loader, pug_loader=pug_loader,
            class_weight=float(cfg.get("class_loss_weight", 1.0)),
            open_weight=float(cfg.get("open_loss_weight", 1.0)),
            competence_weight=float(cfg.get("competence_weight", 0.0)),
            competence_target_mode=str(
                cfg.get("competence_target_mode", "identity_or_zero")),
            proxy_unknown_weight=float(cfg.get("proxy_unknown_weight", 1.0)),
            pug_auxiliary_weight=float(cfg.get("pug_auxiliary_weight", 0.25)),
            supcon_temperature=float(cfg.get("supcon_temperature", 0.1)),
            waveform_supcon_weight=float(
                cfg.get("waveform_supcon_weight", cfg.get("supcon_weight", 0.1))),
            prototype_supcon_weight=float(
                cfg.get("prototype_supcon_weight", cfg.get("supcon_weight", 0.1))),
            view_aux_weight=float(cfg.get("view_aux_weight", 0.25)),
            prototype_compact_weight=float(
                cfg.get("prototype_compact_weight", cfg.get("compact_weight", 0.1))),
            prototype_margin_weight=float(
                cfg.get("prototype_margin_weight", cfg.get("margin_weight", 0.1))),
            prototype_margin=float(cfg.get("prototype_margin", 0.5)),
            clip_grad_norm=float(cfg.get("clip_grad_norm", 5.0)),
        )
        row["epoch"] = float(epoch)
        history.append(row)
    # Epoch count is protocol-fixed; no synthetic/proxy score selects a lucky
    # checkpoint.  The experiment layer may compare whole frozen candidates.
    return history, epochs


@torch.no_grad()
def refresh_enrollment(
    system: nn.Module,
    dataset,
    device: torch.device,
    *,
    batch_size: int = 512,
) -> dict[str, float]:
    """Rebuild Prototype enrollment and tail memory from train-Known only."""

    _assert_training_dataset(dataset, "known")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    agents = system.agents
    waveform = agents.waveform
    prototype = agents.prototype
    training_modes = _capture_training_modes(agents)
    agents.eval()
    waveform_states, waveform_logits, prototype_states = [], [], []
    prototype_view_states, prototype_view_weights, labels = [], [], []
    for x, y in _loader(dataset, batch_size, shuffle=False):
        _require_batch_role(y, "known")
        context = system.local(x.to(device))
        private = agents.training_private(context)
        waveform_states.append(private["waveform"].state.detach())
        waveform_logits.append(context["waveform"].class_logits.detach())
        prototype_states.append(private["prototype"].state.detach())
        prototype_view_states.append(
            private["prototype"].view_states.detach())
        prototype_view_weights.append(
            private["prototype"].view_weights.detach())
        labels.append(y.to(device))
    if not prototype_states:
        raise ValueError("cannot refresh enrollment from an empty dataset")
    all_labels = torch.cat(labels)
    expected = torch.arange(prototype.num_classes, device=all_labels.device)
    observed = torch.unique(all_labels, sorted=True)
    if not torch.equal(observed, expected):
        raise ValueError(
            "enrollment requires every mapped Known class exactly in [0,C)")
    waveform.set_enrollment_memory(
        torch.cat(waveform_states), torch.cat(waveform_logits), all_labels)
    prototype.set_enrollment_memory(
        torch.cat(prototype_states), torch.cat(prototype_view_states),
        torch.cat(prototype_view_weights), all_labels)
    _restore_training_modes(training_modes)
    return {
        "num_samples": float(len(all_labels)),
        "num_classes": float(prototype.num_classes),
        "tail_location_mean": float(prototype.tail_location.mean().cpu()),
        "tail_scale_mean": float(prototype.tail_scale.mean().cpu()),
        "waveform_tail_location_mean": float(
            waveform.tail_location.mean().cpu()),
        "num_prototype_views": float(len(prototype.view_names)),
    }


@torch.no_grad()
def make_competition_pug(
    system: nn.Module,
    dataset,
    device: torch.device,
    *,
    eta: float = 1.5,
    max_per_sample: int = 1,
    repel_weight: float = 0.35,
    jitter: float = 0.10,
    noise: float = 0.03,
    batch_size: int = 512,
    seed: int = 42,
) -> CompetitionPUGDataset:
    """Generate normalised competition-boundary proposals from train Known.

    Rival classes are selected by the two local Agents' mean class logits.
    Each sample moves away from its class mean and away from the selected
    competitor mean.  Channel statistics are restored afterwards so the open
    heads cannot solve the auxiliary task from a trivial amplitude artefact.
    """

    _assert_training_dataset(dataset, "known")
    if eta <= 0 or max_per_sample < 1:
        raise ValueError("eta and max_per_sample must be positive")
    if not 0.0 <= repel_weight <= 1.0 or jitter < 0 or noise < 0:
        raise ValueError("invalid competition-PUG geometry")
    training_modes = _capture_training_modes(system)
    system.eval()
    samples, labels, logits = [], [], []
    for x, y in _loader(dataset, batch_size, shuffle=False):
        _require_batch_role(y, "known")
        local = system.local(x.to(device))
        team_logits = 0.5 * (
            local["waveform"].class_logits + local["prototype"].class_logits)
        samples.append(x.float().cpu())
        labels.append(y.long().cpu())
        logits.append(team_logits.cpu())
    _restore_training_modes(training_modes)
    if not samples:
        raise ValueError("cannot generate PUG from an empty dataset")
    x = torch.cat(samples)
    y = torch.cat(labels)
    team_logits = torch.cat(logits)
    num_classes = int(team_logits.shape[1])
    observed = torch.unique(y, sorted=True)
    if not torch.equal(observed, torch.arange(num_classes)):
        raise ValueError("PUG generation requires all mapped Known classes")

    class_means = torch.stack([x[y == cls].mean(0) for cls in range(num_classes)])
    rival_logits = team_logits.clone()
    rival_logits.scatter_(1, y[:, None], -torch.inf)
    rival = rival_logits.argmax(1)
    own_mean = class_means[y]
    rival_mean = class_means[rival]

    def rms_normalise(value: torch.Tensor) -> torch.Tensor:
        scale = value.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        return value / scale

    outward_raw = x - own_mean
    outward = rms_normalise(outward_raw)
    repel = rms_normalise(x - rival_mean)
    direction = rms_normalise(
        (1.0 - float(repel_weight)) * outward + float(repel_weight) * repel)
    local_scale = outward_raw.square().mean(dim=(1, 2), keepdim=True).sqrt()
    fallback = x.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-3)
    local_scale = torch.where(local_scale > 1e-5, local_scale, 0.05 * fallback)
    source_mean = x.mean(dim=2, keepdim=True)
    source_std = x.std(dim=2, keepdim=True, unbiased=False).clamp_min(1e-5)

    rng = torch.Generator().manual_seed(int(seed))
    proposals, sources, classes = [], [], []
    for variation in range(int(max_per_sample)):
        random_eta = float(eta) * (
            1.0 + (2.0 * torch.rand((len(x), 1, 1), generator=rng) - 1.0)
            * float(jitter))
        perturbation = random_eta * local_scale * direction
        if noise:
            random_noise = torch.randn(x.shape, generator=rng)
            random_noise = rms_normalise(random_noise)
            perturbation = perturbation + float(noise) * local_scale * random_noise
        candidate = x + perturbation
        # Match each source window's per-channel first and second moments.
        candidate = (candidate - candidate.mean(dim=2, keepdim=True))
        candidate = candidate / candidate.std(
            dim=2, keepdim=True, unbiased=False).clamp_min(1e-5)
        candidate = candidate * source_std + source_mean
        proposals.append(candidate)
        sources.append(np.arange(len(x), dtype=np.int64))
        classes.append(y.numpy())
    return CompetitionPUGDataset(
        torch.cat(proposals), np.concatenate(sources), np.concatenate(classes))


def local_agent_sample_losses(
    decisions: Mapping[str, LocalDecision],
    labels: torch.Tensor,
    *,
    class_weight: float = 1.0,
    open_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total/class/open per-sample losses for both local Agents.

    Columns always follow ``(waveform, prototype)``.  This helper consumes
    only :class:`LocalDecision` values and is therefore safe for B2 anchor
    selection, selector distillation and no-regret supervision.
    """

    labels = labels.long()
    unknown = labels.lt(0)
    class_losses, open_losses = [], []
    for name in LOCAL_AGENT_NAMES:
        decision = decisions[name]
        if len(decision.unknown_score) != len(labels):
            raise ValueError("local decisions and labels are not aligned")
        open_loss = float(open_weight) * F.binary_cross_entropy_with_logits(
            decision.unknown_score.reshape(-1), unknown.float(), reduction="none")
        class_loss = open_loss.new_zeros(len(labels))
        known = ~unknown
        if bool(known.any()):
            class_loss[known] = float(class_weight) * F.cross_entropy(
                decision.class_logits[known], labels[known], reduction="none")
        class_losses.append(class_loss)
        open_losses.append(open_loss)
    class_matrix = torch.stack(class_losses, dim=1)
    open_matrix = torch.stack(open_losses, dim=1)
    return class_matrix + open_matrix, class_matrix, open_matrix


def b2_no_regret_objective(
    system: nn.Module,
    decisions: Mapping[str, LocalDecision],
    output: DialogueOutput,
    labels: torch.Tensor,
    *,
    class_weight: float = 1.0,
    open_weight: float = 1.0,
    no_regret_weight: float = 1.0,
    selector_distillation_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Task loss plus explicit best-local no-regret supervision.

    The hinge compares B2 with the lower *actual* local OSR loss for every
    training sample.  Two public selectors are also distilled toward the
    lower-loss expert for identity and rejection separately.  This is not a
    claim that an unseen distribution can be mathematically guaranteed; the
    exact fallback in :class:`FairB2Fusion` provides the empirical safeguard
    after fitting.
    """

    if no_regret_weight < 0 or selector_distillation_weight < 0:
        raise ValueError("B2 no-regret weights must be non-negative")
    team_sample = dialogue_sample_loss(
        output, labels, class_weight=class_weight, open_weight=open_weight)
    local_total, local_class, local_open = local_agent_sample_losses(
        decisions, labels, class_weight=class_weight, open_weight=open_weight)
    best_local = local_total.min(dim=1).values.detach()
    regret = F.relu(team_sample - best_local)

    class_weights, open_weights = system.dialogue.b2_fusion.mixture_weights(decisions)
    known = labels.ge(0)
    class_selector = team_sample.new_zeros(())
    if bool(known.any()):
        class_target = local_class[known].argmin(dim=1)
        class_selector = F.nll_loss(
            class_weights[known].clamp_min(1e-8).log(), class_target)
    open_target = local_open.argmin(dim=1)
    open_selector = F.nll_loss(
        open_weights.clamp_min(1e-8).log(), open_target)
    selector = class_selector + open_selector
    objective = (
        team_sample.mean()
        + float(no_regret_weight) * regret.mean()
        + float(selector_distillation_weight) * selector)
    return objective, {
        "task_loss": team_sample.mean(),
        "regret": regret.mean(),
        "selector_loss": selector,
        "best_local_loss": best_local.mean(),
    }


def _b2_group_loss(
    system: nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weight: float,
    open_weight: float,
    no_regret_weight: float,
    selector_distillation_weight: float,
) -> tuple[torch.Tensor, DialogueOutput, dict[str, torch.Tensor]]:
    # No gradients or private tensors from either Agent enter the B2 fit.
    with torch.no_grad():
        local = system.local(x)
    output = system.b2(local)
    loss, diagnostics = b2_no_regret_objective(
        system, local, output, labels,
        class_weight=class_weight, open_weight=open_weight,
        no_regret_weight=no_regret_weight,
        selector_distillation_weight=selector_distillation_weight)
    return loss, output, diagnostics


def train_fair_b2_epoch(
    system: nn.Module,
    known_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    proxy_unknown_loader=None,
    pug_loader=None,
    class_weight: float = 1.0,
    open_weight: float = 1.0,
    proxy_unknown_weight: float = 1.0,
    pug_auxiliary_weight: float = 0.25,
    no_regret_weight: float = 1.0,
    selector_distillation_weight: float = 0.25,
    clip_grad_norm: float = 5.0,
) -> dict[str, float]:
    """Fit B2 from public local decisions while keeping Agents frozen."""

    _assert_training_dataset(known_loader, "known")
    if proxy_unknown_loader is not None:
        _assert_training_dataset(proxy_unknown_loader, "proxy")
    if pug_loader is not None:
        _assert_training_dataset(pug_loader, "pug")
    if len(known_loader) == 0:
        raise ValueError("known_loader must not be empty")
    system.agents.eval()
    system.dialogue.b2_fusion.train()
    proxy_iterator = iter(proxy_unknown_loader) if proxy_unknown_loader is not None else None
    pug_iterator = iter(pug_loader) if pug_loader is not None else None
    parameters = list(system.dialogue.b2_fusion.parameters())
    totals = {"loss": 0.0, "known_loss": 0.0,
              "proxy_open_loss": 0.0, "pug_open_loss": 0.0,
              "known_accuracy": 0.0, "no_regret": 0.0,
              "selector_loss": 0.0, "best_local_loss": 0.0}
    seen = 0
    for x_known, y_known in known_loader:
        x_known, y_known = x_known.to(device), y_known.to(device)
        _require_batch_role(y_known, "known")
        known_loss, known_output, known_diagnostics = _b2_group_loss(
            system, x_known, y_known, class_weight=class_weight,
            open_weight=open_weight, no_regret_weight=no_regret_weight,
            selector_distillation_weight=selector_distillation_weight)
        values, weights = [known_loss], [1.0]
        proxy_value = known_loss.new_zeros(())
        pug_value = known_loss.new_zeros(())
        if proxy_unknown_loader is not None:
            (x_proxy, y_proxy), proxy_iterator = _next_or_restart(
                proxy_iterator, proxy_unknown_loader)
            x_proxy, y_proxy = x_proxy.to(device), y_proxy.to(device)
            _require_batch_role(y_proxy, "proxy")
            proxy_value, _, _ = _b2_group_loss(
                system, x_proxy, y_proxy, class_weight=class_weight,
                open_weight=open_weight, no_regret_weight=no_regret_weight,
                selector_distillation_weight=selector_distillation_weight)
            if proxy_unknown_weight:
                values.append(proxy_value * float(proxy_unknown_weight))
                weights.append(float(proxy_unknown_weight))
        if pug_loader is not None:
            (x_pug, y_pug), pug_iterator = _next_or_restart(pug_iterator, pug_loader)
            x_pug, y_pug = x_pug.to(device), y_pug.to(device)
            _require_batch_role(y_pug, "pug")
            pug_value, _, _ = _b2_group_loss(
                system, x_pug, y_pug, class_weight=class_weight,
                open_weight=open_weight, no_regret_weight=no_regret_weight,
                selector_distillation_weight=selector_distillation_weight)
            if pug_auxiliary_weight:
                values.append(pug_value * float(pug_auxiliary_weight))
                weights.append(float(pug_auxiliary_weight))
        loss = torch.stack(values).sum() / max(sum(weights), 1e-12)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
        optimizer.step()
        batch = len(y_known)
        totals["loss"] += float(loss.detach()) * batch
        totals["known_loss"] += float(known_loss.detach()) * batch
        totals["proxy_open_loss"] += float(proxy_value.detach()) * batch
        totals["pug_open_loss"] += float(pug_value.detach()) * batch
        totals["no_regret"] += float(known_diagnostics["regret"].detach()) * batch
        totals["selector_loss"] += float(
            known_diagnostics["selector_loss"].detach()) * batch
        totals["best_local_loss"] += float(
            known_diagnostics["best_local_loss"].detach()) * batch
        totals["known_accuracy"] += int(
            (known_output.fused_class_logits.argmax(1) == y_known).sum())
        seen += batch
    return {key: value / max(seen, 1) for key, value in totals.items()}


@torch.no_grad()
def _b2_dataset_statistics(
    system: nn.Module,
    dataset,
    device: torch.device,
    *,
    batch_size: int,
    class_weight: float,
    open_weight: float,
) -> tuple[torch.Tensor, float]:
    """Mean local and current-B2 task losses on one registered partition."""

    local_sum = torch.zeros(2, dtype=torch.float64)
    team_sum = 0.0
    seen = 0
    for x, labels in _loader(dataset, batch_size, shuffle=False):
        x, labels = x.to(device), labels.to(device)
        local = system.local(x)
        local_loss, _, _ = local_agent_sample_losses(
            local, labels, class_weight=class_weight, open_weight=open_weight)
        output = system.b2(local)
        team_loss = dialogue_sample_loss(
            output, labels, class_weight=class_weight, open_weight=open_weight)
        local_sum += local_loss.double().sum(0).cpu()
        team_sum += float(team_loss.double().sum().cpu())
        seen += len(labels)
    if not seen:
        raise ValueError("B2 fitting partition must not be empty")
    return local_sum / seen, team_sum / seen


@torch.no_grad()
def configure_fair_b2_anchor(
    system: nn.Module,
    train_known,
    proxy_unknown,
    pug_dataset,
    device: torch.device,
    *,
    batch_size: int = 256,
    class_weight: float = 1.0,
    open_weight: float = 1.0,
    proxy_unknown_weight: float = 1.0,
    pug_auxiliary_weight: float = 0.25,
) -> dict[str, float | str]:
    """Choose the best global local expert on leakage-safe fitting data.

    A real inner ``train_proxy_unknown`` partition dominates the auxiliary
    PUG term when supplied.  Formal and outer-test unknowns remain rejected by
    the same provenance guards used by every other Stage-7 training helper.
    """

    _assert_training_dataset(train_known, "known")
    groups = [(train_known, 1.0)]
    if proxy_unknown is not None:
        _assert_training_dataset(proxy_unknown, "proxy")
        if proxy_unknown_weight:
            groups.append((proxy_unknown, float(proxy_unknown_weight)))
    if pug_dataset is not None:
        _assert_training_dataset(pug_dataset, "pug")
        if pug_auxiliary_weight:
            groups.append((pug_dataset, float(pug_auxiliary_weight)))
    if any(weight < 0 for _, weight in groups):
        raise ValueError("B2 fitting weights must be non-negative")

    was_training = system.training
    system.eval()
    aggregate = torch.zeros(2, dtype=torch.float64)
    denominator = 0.0
    for dataset, weight in groups:
        local_loss, _ = _b2_dataset_statistics(
            system, dataset, device, batch_size=batch_size,
            class_weight=class_weight, open_weight=open_weight)
        aggregate += weight * local_loss
        denominator += weight
    aggregate /= max(denominator, 1e-12)
    anchor = int(aggregate.argmin().item())
    fusion = system.dialogue.b2_fusion
    fusion.set_anchor(anchor)
    fusion.set_adaptive_enabled(True)
    system.train(was_training)
    return {
        "anchor_agent": LOCAL_AGENT_NAMES[anchor],
        "waveform_objective": float(aggregate[0]),
        "prototype_objective": float(aggregate[1]),
    }


@torch.no_grad()
def _b2_weighted_fit_statistics(
    system: nn.Module,
    groups: Sequence[tuple[object, float]],
    device: torch.device,
    *,
    batch_size: int,
    class_weight: float,
    open_weight: float,
) -> tuple[torch.Tensor, float]:
    local = torch.zeros(2, dtype=torch.float64)
    team = 0.0
    denominator = 0.0
    was_training = system.training
    system.eval()
    for dataset, weight in groups:
        if not weight:
            continue
        local_value, team_value = _b2_dataset_statistics(
            system, dataset, device, batch_size=batch_size,
            class_weight=class_weight, open_weight=open_weight)
        local += float(weight) * local_value
        team += float(weight) * team_value
        denominator += float(weight)
    system.train(was_training)
    if denominator <= 0:
        raise ValueError("at least one positive B2 fitting weight is required")
    return local / denominator, team / denominator


def train_fair_b2(
    system: nn.Module,
    train_known,
    proxy_unknown,
    cfg: Mapping[str, Any],
    device: torch.device,
    seed: int,
    *,
    pug_dataset=None,
) -> list[dict[str, float]]:
    """Train the exact STOP/B2 parent without private feature access."""

    _assert_training_dataset(train_known, "known")
    if proxy_unknown is not None:
        _assert_training_dataset(proxy_unknown, "proxy")
    if pug_dataset is not None:
        _assert_training_dataset(pug_dataset, "pug")
    set_seed(seed)
    system.to(device)
    epochs = int(cfg.get("b2_epochs", cfg.get("response_epochs", 5)))
    batch_size = int(cfg.get("batch_size", 256))
    if epochs < 1 or batch_size < 1:
        raise ValueError("B2 training requires positive epochs and batch_size")
    class_weight = float(cfg.get("class_loss_weight", 1.0))
    open_weight = float(cfg.get("open_loss_weight", 1.0))
    proxy_weight = float(cfg.get("proxy_unknown_weight", 1.0))
    pug_weight = float(cfg.get("pug_auxiliary_weight", 0.25))
    no_regret_weight = float(cfg.get("b2_no_regret_weight", 1.0))
    selector_weight = float(cfg.get("b2_selector_distillation_weight", 0.25))
    anchor_report = configure_fair_b2_anchor(
        system, train_known, proxy_unknown, pug_dataset, device,
        batch_size=batch_size, class_weight=class_weight,
        open_weight=open_weight, proxy_unknown_weight=proxy_weight,
        pug_auxiliary_weight=pug_weight)
    parameters = list(system.dialogue.b2_fusion.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(cfg.get("b2_lr", cfg.get("response_lr", 5e-4))),
        weight_decay=float(cfg.get("weight_decay", 1e-4)))
    generator = torch.Generator().manual_seed(int(seed) + 101)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        row = train_fair_b2_epoch(
            system,
            _loader(train_known, batch_size, shuffle=True, generator=generator),
            optimizer, device,
            proxy_unknown_loader=(
                _loader(proxy_unknown, batch_size, shuffle=True, generator=generator)
                if proxy_unknown is not None else None),
            pug_loader=(
                _loader(pug_dataset, batch_size, shuffle=True, generator=generator)
                if pug_dataset is not None else None),
            class_weight=class_weight,
            open_weight=open_weight,
            proxy_unknown_weight=proxy_weight,
            pug_auxiliary_weight=pug_weight,
            no_regret_weight=no_regret_weight,
            selector_distillation_weight=selector_weight,
            clip_grad_norm=float(cfg.get("clip_grad_norm", 5.0)),
        )
        row["epoch"] = float(epoch)
        history.append(row)
    # Empirical no-regret safeguard.  Candidate and anchor are compared on the
    # exact same leakage-safe fitting partitions and task objective.  If the
    # adaptive selector does not beat its strongest local parent, deployment
    # becomes the exact parent (not an approximate penalty-based fallback).
    groups: list[tuple[object, float]] = [(train_known, 1.0)]
    if proxy_unknown is not None:
        groups.append((proxy_unknown, proxy_weight))
    if pug_dataset is not None:
        groups.append((pug_dataset, pug_weight))
    local_fit, candidate_fit = _b2_weighted_fit_statistics(
        system, groups, device, batch_size=batch_size,
        class_weight=class_weight, open_weight=open_weight)
    anchor_index = system.dialogue.b2_fusion.anchor_index
    if anchor_index is None:
        raise RuntimeError("B2 anchor was not configured")
    anchor_fit = float(local_fit[anchor_index])
    required_gain = float(cfg.get("b2_required_fit_gain", 0.0))
    adaptive_selected = candidate_fit + required_gain <= anchor_fit
    system.dialogue.b2_fusion.set_adaptive_enabled(adaptive_selected)
    history[-1].update({
        **anchor_report,
        "candidate_fit_objective": float(candidate_fit),
        "anchor_fit_objective": anchor_fit,
        "adaptive_selected": float(adaptive_selected),
        "empirical_no_regret": float(min(candidate_fit, anchor_fit)),
    })
    return history


@torch.no_grad()
def collect_predictions(
    system: nn.Module,
    dataset,
    device: torch.device,
    *,
    route_action: RouteAction | int | str | None = RouteAction.STOP,
    intervention: str | None = None,
    protocol: str = "sequential",
    batch_size: int = 1024,
    allow_formal_unknown: bool = False,
) -> dict[str, np.ndarray]:
    """Collect local, B2 or routed outputs with explicit unknown access.

    ``allow_formal_unknown=True`` is intentionally noisy at the call site and
    is reserved for the final evaluation phase.  Proxy unknown episode data do
    not need the override because they are not marked formal.
    """

    dataset = _dataset_of(dataset)
    if not allow_formal_unknown:
        assert_no_formal_unknown(dataset)
        if bool(getattr(dataset, "formal_unknown", False)):
            raise RuntimeError("formal unknown prediction requires explicit opt-in")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    system.eval()
    names = (
        "y", "pred", "raw_unknown", "known_weights", "route", "bits",
        "route_logits", "public_features",
        "pred_waveform", "raw_waveform", "reliability_waveform",
        "top1_waveform", "top2_waveform", "summary_waveform", "logits_waveform",
        "pred_prototype", "raw_prototype", "reliability_prototype",
        "top1_prototype", "top2_prototype", "summary_prototype", "logits_prototype",
    )
    rows: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for x, y in _loader(dataset, batch_size, shuffle=False):
        x = x.to(device)
        local = system.local(x)
        action = route_action
        if isinstance(route_action, str):
            if route_action != "reliability":
                raise ValueError(f"unsupported route policy: {route_action}")
            action = torch.where(
                local["waveform"].reliability >= local["prototype"].reliability,
                torch.full_like(local["waveform"].top1, int(RouteAction.W_FIRST)),
                torch.full_like(local["waveform"].top1, int(RouteAction.P_FIRST)),
            )
        output = system.deliberate(
            local, route_action=action, intervention=intervention,
            protocol=protocol)
        rows["y"].append(y.numpy())
        rows["pred"].append(output.fused_class_logits.argmax(1).cpu().numpy())
        rows["raw_unknown"].append(output.unknown_score.cpu().numpy())
        rows["known_weights"].append(output.known_weights.cpu().numpy())
        rows["route"].append(output.route_action.cpu().numpy())
        rows["bits"].append(output.bit_cost.cpu().numpy())
        route_logits = output.route_logits
        if route_logits is None:
            route_logits = system.route_logits(local)
        rows["route_logits"].append(route_logits.cpu().numpy())
        rows["public_features"].append(public_team_features(local).cpu().numpy())
        for name in LOCAL_AGENT_NAMES:
            decision = local[name]
            rows[f"pred_{name}"].append(
                decision.class_logits.argmax(1).cpu().numpy())
            rows[f"raw_{name}"].append(decision.unknown_score.cpu().numpy())
            rows[f"reliability_{name}"].append(decision.reliability.cpu().numpy())
            rows[f"top1_{name}"].append(decision.top1.cpu().numpy())
            rows[f"top2_{name}"].append(decision.top2.cpu().numpy())
            rows[f"summary_{name}"].append(decision.public_summary.cpu().numpy())
            rows[f"logits_{name}"].append(decision.class_logits.cpu().numpy())
    if not rows["y"]:
        raise ValueError("cannot collect predictions from an empty dataset")
    result = {name: np.concatenate(values, axis=0)
              for name, values in rows.items()}
    # A final formal call must still be read-only.  Returning sample keys aids
    # audit without passing the dataset back into any fitting helper.
    if hasattr(dataset, "sample_keys"):
        result["sample_keys"] = np.asarray(dataset.sample_keys, dtype=str)
    return result


@dataclass(frozen=True)
class CounterfactualSupervision:
    action_losses: torch.Tensor
    targets: torch.Tensor
    gains_over_stop: torch.Tensor
    bit_costs: torch.Tensor
    worthwhile: torch.Tensor

    def numpy(self) -> dict[str, np.ndarray]:
        return {
            "action_losses": self.action_losses.detach().cpu().numpy(),
            "targets": self.targets.detach().cpu().numpy(),
            "gains_over_stop": self.gains_over_stop.detach().cpu().numpy(),
            "bit_costs": self.bit_costs.detach().cpu().numpy(),
            "worthwhile": self.worthwhile.detach().cpu().numpy(),
        }


def dialogue_sample_loss(
    output: DialogueOutput,
    labels: torch.Tensor,
    is_unknown: torch.Tensor | None = None,
    *,
    class_weight: float = 1.0,
    open_weight: float = 1.0,
) -> torch.Tensor:
    """Per-sample OSR loss; ``unknown_score`` is a raw unknown logit."""
    labels = labels.long()
    unknown = labels.lt(0) if is_unknown is None else is_unknown.bool()
    if unknown.shape != labels.shape:
        raise ValueError("is_unknown must match labels")
    unknown_logit = output.unknown_score.reshape(-1)
    loss = float(open_weight) * F.binary_cross_entropy_with_logits(
        unknown_logit, unknown.float(), reduction="none")
    known = ~unknown
    if bool(known.any()):
        loss = loss.clone()
        loss[known] += float(class_weight) * F.cross_entropy(
            output.fused_class_logits[known], labels[known], reduction="none")
    return loss


class ExactCounterfactualSupervisor:
    """Enumerate STOP/W_FIRST/P_FIRST and label the actual best action.

    Unlike Stage-6's value critic, targets are never self-generated by the
    router.  Each action is executed with the same frozen local decisions and
    compared sample by sample.  Ties and non-positive gains fall back to STOP.
    """

    def __init__(
        self,
        *,
        communication_cost: float = 0.01,
        reference_bits: float = 64.0,
        minimum_gain: float = 0.0,
        class_weight: float = 1.0,
        open_weight: float = 1.0,
    ) -> None:
        if communication_cost < 0 or reference_bits <= 0 or minimum_gain < 0:
            raise ValueError("invalid counterfactual cost configuration")
        self.communication_cost = float(communication_cost)
        self.reference_bits = float(reference_bits)
        self.minimum_gain = float(minimum_gain)
        self.class_weight = float(class_weight)
        self.open_weight = float(open_weight)

    @torch.no_grad()
    def enumerate(
        self,
        system: nn.Module,
        local_context,
        labels: torch.Tensor,
        is_unknown: torch.Tensor | None = None,
    ) -> tuple[CounterfactualSupervision, dict[RouteAction, DialogueOutput]]:
        labels = labels.long()
        unknown = labels.lt(0) if is_unknown is None else is_unknown.bool()
        losses, costs, outputs = [], [], {}
        for action in ACTION_ORDER:
            route = torch.full(
                (len(labels),), int(action), dtype=torch.long, device=labels.device)
            output = system.deliberate(local_context, route_action=route)
            outputs[action] = output
            bit_cost = output.bit_cost
            if not torch.is_tensor(bit_cost):
                bit_cost = labels.new_full(labels.shape, float(bit_cost), dtype=torch.float32)
            bit_cost = bit_cost.float().reshape(-1)
            if len(bit_cost) == 1 and len(labels) != 1:
                bit_cost = bit_cost.expand(len(labels))
            if len(bit_cost) != len(labels):
                raise ValueError("DialogueOutput.bit_cost must be scalar or per-sample")
            task_loss = dialogue_sample_loss(
                output, labels, unknown, class_weight=self.class_weight,
                open_weight=self.open_weight)
            total_loss = task_loss + (
                self.communication_cost * bit_cost / self.reference_bits)
            losses.append(total_loss)
            costs.append(bit_cost)
        action_losses = torch.stack(losses, dim=1)
        bit_costs = torch.stack(costs, dim=1)
        targets = action_losses.argmin(dim=1)
        stop_loss = action_losses[:, int(RouteAction.STOP)]
        gains = stop_loss[:, None] - action_losses
        chosen_gain = gains.gather(1, targets[:, None]).squeeze(1)
        worthwhile = (targets != int(RouteAction.STOP)) & (
            chosen_gain > self.minimum_gain)
        targets = torch.where(
            worthwhile, targets, torch.full_like(targets, int(RouteAction.STOP)))
        return CounterfactualSupervision(
            action_losses.detach(), targets.detach(), gains.detach(),
            bit_costs.detach(), worthwhile.detach(),
        ), outputs

    @torch.no_grad()
    def from_batch(
        self, system: nn.Module, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[CounterfactualSupervision, dict[RouteAction, DialogueOutput]]:
        return self.enumerate(system, system.local(x), labels)


def assert_stop_matches_b2(
    system: nn.Module,
    local_context,
    *,
    atol: float = 0.0,
) -> None:
    """Hard invariant: STOP is the fair no-communication parent, bit-for-bit."""
    with torch.no_grad():
        sample = local_context["waveform"].class_logits
        route = torch.full(
            (len(sample),), int(RouteAction.STOP), dtype=torch.long,
            device=sample.device)
        stop = system.deliberate(local_context, route_action=route)
        baseline = system.b2(local_context)
    for name in ("fused_class_logits", "unknown_score"):
        left, right = getattr(stop, name), getattr(baseline, name)
        if not torch.allclose(left, right, atol=atol, rtol=0.0):
            raise AssertionError(f"STOP differs from B2 in {name}")
    if not torch.equal(stop.bit_cost, torch.zeros_like(stop.bit_cost)):
        raise AssertionError("STOP must have exactly zero communication cost")


def balanced_action_weights(targets: torch.Tensor, num_actions: int = 3) -> torch.Tensor:
    counts = torch.bincount(targets.long(), minlength=num_actions).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights


def forced_response_loss(
    system: nn.Module,
    local_context,
    labels: torch.Tensor,
    *,
    class_weight: float = 1.0,
    open_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train both conditional response directions with an exact 50/50 loss."""
    rows = []
    metrics = {}
    for action in (RouteAction.W_FIRST, RouteAction.P_FIRST):
        route = torch.full(
            (len(labels),), int(action), dtype=torch.long, device=labels.device)
        output = system.deliberate(local_context, route_action=route)
        value = dialogue_sample_loss(
            output, labels, class_weight=class_weight,
            open_weight=open_weight).mean()
        rows.append(value)
        metrics[action.name.lower()] = float(value.detach())
    return 0.5 * (rows[0] + rows[1]), metrics


def train_forced_responders_epoch(
    system: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    class_weight: float = 1.0,
    open_weight: float = 1.0,
    clip_grad_norm: float = 5.0,
) -> dict[str, float]:
    system.train()
    totals = {"loss": 0.0, "w_first": 0.0, "p_first": 0.0}
    count = 0
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        local = system.local(x)
        loss, parts = forced_response_loss(
            system, local, labels, class_weight=class_weight,
            open_weight=open_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups
             for parameter in group["params"]], clip_grad_norm)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["w_first"] += parts["w_first"]
        totals["p_first"] += parts["p_first"]
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_route_policy_epoch(
    system: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    supervisor: ExactCounterfactualSupervisor,
    device: torch.device,
    *,
    clip_grad_norm: float = 5.0,
) -> dict[str, float]:
    """Imitate exact action labels using public local summaries only."""
    system.train()
    total_loss = 0.0
    total = 0
    correct = 0
    target_counts = torch.zeros(3, dtype=torch.long)
    prediction_counts = torch.zeros(3, dtype=torch.long)
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        local = system.local(x)
        supervision, _ = supervisor.enumerate(system, local, labels)
        logits = system.route_logits(local)
        weights = balanced_action_weights(supervision.targets).to(logits.device)
        loss = F.cross_entropy(logits, supervision.targets, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups
             for parameter in group["params"]], clip_grad_norm)
        optimizer.step()
        prediction = logits.argmax(1)
        batch = len(labels)
        total_loss += float(loss.detach()) * batch
        total += batch
        correct += int((prediction == supervision.targets).sum())
        target_counts += torch.bincount(
            supervision.targets.cpu(), minlength=3)
        prediction_counts += torch.bincount(prediction.detach().cpu(), minlength=3)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        **{f"target_{ACTION_ORDER[index].name.lower()}_rate":
           int(target_counts[index]) / max(total, 1) for index in range(3)},
        **{f"predicted_{ACTION_ORDER[index].name.lower()}_rate":
           int(prediction_counts[index]) / max(total, 1) for index in range(3)},
    }


@torch.no_grad()
def collect_counterfactual_supervision(
    system: nn.Module,
    loader,
    supervisor: ExactCounterfactualSupervisor,
    device: torch.device,
    *,
    dataset=None,
) -> dict[str, np.ndarray]:
    if dataset is not None:
        assert_no_formal_unknown(dataset)
    system.eval()
    collected = {name: [] for name in (
        "labels", "action_losses", "targets", "gains_over_stop",
        "bit_costs", "worthwhile", "route_logits")}
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        local = system.local(x)
        supervision, _ = supervisor.enumerate(system, local, labels)
        collected["labels"].append(labels.cpu().numpy())
        for name, value in supervision.numpy().items():
            collected[name].append(value)
        collected["route_logits"].append(system.route_logits(local).cpu().numpy())
    return {name: np.concatenate(values, axis=0) for name, values in collected.items()}


def router_diagnostics(
    action_losses: np.ndarray,
    route_logits: np.ndarray,
) -> dict[str, float]:
    losses = np.asarray(action_losses, dtype=np.float64)
    logits = np.asarray(route_logits, dtype=np.float64)
    if losses.shape != logits.shape or losses.shape[1] != 3:
        raise ValueError("action_losses and route_logits must both be [N,3]")
    chosen = logits.argmax(1)
    oracle = losses.argmin(1)
    rows = np.arange(len(losses))
    stop = losses[:, int(RouteAction.STOP)]
    oracle_gain = np.maximum(stop - losses[rows, oracle], 0.0).mean()
    realised_gain = (stop - losses[rows, chosen]).mean()
    # Expected gain proxy from STOP-vs-best-communication route logit margin.
    predicted_gain = logits[:, 1:].max(1) - logits[:, 0]
    true_gain = stop - losses[:, 1:].min(1)
    correlation = spearmanr(predicted_gain, true_gain).statistic
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "action_accuracy": float((chosen == oracle).mean()),
        "oracle_gain": float(oracle_gain),
        "realised_gain": float(realised_gain),
        "oracle_gain_recovery": float(realised_gain / max(oracle_gain, 1e-12)),
        "gain_spearman": float(correlation),
        "query_rate": float((chosen != int(RouteAction.STOP)).mean()),
        "w_first_rate": float((chosen == int(RouteAction.W_FIRST)).mean()),
        "p_first_rate": float((chosen == int(RouteAction.P_FIRST)).mean()),
    }


def cross_fitted_public_selector(
    public_features: np.ndarray,
    action_losses: np.ndarray,
    *,
    num_folds: int = 5,
    seed: int = 2026,
) -> dict[str, np.ndarray | float]:
    """Leakage-safe diagnostic of whether public summaries expose headroom."""
    features = np.asarray(public_features, dtype=np.float64)
    losses = np.asarray(action_losses, dtype=np.float64)
    if features.ndim != 2 or losses.ndim != 2 or len(features) != len(losses):
        raise ValueError("features and action_losses must be aligned matrices")
    target = losses.argmin(1)
    counts = np.bincount(target, minlength=losses.shape[1])
    positive = counts[counts > 0]
    effective_folds = min(num_folds, int(positive.min()) if len(positive) else 0)
    if effective_folds < 2:
        prediction = np.full(len(target), int(np.argmin(losses.mean(0))), dtype=np.int64)
    else:
        prediction = np.empty(len(target), dtype=np.int64)
        splitter = StratifiedKFold(
            effective_folds, shuffle=True, random_state=seed)
        for train_index, test_index in splitter.split(features, target):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000, class_weight="balanced",
                    random_state=seed, multi_class="auto"),
            )
            model.fit(features[train_index], target[train_index])
            prediction[test_index] = model.predict(features[test_index])
    rows = np.arange(len(losses))
    stop_loss = losses[:, int(RouteAction.STOP)]
    selected_loss = losses[rows, prediction]
    oracle_loss = losses.min(1)
    denominator = max(float((stop_loss - oracle_loss).mean()), 1e-12)
    return {
        "prediction": prediction,
        "target": target,
        "mean_loss": float(selected_loss.mean()),
        "gain_over_stop": float((stop_loss - selected_loss).mean()),
        "oracle_gain_recovery": float(
            (stop_loss - selected_loss).mean() / denominator),
        "accuracy": float((prediction == target).mean()),
    }
