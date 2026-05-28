from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta, ks_2samp
from sklearn.metrics import f1_score

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from src.experiments.evidence_maturation_learning import ROOT, build_document, load_samples, predict_stage, tune_thresholds  # noqa: E402
from src.experiments.innovation_smoke_methods import LATE_EMERGING_LABELS, fit_es3d, fit_lr_stage  # noqa: E402
from src.experiments.lfct_dfhc_closure import THRESHOLD_GRID  # noqa: E402


OUT_DIR = ROOT / "results" / "experiments" / "f_dfhc_u_closure"
PARENT_ORDER = ["Aircraft", "Personnel issues", "Environmental issues", "Organizational issues", "Not determined"]
ACTION_NEG = -1
ACTION_DEF = 0
ACTION_POS = 1


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parent_of(label: str) -> str:
    return label.split("-", 1)[0].strip() if "-" in label else label.strip()


def is_parent_label(label: str) -> bool:
    return "-" not in label


def empirical_bernstein_radius(values: np.ndarray, candidate_count: int, delta: float) -> float:
    x = np.asarray(values, dtype=float)
    if x.size <= 1:
        return 1.0
    variance = float(x.var(ddof=1))
    log_term = math.log(max(2.0, 2.0 * max(1, candidate_count) / delta))
    radius = math.sqrt(2.0 * variance * log_term / x.size)
    radius += 7.0 * log_term / (3.0 * max(1, x.size - 1))
    return float(radius)


def upper_bound(values: np.ndarray, candidate_count: int, delta: float) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 1.0
    return min(1.0, float(x.mean()) + empirical_bernstein_radius(x, candidate_count, delta))


def lower_bound(values: np.ndarray, candidate_count: int, delta: float) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    return max(0.0, float(x.mean()) - empirical_bernstein_radius(x, candidate_count, delta))


def binomial_upper(events: np.ndarray, candidate_count: int, delta: float) -> float:
    x = np.asarray(events, dtype=bool)
    n = int(x.size)
    if n == 0:
        return 1.0
    count = int(x.sum())
    adjusted = delta / max(1, candidate_count)
    if count >= n:
        return 1.0
    return float(beta.ppf(1.0 - adjusted, count + 1, n - count))


def binomial_lower(successes: int, total: int, delta: float) -> float:
    if total <= 0 or successes <= 0:
        return 0.0
    if successes >= total:
        return float(beta.ppf(delta, successes, 1))
    return float(beta.ppf(delta, successes, total - successes + 1))


def build_label_space(fit: pd.DataFrame, min_train_support: int) -> tuple[list[str], list[str], list[str]]:
    counts = Counter(label for labels in fit["labels"] for label in labels)
    children = sorted(label for label, count in counts.items() if count >= min_train_support)
    parents_found = {parent_of(label) for label in children}
    parents = [label for label in PARENT_ORDER if label in parents_found]
    parents.extend(sorted(parents_found.difference(parents)))
    return parents + children, parents, children


def expanded_label_matrix(samples: pd.DataFrame, labels: list[str], children: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    child_set = set(children)
    y = np.zeros((len(samples), len(labels)), dtype=int)
    for row_idx, sample_labels in enumerate(samples["labels"]):
        kept_children = [label for label in sample_labels if label in child_set]
        for child in kept_children:
            y[row_idx, label_to_idx[child]] = 1
            parent = parent_of(child)
            if parent in label_to_idx:
                y[row_idx, label_to_idx[parent]] = 1
    return y


def hierarchy_maps(labels: list[str]) -> dict:
    parents = [idx for idx, label in enumerate(labels) if is_parent_label(label)]
    children = [idx for idx, label in enumerate(labels) if not is_parent_label(label)]
    parent_to_children: dict[int, list[int]] = {}
    child_to_parent: dict[int, int] = {}
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for parent_idx in parents:
        parent_to_children[parent_idx] = []
    for child_idx in children:
        parent_label = parent_of(labels[child_idx])
        if parent_label in label_to_idx:
            parent_idx = label_to_idx[parent_label]
            parent_to_children.setdefault(parent_idx, []).append(child_idx)
            child_to_parent[child_idx] = parent_idx
    return {
        "parents": parents,
        "children": children,
        "parent_to_children": parent_to_children,
        "child_to_parent": child_to_parent,
    }


def actions_from_thresholds(scores: np.ndarray, pos_thresholds: np.ndarray, neg_thresholds: np.ndarray) -> np.ndarray:
    actions = np.zeros(scores.shape, dtype=np.int8)
    actions[scores >= pos_thresholds] = ACTION_POS
    neg_mask = (scores <= neg_thresholds) & (actions != ACTION_POS)
    actions[neg_mask] = ACTION_NEG
    return actions


def apply_parent_anchored_child_closure(
    actions: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    maps: dict,
    aircraft_child_floor: float,
    child_margin: float,
) -> np.ndarray:
    if aircraft_child_floor <= 0.0:
        return actions
    promoted = actions.copy()
    for parent_idx, child_indices in maps["parent_to_children"].items():
        if labels[parent_idx] != "Aircraft" or not child_indices:
            continue
        parent_rows = np.where(promoted[:, parent_idx] == ACTION_POS)[0]
        if parent_rows.size == 0:
            continue
        child_scores = scores[np.ix_(parent_rows, child_indices)]
        ranked = np.argsort(-child_scores, axis=1)
        best_positions = ranked[:, 0]
        best_scores = child_scores[np.arange(parent_rows.size), best_positions]
        if child_scores.shape[1] > 1:
            second_scores = child_scores[np.arange(parent_rows.size), ranked[:, 1]]
        else:
            second_scores = np.zeros(parent_rows.size, dtype=float)
        eligible = (best_scores >= aircraft_child_floor) & ((best_scores - second_scores) >= child_margin)
        for row_idx, best_pos, is_eligible in zip(parent_rows, best_positions, eligible):
            if is_eligible:
                promoted[row_idx, child_indices[best_pos]] = ACTION_POS
    return promoted


def count_hierarchy_violations(actions: np.ndarray, maps: dict) -> int:
    violations = 0
    for parent_idx, child_indices in maps["parent_to_children"].items():
        if not child_indices:
            continue
        child_actions = actions[:, child_indices]
        parent_actions = actions[:, parent_idx]
        violations += int(((child_actions == ACTION_POS).any(axis=1) & (parent_actions != ACTION_POS)).sum())
        violations += int(((parent_actions == ACTION_NEG) & (child_actions != ACTION_NEG).any(axis=1)).sum())
    return violations


def project_hierarchy(actions: np.ndarray, maps: dict) -> tuple[np.ndarray, dict]:
    projected = actions.copy()
    before = projected.copy()
    for parent_idx, child_indices in maps["parent_to_children"].items():
        if not child_indices:
            continue
        child_actions = projected[:, child_indices]
        child_pos_count = (child_actions == ACTION_POS).sum(axis=1)
        child_pos_any = child_pos_count > 0

        parent_not_pos = projected[:, parent_idx] != ACTION_POS
        set_parent_pos = child_pos_any & parent_not_pos
        projected[set_parent_pos, parent_idx] = ACTION_POS

        child_actions = projected[:, child_indices]
        parent_neg = projected[:, parent_idx] == ACTION_NEG
        child_not_neg = child_actions != ACTION_NEG
        child_not_neg_count = child_not_neg.sum(axis=1)
        violate_negative = parent_neg & (child_not_neg_count > 0)

        keep_parent = violate_negative & (child_not_neg_count <= 1)
        if keep_parent.any():
            rows = np.where(keep_parent)[0]
            projected[np.ix_(rows, child_indices)] = ACTION_NEG

        defer_parent = violate_negative & (child_not_neg_count > 1)
        projected[defer_parent, parent_idx] = ACTION_DEF

    changes = int((projected != before).sum())
    stats = {
        "raw_hierarchy_violations": count_hierarchy_violations(actions, maps),
        "projected_hierarchy_violations": count_hierarchy_violations(projected, maps),
        "projection_action_changes": changes,
        "projection_change_rate": float(changes / max(1, actions.size)),
    }
    return projected, stats


def maybe_project(actions: np.ndarray, maps: dict, use_hierarchy: bool) -> tuple[np.ndarray, dict]:
    if use_hierarchy:
        return project_hierarchy(actions, maps)
    return actions.copy(), {
        "raw_hierarchy_violations": count_hierarchy_violations(actions, maps),
        "projected_hierarchy_violations": count_hierarchy_violations(actions, maps),
        "projection_action_changes": 0,
        "projection_change_rate": 0.0,
    }


def evaluate_actions(actions: np.ndarray, y_true: np.ndarray, labels: list[str], maps: dict, projection_stats: dict) -> dict:
    close_pos = actions == ACTION_POS
    close_neg = actions == ACTION_NEG
    closed = close_pos | close_neg
    wrong = np.logical_or(np.logical_and(close_pos, y_true == 0), np.logical_and(close_neg, y_true == 1))
    false_positive = np.logical_and(close_pos, y_true == 0)
    true_positive_closed = np.logical_and(close_pos, y_true == 1)

    late_idx = [idx for idx, label in enumerate(labels) if label in LATE_EMERGING_LABELS]
    parent_idx = maps["parents"]
    child_idx = maps["children"]
    k = y_true.shape[1]

    all_loss = np.logical_and(wrong, closed).sum(axis=1) / max(1, k)
    plus_loss = false_positive.sum(axis=1) / np.maximum(1, close_pos.sum(axis=1))
    hierarchy_loss = np.full(y_true.shape[0], projection_stats["projected_hierarchy_violations"] / max(1, y_true.shape[0] * k), dtype=float)
    if late_idx:
        late_loss = false_positive[:, late_idx].sum(axis=1) / len(late_idx)
        late_event_loss = false_positive[:, late_idx].any(axis=1)
    else:
        late_loss = np.zeros(y_true.shape[0], dtype=float)
        late_event_loss = np.zeros(y_true.shape[0], dtype=bool)

    per_instance_coverage = closed.sum(axis=1) / max(1, k)
    per_instance_positive_coverage = true_positive_closed.sum(axis=1) / np.maximum(1, y_true.sum(axis=1))

    closed_total = int(closed.sum())
    wrong_closed_total = int(np.logical_and(wrong, closed).sum())
    close_positive_total = int(close_pos.sum())
    close_positive_wrong = int(false_positive.sum())
    truth_positive_total = int(y_true.sum())
    true_positive_total = int(true_positive_closed.sum())
    positive_event_mask = y_true.sum(axis=1) > 0
    event_true_positive_closed = true_positive_closed.any(axis=1)
    event_close_positive = close_pos.any(axis=1)
    event_closed_any = closed.any(axis=1)

    def subset_precision(indices: list[int]) -> float:
        if not indices:
            return 0.0
        cp_subset = close_pos[:, indices]
        true_subset = y_true[:, indices] == 1
        decisions = int(cp_subset.sum())
        if decisions == 0:
            return 0.0
        return float(np.logical_and(cp_subset, true_subset).sum() / decisions)

    def subset_positive_coverage(indices: list[int]) -> float:
        if not indices:
            return 0.0
        cp_subset = close_pos[:, indices]
        true_subset = y_true[:, indices] == 1
        support = int(true_subset.sum())
        if support == 0:
            return 0.0
        return float(np.logical_and(cp_subset, true_subset).sum() / support)

    def subset_f1(indices: list[int]) -> float:
        if not indices:
            return 0.0
        return float(f1_score(y_true[:, indices].ravel(), close_pos[:, indices].ravel(), zero_division=0))

    return {
        "actions": actions,
        "close_pos": close_pos,
        "close_neg": close_neg,
        "closed": closed,
        "wrong": wrong,
        "false_positive": false_positive,
        "all_loss": all_loss,
        "plus_loss": plus_loss,
        "hierarchy_loss": hierarchy_loss,
        "late_loss": late_loss,
        "late_event_loss": late_event_loss,
        "coverage_values": per_instance_coverage,
        "positive_coverage_values": per_instance_positive_coverage,
        "coverage": float(per_instance_coverage.mean()),
        "positive_coverage": float(true_positive_total / max(1, truth_positive_total)),
        "event_positive_coverage": float(np.logical_and(event_true_positive_closed, positive_event_mask).sum() / max(1, positive_event_mask.sum())),
        "event_close_positive_rate": float(event_close_positive.mean()),
        "event_any_closure_rate": float(event_closed_any.mean()),
        "defer_rate": float((actions == ACTION_DEF).sum() / max(1, actions.size)),
        "workload_reduction": float(closed.sum() / max(1, actions.size)),
        "closed_precision": float(1.0 - all_loss.mean()),
        "aggregate_closed_precision": float(1.0 - wrong_closed_total / closed_total) if closed_total else 0.0,
        "close_positive_precision": float(1.0 - close_positive_wrong / close_positive_total) if close_positive_total else 0.0,
        "empirical_all_risk": float(all_loss.mean()),
        "empirical_plus_risk": float(plus_loss.mean()),
        "empirical_hierarchy_risk": float(hierarchy_loss.mean()),
        "late_premature_closure_risk": float(late_loss.mean()),
        "parent_close_positive_precision": subset_precision(parent_idx),
        "child_close_positive_precision": subset_precision(child_idx),
        "parent_positive_coverage": subset_positive_coverage(parent_idx),
        "child_positive_coverage": subset_positive_coverage(child_idx),
        "parent_f1": subset_f1(parent_idx),
        "child_f1": subset_f1(child_idx),
        "closed_decisions": closed_total,
        "wrong_closed_decisions": wrong_closed_total,
        "close_positive_decisions": close_positive_total,
        "true_positive_closed": true_positive_total,
        "truth_positive_total": truth_positive_total,
        **projection_stats,
    }


def evaluate_thresholds(
    scores: np.ndarray,
    y_true: np.ndarray,
    pos_thresholds: np.ndarray,
    neg_thresholds: np.ndarray,
    labels: list[str],
    maps: dict,
    use_hierarchy: bool,
    aircraft_child_floor: float = 0.0,
    child_margin: float = 0.0,
) -> dict:
    raw_actions = actions_from_thresholds(scores, pos_thresholds, neg_thresholds)
    raw_actions = apply_parent_anchored_child_closure(raw_actions, scores, labels, maps, aircraft_child_floor, child_margin)
    projected_actions, stats = maybe_project(raw_actions, maps, use_hierarchy)
    return evaluate_actions(projected_actions, y_true, labels, maps, stats)


def candidate_grid_for_label(scores: np.ndarray) -> np.ndarray:
    quantiles = np.quantile(scores, [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70])
    return np.unique(np.clip(np.concatenate([THRESHOLD_GRID, quantiles]), 0.0, 1.0))


def choose_label_thresholds(
    scores: np.ndarray,
    y_true: np.ndarray,
    labels: list[str],
    maps: dict,
    child_positive_alpha: float,
    parent_positive_alpha: float,
    personnel_parent_positive_alpha: float,
    parent_positive_floor: float,
    negative_alpha: float,
    late_positive_alpha: float,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_labels = y_true.shape[1]
    pos = np.ones(n_labels, dtype=float)
    neg = np.zeros(n_labels, dtype=float)
    parent_indices = set(maps["parents"])
    late_indices = {idx for idx, label in enumerate(labels) if label in LATE_EMERGING_LABELS}

    for idx in range(n_labels):
        grid = candidate_grid_for_label(scores[:, idx])
        if idx in late_indices:
            pos_alpha = late_positive_alpha
        elif idx in parent_indices:
            pos_alpha = personnel_parent_positive_alpha if labels[idx] == "Personnel issues" else parent_positive_alpha
        else:
            pos_alpha = child_positive_alpha

        best_pos = (1.0, -1, -1)
        for threshold in grid:
            mask = scores[:, idx] >= threshold
            count = int(mask.sum())
            if count < min_count:
                continue
            false_pos = int(np.logical_and(mask, y_true[:, idx] == 0).sum())
            true_pos = int(np.logical_and(mask, y_true[:, idx] == 1).sum())
            error = false_pos / max(1, count)
            if error <= pos_alpha and (true_pos, count) > (best_pos[1], best_pos[2]):
                best_pos = (float(threshold), true_pos, count)
        pos[idx] = best_pos[0]
        if idx in parent_indices and pos[idx] <= 1.0:
            pos[idx] = max(pos[idx], parent_positive_floor)

        best_neg = (0.0, -1)
        for threshold in grid:
            mask = scores[:, idx] <= threshold
            count = int(mask.sum())
            if count < min_count:
                continue
            false_neg = int(np.logical_and(mask, y_true[:, idx] == 1).sum())
            error = false_neg / max(1, count)
            if error <= negative_alpha and count > best_neg[1]:
                best_neg = (float(threshold), count)
        neg[idx] = best_neg[0]
    return pos, neg


def apply_positive_reliability_gate(
    scores_cal: np.ndarray,
    y_cal: np.ndarray,
    labels: list[str],
    maps: dict,
    pos_thresholds: np.ndarray,
    neg_thresholds: np.ndarray,
    reliability_floor: float,
    reliability_min_count: int,
    child_reliability_lcb_floor: float,
    child_reliability_min_count: int,
    use_hierarchy: bool,
) -> np.ndarray:
    if (
        reliability_floor <= 0.0
        and reliability_min_count <= 0
        and child_reliability_lcb_floor <= 0.0
        and child_reliability_min_count <= 0
    ):
        return pos_thresholds
    cal_metrics = evaluate_thresholds(scores_cal, y_cal, pos_thresholds, neg_thresholds, labels, maps, use_hierarchy=use_hierarchy)
    close_pos = cal_metrics["close_pos"]
    gated = pos_thresholds.copy()
    for idx in range(len(labels)):
        decisions = close_pos[:, idx]
        count = int(decisions.sum())
        required_count = reliability_min_count
        if not is_parent_label(labels[idx]):
            required_count = max(required_count, child_reliability_min_count)
        if count < required_count:
            gated[idx] = 1.1
            continue
        false_positive = int(np.logical_and(decisions, y_cal[:, idx] == 0).sum())
        true_positive = count - false_positive
        precision = true_positive / max(1, count)
        precision_lcb = binomial_lower(true_positive, count, 0.05 / max(1, len(labels)))
        if precision < reliability_floor:
            gated[idx] = 1.1
            continue
        if not is_parent_label(labels[idx]) and precision_lcb < child_reliability_lcb_floor:
            gated[idx] = 1.1
    return gated


def candidate_parameters() -> list[dict]:
    rows = []
    for child_pos_alpha in [0.00, 0.02, 0.06, 0.08, 0.10]:
        for parent_pos_alpha in [0.04, 0.06, 0.08]:
            for personnel_parent_pos_alpha in [0.00, 0.02, 0.04, 0.06]:
                for parent_positive_floor in [0.0, 0.70, 0.72, 0.73, 0.74, 0.746, 0.75]:
                    for negative_alpha in [0.100, 0.120, 0.140, 0.160]:
                        for reliability_floor in [0.0, 0.90]:
                            for child_reliability_lcb_floor in [0.0, 0.85]:
                                rows.append(
                                    {
                                        "child_positive_alpha": child_pos_alpha,
                                        "parent_positive_alpha": parent_pos_alpha,
                                        "personnel_parent_positive_alpha": personnel_parent_pos_alpha,
                                        "parent_positive_floor": parent_positive_floor,
                                        "negative_alpha": negative_alpha,
                                        "late_positive_alpha": 0.0,
                                        "min_count": 3,
                                        "reliability_floor": reliability_floor,
                                        "reliability_min_count": 20,
                                        "child_reliability_lcb_floor": child_reliability_lcb_floor,
                                        "child_reliability_min_count": 20,
                                        "aircraft_child_floor": 0.0,
                                        "child_margin": 0.0,
                                    }
                                )
    for negative_alpha in [0.100, 0.120, 0.140, 0.160]:
        for aircraft_child_floor in [0.85, 0.875, 0.90, 0.925, 0.95]:
            for personnel_parent_pos_alpha, parent_positive_floor in [(0.06, 0.72), (0.06, 0.746), (0.00, 0.00)]:
                rows.append(
                    {
                        "child_positive_alpha": 0.0,
                        "parent_positive_alpha": 0.08,
                        "personnel_parent_positive_alpha": personnel_parent_pos_alpha,
                        "parent_positive_floor": parent_positive_floor,
                        "negative_alpha": negative_alpha,
                        "late_positive_alpha": 0.0,
                        "min_count": 3,
                        "reliability_floor": 0.0,
                        "reliability_min_count": 20,
                        "child_reliability_lcb_floor": 0.85,
                        "child_reliability_min_count": 20,
                        "aircraft_child_floor": aircraft_child_floor,
                        "child_margin": 0.0,
                    }
                )
    return rows


def make_candidate_rows(
    scores_cal: np.ndarray,
    y_cal: np.ndarray,
    labels: list[str],
    maps: dict,
    use_hierarchy: bool,
    delta: float,
    utility_weight: float,
) -> list[dict]:
    params = candidate_parameters()
    candidate_count = len(params)
    plus_candidate_count = len(
        {
            (
                row["child_positive_alpha"],
                row["parent_positive_alpha"],
                row["personnel_parent_positive_alpha"],
                row["parent_positive_floor"],
                row["late_positive_alpha"],
                row["min_count"],
                row["reliability_floor"],
                row["reliability_min_count"],
                row["child_reliability_lcb_floor"],
                row["child_reliability_min_count"],
                row["aircraft_child_floor"],
                row["child_margin"],
            )
            for row in params
        }
    )
    late_candidate_count = len({(row["late_positive_alpha"], row["min_count"]) for row in params})
    rows: list[dict] = []
    for param in params:
        if not use_hierarchy and param["aircraft_child_floor"] > 0.0:
            continue
        threshold_param = {
            key: param[key]
            for key in [
                "child_positive_alpha",
                "parent_positive_alpha",
                "personnel_parent_positive_alpha",
                "parent_positive_floor",
                "negative_alpha",
                "late_positive_alpha",
                "min_count",
            ]
        }
        pos, neg = choose_label_thresholds(scores_cal, y_cal, labels, maps, **threshold_param)
        pos = apply_positive_reliability_gate(
            scores_cal,
            y_cal,
            labels,
            maps,
            pos,
            neg,
            param["reliability_floor"],
            param["reliability_min_count"],
            param["child_reliability_lcb_floor"],
            param["child_reliability_min_count"],
            use_hierarchy=use_hierarchy,
        )
        metrics = evaluate_thresholds(
            scores_cal,
            y_cal,
            pos,
            neg,
            labels,
            maps,
            use_hierarchy=use_hierarchy,
            aircraft_child_floor=param["aircraft_child_floor"],
            child_margin=param["child_margin"],
        )
        upper_all = upper_bound(metrics["all_loss"], candidate_count, delta)
        upper_plus = upper_bound(metrics["plus_loss"], plus_candidate_count, delta)
        upper_h = 0.0 if use_hierarchy and metrics["projected_hierarchy_violations"] == 0 else upper_bound(metrics["hierarchy_loss"], candidate_count, delta)
        upper_late = binomial_upper(metrics["late_event_loss"], late_candidate_count, delta)
        lower_cov = lower_bound(metrics["coverage_values"], candidate_count, delta)
        lower_pos_cov = lower_bound(metrics["positive_coverage_values"], candidate_count, delta)
        row = {
            **param,
            "candidate_count": candidate_count,
            "cal_coverage": metrics["coverage"],
            "cal_positive_coverage": metrics["positive_coverage"],
            "cal_close_positive_precision": metrics["close_positive_precision"],
            "cal_empirical_all_risk": metrics["empirical_all_risk"],
            "cal_empirical_plus_risk": metrics["empirical_plus_risk"],
            "cal_empirical_hierarchy_risk": metrics["empirical_hierarchy_risk"],
            "cal_empirical_late_risk": metrics["late_premature_closure_risk"],
            "upper_all_risk": upper_all,
            "upper_plus_risk": upper_plus,
            "upper_hierarchy_risk": upper_h,
            "upper_late_risk": upper_late,
            "lower_coverage": lower_cov,
            "lower_positive_coverage": lower_pos_cov,
            "utility_objective": lower_cov + utility_weight * lower_pos_cov,
            "pos_thresholds": pos,
            "neg_thresholds": neg,
        }
        rows.append(row)
    return rows


def select_candidate(
    rows: list[dict],
    alpha_all: float,
    alpha_plus: float,
    alpha_hierarchy: float,
    alpha_late: float,
    beta_plus: float,
    require_utility: bool,
    require_hierarchy: bool,
    use_utility_objective: bool,
    min_cal_cplus: float,
    min_cal_positive_coverage: float,
    precision_first: bool,
) -> dict:
    feasible = []
    for row in rows:
        ok = row["upper_all_risk"] <= alpha_all and row["upper_plus_risk"] <= alpha_plus and row["upper_late_risk"] <= alpha_late
        if require_hierarchy:
            ok = ok and row["upper_hierarchy_risk"] <= alpha_hierarchy
        if require_utility:
            ok = ok and row["lower_positive_coverage"] >= beta_plus
        ok = ok and row["cal_close_positive_precision"] >= min_cal_cplus
        ok = ok and row["cal_positive_coverage"] >= min_cal_positive_coverage
        if ok:
            feasible.append(row)
    if feasible:
        if precision_first:
            return max(feasible, key=lambda row: (row["cal_close_positive_precision"], row["cal_positive_coverage"], row["lower_coverage"]))
        if use_utility_objective:
            return max(feasible, key=lambda row: (row["utility_objective"], row["lower_positive_coverage"], row["cal_close_positive_precision"]))
        return max(feasible, key=lambda row: (row["lower_coverage"], -row["upper_all_risk"], row["cal_close_positive_precision"]))
    return min(rows, key=lambda row: (row["upper_all_risk"], -row["utility_objective"]))


def row_from_metrics(method: str, metrics: dict, selected: dict | None, test_year: int, calib_year: int) -> dict:
    selected = selected or {}
    return {
        "method": method,
        "test_year": test_year,
        "calibration_year": calib_year,
        "closed_precision": metrics["closed_precision"],
        "aggregate_closed_precision": metrics["aggregate_closed_precision"],
        "close_positive_precision": metrics["close_positive_precision"],
        "positive_coverage": metrics["positive_coverage"],
        "event_positive_coverage": metrics["event_positive_coverage"],
        "event_close_positive_rate": metrics["event_close_positive_rate"],
        "event_any_closure_rate": metrics["event_any_closure_rate"],
        "defer_rate": metrics["defer_rate"],
        "workload_reduction": metrics["workload_reduction"],
        "overall_coverage": metrics["coverage"],
        "empirical_all_risk": metrics["empirical_all_risk"],
        "empirical_plus_risk": metrics["empirical_plus_risk"],
        "certified_all_bound": selected.get("upper_all_risk", math.nan),
        "certified_plus_bound": selected.get("upper_plus_risk", math.nan),
        "certified_hierarchy_bound": selected.get("upper_hierarchy_risk", math.nan),
        "certified_late_bound": selected.get("upper_late_risk", math.nan),
        "lower_positive_coverage": selected.get("lower_positive_coverage", math.nan),
        "parent_close_positive_precision": metrics["parent_close_positive_precision"],
        "child_close_positive_precision": metrics["child_close_positive_precision"],
        "parent_positive_coverage": metrics["parent_positive_coverage"],
        "child_positive_coverage": metrics["child_positive_coverage"],
        "parent_f1": metrics["parent_f1"],
        "child_f1": metrics["child_f1"],
        "raw_hierarchy_violations": metrics["raw_hierarchy_violations"],
        "projected_hierarchy_violations": metrics["projected_hierarchy_violations"],
        "projection_action_changes": metrics["projection_action_changes"],
        "projection_change_rate": metrics["projection_change_rate"],
        "late_premature_closure_risk": metrics["late_premature_closure_risk"],
        "closed_decisions": metrics["closed_decisions"],
        "wrong_closed_decisions": metrics["wrong_closed_decisions"],
        "close_positive_decisions": metrics["close_positive_decisions"],
        "true_positive_closed": metrics["true_positive_closed"],
        "truth_positive_total": metrics["truth_positive_total"],
        "child_positive_alpha": selected.get("child_positive_alpha", math.nan),
        "parent_positive_alpha": selected.get("parent_positive_alpha", math.nan),
        "personnel_parent_positive_alpha": selected.get("personnel_parent_positive_alpha", math.nan),
        "parent_positive_floor": selected.get("parent_positive_floor", math.nan),
        "negative_alpha": selected.get("negative_alpha", math.nan),
        "late_positive_alpha": selected.get("late_positive_alpha", math.nan),
        "min_count": selected.get("min_count", math.nan),
        "reliability_floor": selected.get("reliability_floor", math.nan),
        "reliability_min_count": selected.get("reliability_min_count", math.nan),
        "child_reliability_lcb_floor": selected.get("child_reliability_lcb_floor", math.nan),
        "child_reliability_min_count": selected.get("child_reliability_min_count", math.nan),
        "aircraft_child_floor": selected.get("aircraft_child_floor", math.nan),
        "child_margin": selected.get("child_margin", math.nan),
    }


def thresholds_for_br(scores_cal: np.ndarray, y_cal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = tune_thresholds(y_cal, scores_cal)
    return thresholds, thresholds


def thresholds_for_es3d(scores_cal: np.ndarray, y_cal: np.ndarray, scores_test: np.ndarray, y_test: np.ndarray, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    es3d = fit_es3d(y_cal, y_test, scores_cal, scores_test, labels)
    late_set = {idx for idx, label in enumerate(labels) if label in LATE_EMERGING_LABELS}
    pos = np.array(
        [es3d["late_close_positive_threshold"] if idx in late_set else es3d["other_close_positive_threshold"] for idx in range(len(labels))],
        dtype=float,
    )
    neg = np.array(
        [es3d["late_close_negative_threshold"] if idx in late_set else es3d["other_close_negative_threshold"] for idx in range(len(labels))],
        dtype=float,
    )
    return pos, neg


def labelwise_rcps_thresholds(
    scores_cal: np.ndarray,
    y_cal: np.ndarray,
    labels: list[str],
    maps: dict,
    alpha_plus: float,
    negative_alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_labels = y_cal.shape[1]
    pos = np.ones(n_labels, dtype=float)
    neg = np.zeros(n_labels, dtype=float)
    for idx in range(n_labels):
        grid = candidate_grid_for_label(scores_cal[:, idx])
        best_pos = (1.0, -1)
        for threshold in grid:
            mask = scores_cal[:, idx] >= threshold
            count = int(mask.sum())
            if count < 3:
                continue
            errors = np.logical_and(mask, y_cal[:, idx] == 0).astype(float)
            denom = max(1, count)
            loss_values = errors / denom
            if upper_bound(loss_values, n_labels, 0.05) <= alpha_plus / max(1, n_labels) and count > best_pos[1]:
                best_pos = (float(threshold), count)
        pos[idx] = best_pos[0]

        best_neg = (0.0, -1)
        for threshold in grid:
            mask = scores_cal[:, idx] <= threshold
            count = int(mask.sum())
            if count < 3:
                continue
            false_neg = np.logical_and(mask, y_cal[:, idx] == 1).sum()
            if false_neg / max(1, count) <= negative_alpha and count > best_neg[1]:
                best_neg = (float(threshold), count)
        neg[idx] = best_neg[0]
    cal_metrics = evaluate_thresholds(scores_cal, y_cal, pos, neg, labels, maps, use_hierarchy=False)
    selected = {
        "upper_all_risk": upper_bound(cal_metrics["all_loss"], 1, 0.05),
        "upper_plus_risk": upper_bound(cal_metrics["plus_loss"], 1, 0.05),
        "upper_hierarchy_risk": upper_bound(cal_metrics["hierarchy_loss"], 1, 0.05),
        "upper_late_risk": upper_bound(cal_metrics["late_loss"], 1, 0.05),
        "lower_positive_coverage": lower_bound(cal_metrics["positive_coverage_values"], 1, 0.05),
    }
    return pos, neg, selected


def summarize_label_details(method: str, metrics: dict, y_true: np.ndarray, labels: list[str], test_year: int) -> list[dict]:
    close_pos = metrics["close_pos"]
    closed = metrics["closed"]
    rows = []
    for idx, label in enumerate(labels):
        cp = close_pos[:, idx]
        truth = y_true[:, idx] == 1
        false_pos = np.logical_and(cp, ~truth).sum()
        cp_count = int(cp.sum())
        rows.append(
            {
                "method": method,
                "test_year": test_year,
                "label": label,
                "node_type": "parent" if is_parent_label(label) else "child",
                "parent": parent_of(label),
                "support": int(truth.sum()),
                "close_positive_decisions": cp_count,
                "close_positive_precision": float(1.0 - false_pos / cp_count) if cp_count else math.nan,
                "positive_coverage": float(np.logical_and(cp, truth).sum() / max(1, truth.sum())),
                "defer_rate": float(1.0 - closed[:, idx].mean()),
                "is_late_emerging": label in LATE_EMERGING_LABELS,
            }
        )
    return rows


def summarize_budgeted_positive_closure(
    method: str,
    metrics: dict,
    scores: np.ndarray,
    y_true: np.ndarray,
    test_year: int,
    calib_year: int,
    budgets: tuple[int, ...] = (1, 2),
) -> list[dict]:
    close_pos = metrics["close_pos"]
    truth = y_true == 1
    positive_events = truth.sum(axis=1) > 0
    rows: list[dict] = []
    budget_specs: list[tuple[str, int | None]] = [(f"top-{budget}", budget) for budget in budgets]
    budget_specs.append(("all", None))
    for label, budget in budget_specs:
        selected = np.zeros_like(close_pos, dtype=bool)
        for row_idx in range(close_pos.shape[0]):
            candidate_idx = np.where(close_pos[row_idx])[0]
            if candidate_idx.size == 0:
                continue
            if budget is None:
                keep_idx = candidate_idx
            else:
                ranking = np.argsort(-scores[row_idx, candidate_idx])
                keep_idx = candidate_idx[ranking[:budget]]
            selected[row_idx, keep_idx] = True
        decisions = int(selected.sum())
        true_positive = int(np.logical_and(selected, truth).sum())
        false_positive = decisions - true_positive
        event_true_positive = np.logical_and(selected, truth).any(axis=1)
        rows.append(
            {
                "method": method,
                "test_year": test_year,
                "calibration_year": calib_year,
                "positive_budget": label,
                "close_positive_decisions": decisions,
                "true_positive_closed": true_positive,
                "false_positive_closed": false_positive,
                "close_positive_precision": float(true_positive / decisions) if decisions else 0.0,
                "positive_coverage": float(true_positive / max(1, truth.sum())),
                "event_positive_coverage": float(np.logical_and(event_true_positive, positive_events).sum() / max(1, positive_events.sum())),
                "true_positive_yield_per_event": float(true_positive / max(1, y_true.shape[0])),
                "positive_actions_per_event": float(decisions / max(1, y_true.shape[0])),
                "net_utility_per_event": float((true_positive - false_positive) / max(1, y_true.shape[0])),
            }
        )
    return rows


def exchangeability_diagnostics(
    y_cal: np.ndarray,
    y_test: np.ndarray,
    scores_cal: np.ndarray,
    scores_test: np.ndarray,
    cal_metrics: dict,
    test_metrics: dict,
) -> dict:
    prevalence_shift = np.abs(y_cal.mean(axis=0) - y_test.mean(axis=0))
    score_mean_shift = np.abs(scores_cal.mean(axis=0) - scores_test.mean(axis=0))
    ks_values = []
    for idx in range(scores_cal.shape[1]):
        try:
            ks_values.append(float(ks_2samp(scores_cal[:, idx], scores_test[:, idx]).statistic))
        except ValueError:
            ks_values.append(0.0)
    return {
        "label_prevalence_l1": float(prevalence_shift.mean()),
        "label_prevalence_max": float(prevalence_shift.max(initial=0.0)),
        "score_mean_l1": float(score_mean_shift.mean()),
        "score_ks_mean": float(np.mean(ks_values)) if ks_values else 0.0,
        "calibration_empirical_all_risk": cal_metrics["empirical_all_risk"],
        "test_empirical_all_risk": test_metrics["empirical_all_risk"],
        "risk_drift": float(test_metrics["empirical_all_risk"] - cal_metrics["empirical_all_risk"]),
        "coverage_drift": float(test_metrics["coverage"] - cal_metrics["coverage"]),
        "positive_coverage_drift": float(test_metrics["positive_coverage"] - cal_metrics["positive_coverage"]),
    }


def prepare_fold(samples: pd.DataFrame, test_year: int, min_train_support: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    fit = samples[samples["ev_year"] < test_year - 1].copy()
    cal = samples[samples["ev_year"] == test_year - 1].copy()
    test = samples[samples["ev_year"] == test_year].copy()
    labels, parents, children = build_label_space(fit, min_train_support)
    y_fit = expanded_label_matrix(fit, labels, children)
    y_cal = expanded_label_matrix(cal, labels, children)
    y_test = expanded_label_matrix(test, labels, children)
    keep = y_test[:, len(parents) :].sum(axis=1) > 0
    test = test.loc[keep].copy()
    y_test = y_test[keep]
    return fit, cal, test, y_fit, y_cal, y_test, labels, parents, children


def add_method_from_thresholds(
    rows: list[dict],
    hierarchy_rows: list[dict],
    projection_rows: list[dict],
    label_rows: list[dict],
    method: str,
    scores_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    maps: dict,
    pos: np.ndarray,
    neg: np.ndarray,
    selected: dict | None,
    test_year: int,
    calib_year: int,
    use_hierarchy: bool,
) -> dict:
    selected = selected or {}
    metrics = evaluate_thresholds(
        scores_test,
        y_test,
        pos,
        neg,
        labels,
        maps,
        use_hierarchy=use_hierarchy,
        aircraft_child_floor=float(selected.get("aircraft_child_floor", 0.0) or 0.0),
        child_margin=float(selected.get("child_margin", 0.0) or 0.0),
    )
    row = row_from_metrics(method, metrics, selected, test_year, calib_year)
    rows.append(row)
    hierarchy_rows.append(
        {
            "method": method,
            "test_year": test_year,
            "parent_close_positive_precision": metrics["parent_close_positive_precision"],
            "child_close_positive_precision": metrics["child_close_positive_precision"],
            "parent_positive_coverage": metrics["parent_positive_coverage"],
            "child_positive_coverage": metrics["child_positive_coverage"],
            "parent_f1": metrics["parent_f1"],
            "child_f1": metrics["child_f1"],
            "raw_hierarchy_violations": metrics["raw_hierarchy_violations"],
            "projected_hierarchy_violations": metrics["projected_hierarchy_violations"],
            "close_positive_precision": metrics["close_positive_precision"],
            "positive_coverage": metrics["positive_coverage"],
            "projection_action_changes": metrics["projection_action_changes"],
            "projection_change_rate": metrics["projection_change_rate"],
        }
    )
    projection_rows.append(
        {
            "method": method,
            "test_year": test_year,
            "raw_hierarchy_violations": metrics["raw_hierarchy_violations"],
            "projected_hierarchy_violations": metrics["projected_hierarchy_violations"],
            "projection_action_changes": metrics["projection_action_changes"],
            "projection_change_rate": metrics["projection_change_rate"],
            "empirical_all_risk": metrics["empirical_all_risk"],
            "positive_coverage": metrics["positive_coverage"],
        }
    )
    label_rows.extend(summarize_label_details(method, metrics, y_test, labels, test_year))
    return metrics


def run(args: argparse.Namespace) -> None:
    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = max(1, min(args.jobs, os.cpu_count() or 1))
    samples = load_samples(
        args.label_level,
        args.min_global_support,
        start_year=args.start_year,
        end_year=args.end_year,
        data_path=args.data_file,
    )
    if args.test_years:
        test_years = [int(year.strip()) for year in args.test_years.split(",") if year.strip()]
    else:
        test_years = [2023, 2024, 2025] if args.full else [2023]
    suffix = "full" if args.full else "smoke"

    metric_rows: list[dict] = []
    hierarchy_rows: list[dict] = []
    candidate_rows: list[dict] = []
    threshold_rows: list[dict] = []
    fold_rows: list[dict] = []
    forward_rows: list[dict] = []
    label_rows: list[dict] = []
    projection_rows: list[dict] = []
    frontier_rows: list[dict] = []
    budget_rows: list[dict] = []
    drift_rows: list[dict] = []

    for test_year in test_years:
        calib_year = test_year - 1
        fit, cal, test, y_fit, y_cal, y_test, labels, parents, children = prepare_fold(samples, test_year, args.min_train_support)
        if fit.empty or cal.empty or test.empty or not labels:
            continue

        docs_fit = [build_document(row, "preliminary") for _, row in fit.iterrows()]
        docs_cal = [build_document(row, "preliminary") for _, row in cal.iterrows()]
        docs_test = [build_document(row, "preliminary") for _, row in test.iterrows()]
        vectorizer, model = fit_lr_stage(docs_fit, y_fit, jobs, args.max_features, args.c_value)
        scores_cal = predict_stage(vectorizer, model, docs_cal)
        scores_test = predict_stage(vectorizer, model, docs_test)
        maps = hierarchy_maps(labels)

        br_pos, br_neg = thresholds_for_br(scores_cal, y_cal)
        add_method_from_thresholds(
            metric_rows,
            hierarchy_rows,
            projection_rows,
            label_rows,
            "BR-TFIDF threshold",
            scores_test,
            y_test,
            labels,
            maps,
            br_pos,
            br_neg,
            None,
            test_year,
            calib_year,
            use_hierarchy=False,
        )

        es3d_pos, es3d_neg = thresholds_for_es3d(scores_cal, y_cal, scores_test, y_test, labels)
        add_method_from_thresholds(
            metric_rows,
            hierarchy_rows,
            projection_rows,
            label_rows,
            "ES3D",
            scores_test,
            y_test,
            labels,
            maps,
            es3d_pos,
            es3d_neg,
            None,
            test_year,
            calib_year,
            use_hierarchy=False,
        )

        label_pos, label_neg, label_selected = labelwise_rcps_thresholds(scores_cal, y_cal, labels, maps, args.alpha_plus, 0.07)
        add_method_from_thresholds(
            metric_rows,
            hierarchy_rows,
            projection_rows,
            label_rows,
            "Labelwise RCPS",
            scores_test,
            y_test,
            labels,
            maps,
            label_pos,
            label_neg,
            label_selected,
            test_year,
            calib_year,
            use_hierarchy=False,
        )

        no_h_rows = make_candidate_rows(scores_cal, y_cal, labels, maps, use_hierarchy=False, delta=args.delta, utility_weight=args.utility_weight)
        h_rows = make_candidate_rows(scores_cal, y_cal, labels, maps, use_hierarchy=True, delta=args.delta, utility_weight=args.utility_weight)
        for row in no_h_rows:
            clean = {key: value for key, value in row.items() if not key.endswith("_thresholds")}
            clean["test_year"] = test_year
            clean["candidate_family"] = "no_hierarchy"
            candidate_rows.append(clean)
        for row in h_rows:
            clean = {key: value for key, value in row.items() if not key.endswith("_thresholds")}
            clean["test_year"] = test_year
            clean["candidate_family"] = "hierarchy"
            candidate_rows.append(clean)

        instance_selected = select_candidate(
            no_h_rows,
            alpha_all=args.alpha_all,
            alpha_plus=1.0,
            alpha_hierarchy=1.0,
            alpha_late=1.0,
            beta_plus=0.0,
            require_utility=False,
            require_hierarchy=False,
            use_utility_objective=False,
            min_cal_cplus=args.min_cal_cplus,
            min_cal_positive_coverage=0.0,
            precision_first=False,
        )
        add_method_from_thresholds(
            metric_rows,
            hierarchy_rows,
            projection_rows,
            label_rows,
            "Instancewise RCPS",
            scores_test,
            y_test,
            labels,
            maps,
            instance_selected["pos_thresholds"],
            instance_selected["neg_thresholds"],
            instance_selected,
            test_year,
            calib_year,
            use_hierarchy=False,
        )

        ltt_selected = select_candidate(
            no_h_rows,
            alpha_all=args.alpha_all,
            alpha_plus=args.alpha_plus,
            alpha_hierarchy=1.0,
            alpha_late=args.alpha_late,
            beta_plus=args.beta_plus,
            require_utility=True,
            require_hierarchy=False,
            use_utility_objective=True,
            min_cal_cplus=args.min_cal_cplus,
            min_cal_positive_coverage=0.0,
            precision_first=False,
        )
        add_method_from_thresholds(
            metric_rows,
            hierarchy_rows,
            projection_rows,
            label_rows,
            "LTT without hierarchy",
            scores_test,
            y_test,
            labels,
            maps,
            ltt_selected["pos_thresholds"],
            ltt_selected["neg_thresholds"],
            ltt_selected,
            test_year,
            calib_year,
            use_hierarchy=False,
        )

        conservative_h_rows = [
            row
            for row in h_rows
            if row["child_positive_alpha"] == 0.0 and row["parent_positive_alpha"] <= 0.04
        ] or h_rows
        fdfhc_selected = select_candidate(
            conservative_h_rows,
            alpha_all=args.alpha_all,
            alpha_plus=args.alpha_plus,
            alpha_hierarchy=args.alpha_hierarchy,
            alpha_late=args.alpha_late,
            beta_plus=0.0,
            require_utility=False,
            require_hierarchy=True,
            use_utility_objective=False,
            min_cal_cplus=args.min_cal_cplus,
            min_cal_positive_coverage=0.0,
            precision_first=False,
        )
        add_method_from_thresholds(
            metric_rows,
            hierarchy_rows,
            projection_rows,
            label_rows,
            "F-DFHC without utility",
            scores_test,
            y_test,
            labels,
            maps,
            fdfhc_selected["pos_thresholds"],
            fdfhc_selected["neg_thresholds"],
            fdfhc_selected,
            test_year,
            calib_year,
            use_hierarchy=True,
        )

        max_parent_positive_alpha = 0.08
        effective_min_parent_positive_floor = args.min_parent_positive_floor if len(sorted(fit["ev_year"].unique())) <= 2 else 0.0
        guarded_h_rows = [
            row
            for row in h_rows
            if row.get("reliability_min_count", 0) >= args.min_reliability_count
            and row.get("child_reliability_lcb_floor", 0.0) >= args.min_child_reliability_lcb_floor
            and row.get("parent_positive_alpha", 0.0) <= max_parent_positive_alpha
            and row.get("parent_positive_floor", 0.0) >= effective_min_parent_positive_floor
        ] or h_rows
        fdfhcu_selected = select_candidate(
            guarded_h_rows,
            alpha_all=args.alpha_all,
            alpha_plus=args.alpha_plus,
            alpha_hierarchy=args.alpha_hierarchy,
            alpha_late=args.alpha_late,
            beta_plus=args.beta_plus,
            require_utility=True,
            require_hierarchy=True,
            use_utility_objective=True,
            min_cal_cplus=args.min_cal_cplus,
            min_cal_positive_coverage=args.min_cal_positive_coverage,
            precision_first=False,
        )
        fdfhcu_metrics = add_method_from_thresholds(
            metric_rows,
            hierarchy_rows,
            projection_rows,
            label_rows,
            "F-DFHC-U full",
            scores_test,
            y_test,
            labels,
            maps,
            fdfhcu_selected["pos_thresholds"],
            fdfhcu_selected["neg_thresholds"],
            fdfhcu_selected,
            test_year,
            calib_year,
            use_hierarchy=True,
        )
        cal_fdfhcu_metrics = evaluate_thresholds(
            scores_cal,
            y_cal,
            fdfhcu_selected["pos_thresholds"],
            fdfhcu_selected["neg_thresholds"],
            labels,
            maps,
            use_hierarchy=True,
            aircraft_child_floor=float(fdfhcu_selected.get("aircraft_child_floor", 0.0) or 0.0),
            child_margin=float(fdfhcu_selected.get("child_margin", 0.0) or 0.0),
        )
        budget_rows.extend(
            summarize_budgeted_positive_closure(
                "F-DFHC-U full",
                fdfhcu_metrics,
                scores_test,
                y_test,
                test_year,
                calib_year,
            )
        )
        drift = exchangeability_diagnostics(y_cal, y_test, scores_cal, scores_test, cal_fdfhcu_metrics, fdfhcu_metrics)
        drift_rows.append(
            {
                "test_year": test_year,
                "calibration_year": calib_year,
                "calibration_samples": int(len(cal)),
                "test_samples": int(len(test)),
                "certificate_valid": bool(fdfhcu_metrics["empirical_all_risk"] <= fdfhcu_selected["upper_all_risk"]),
                "certified_all_bound": fdfhcu_selected["upper_all_risk"],
                **drift,
            }
        )
        add_method_from_thresholds(
            metric_rows,
            hierarchy_rows,
            projection_rows,
            label_rows,
            "F-DFHC-U w/o projection",
            scores_test,
            y_test,
            labels,
            maps,
            fdfhcu_selected["pos_thresholds"],
            fdfhcu_selected["neg_thresholds"],
            fdfhcu_selected,
            test_year,
            calib_year,
            use_hierarchy=False,
        )

        for method, selected in [
            ("Instancewise RCPS", instance_selected),
            ("LTT without hierarchy", ltt_selected),
            ("F-DFHC without utility", fdfhc_selected),
            ("F-DFHC-U full", fdfhcu_selected),
            ("F-DFHC-U w/o projection", fdfhcu_selected),
        ]:
            for idx, label in enumerate(labels):
                threshold_rows.append(
                    {
                        "method": method,
                        "test_year": test_year,
                        "label": label,
                        "node_type": "parent" if is_parent_label(label) else "child",
                        "parent": parent_of(label),
                        "is_late_emerging": label in LATE_EMERGING_LABELS,
                        "close_positive_threshold": float(selected["pos_thresholds"][idx]),
                        "close_negative_threshold": float(selected["neg_thresholds"][idx]),
                        "aircraft_child_floor": selected.get("aircraft_child_floor", 0.0),
                        "child_margin": selected.get("child_margin", 0.0),
                    }
                )

        for method, rows, use_h in [
            ("LTT without hierarchy", no_h_rows, False),
            ("F-DFHC-U", h_rows, True),
        ]:
            top = sorted(rows, key=lambda row: (row["lower_positive_coverage"], row["cal_close_positive_precision"]))
            for row in top:
                if row["upper_all_risk"] > args.alpha_all or row["upper_plus_risk"] > args.alpha_plus or row["upper_late_risk"] > args.alpha_late:
                    continue
                if use_h and row["upper_hierarchy_risk"] > args.alpha_hierarchy:
                    continue
                test_metrics = evaluate_thresholds(
                    scores_test,
                    y_test,
                    row["pos_thresholds"],
                    row["neg_thresholds"],
                    labels,
                    maps,
                    use_hierarchy=use_h,
                    aircraft_child_floor=float(row.get("aircraft_child_floor", 0.0) or 0.0),
                    child_margin=float(row.get("child_margin", 0.0) or 0.0),
                )
                frontier_rows.append(
                    {
                        "method": method,
                        "test_year": test_year,
                        "positive_coverage": test_metrics["positive_coverage"],
                        "close_positive_precision": test_metrics["close_positive_precision"],
                        "overall_coverage": test_metrics["coverage"],
                        "empirical_all_risk": test_metrics["empirical_all_risk"],
                        "lower_positive_coverage": row["lower_positive_coverage"],
                        "certified_all_bound": row["upper_all_risk"],
                        "child_positive_alpha": row["child_positive_alpha"],
                        "parent_positive_alpha": row["parent_positive_alpha"],
                        "personnel_parent_positive_alpha": row["personnel_parent_positive_alpha"],
                        "parent_positive_floor": row["parent_positive_floor"],
                        "negative_alpha": row["negative_alpha"],
                        "reliability_floor": row["reliability_floor"],
                        "child_reliability_lcb_floor": row["child_reliability_lcb_floor"],
                        "aircraft_child_floor": row.get("aircraft_child_floor", 0.0),
                        "child_margin": row.get("child_margin", 0.0),
                    }
                )
        for method, pos, neg, use_h in [
            ("BR-TFIDF threshold", br_pos, br_neg, False),
            ("ES3D", es3d_pos, es3d_neg, False),
        ]:
            metrics = evaluate_thresholds(scores_test, y_test, pos, neg, labels, maps, use_hierarchy=use_h)
            frontier_rows.append(
                {
                    "method": method,
                    "test_year": test_year,
                    "positive_coverage": metrics["positive_coverage"],
                    "close_positive_precision": metrics["close_positive_precision"],
                    "overall_coverage": metrics["coverage"],
                    "empirical_all_risk": metrics["empirical_all_risk"],
                    "lower_positive_coverage": math.nan,
                    "certified_all_bound": math.nan,
                }
            )

        forward_rows.append(
            {
                "test_year": test_year,
                "calibration_year": calib_year,
                "empirical_risk": fdfhcu_metrics["empirical_all_risk"],
                "certified_bound": fdfhcu_selected["upper_all_risk"],
                "bound_valid": bool(fdfhcu_metrics["empirical_all_risk"] <= fdfhcu_selected["upper_all_risk"]),
                "close_positive_precision": fdfhcu_metrics["close_positive_precision"],
                "positive_coverage": fdfhcu_metrics["positive_coverage"],
                "event_positive_coverage": fdfhcu_metrics["event_positive_coverage"],
                "overall_coverage": fdfhcu_metrics["coverage"],
            }
        )
        fold_rows.append(
            {
                "test_year": test_year,
                "fit_years": ",".join(map(str, sorted(fit["ev_year"].unique()))),
                "calibration_year": calib_year,
                "fit_samples": int(len(fit)),
                "calibration_samples": int(len(cal)),
                "test_samples": int(len(test)),
                "parent_labels": int(len(parents)),
                "child_labels": int(len(children)),
                "labels": int(len(labels)),
                "selected_child_positive_alpha": fdfhcu_selected["child_positive_alpha"],
                "selected_parent_positive_alpha": fdfhcu_selected["parent_positive_alpha"],
                "selected_personnel_parent_positive_alpha": fdfhcu_selected["personnel_parent_positive_alpha"],
                "selected_parent_positive_floor": fdfhcu_selected["parent_positive_floor"],
                "selected_negative_alpha": fdfhcu_selected["negative_alpha"],
                "selected_late_positive_alpha": fdfhcu_selected["late_positive_alpha"],
                "selected_lower_positive_coverage": fdfhcu_selected["lower_positive_coverage"],
                "selected_child_reliability_lcb_floor": fdfhcu_selected["child_reliability_lcb_floor"],
                "selected_aircraft_child_floor": fdfhcu_selected.get("aircraft_child_floor", 0.0),
                "selected_child_margin": fdfhcu_selected.get("child_margin", 0.0),
                "selected_max_parent_positive_alpha": max_parent_positive_alpha,
                "effective_min_parent_positive_floor": effective_min_parent_positive_floor,
            }
        )

    write_csv(OUT_DIR / f"f_dfhc_u_metrics_{suffix}.csv", metric_rows)
    write_csv(OUT_DIR / f"f_dfhc_u_hierarchy_{suffix}.csv", hierarchy_rows)
    write_csv(OUT_DIR / f"f_dfhc_u_candidates_{suffix}.csv", candidate_rows)
    write_csv(OUT_DIR / f"f_dfhc_u_thresholds_{suffix}.csv", threshold_rows)
    write_csv(OUT_DIR / f"f_dfhc_u_folds_{suffix}.csv", fold_rows)
    write_csv(OUT_DIR / f"f_dfhc_u_forward_certificate_{suffix}.csv", forward_rows)
    write_csv(OUT_DIR / f"f_dfhc_u_label_details_{suffix}.csv", label_rows)
    write_csv(OUT_DIR / f"f_dfhc_u_projection_stats_{suffix}.csv", projection_rows)
    write_csv(OUT_DIR / f"f_dfhc_u_frontier_{suffix}.csv", frontier_rows)
    write_csv(OUT_DIR / f"f_dfhc_u_budgeted_closure_{suffix}.csv", budget_rows)
    write_csv(OUT_DIR / f"f_dfhc_u_exchangeability_diagnostics_{suffix}.csv", drift_rows)

    df = pd.DataFrame(metric_rows)
    if not df.empty:
        summary = (
            df.groupby("method", as_index=False)
            .agg(
                test_years=("test_year", "nunique"),
                close_positive_precision=("close_positive_precision", "mean"),
                positive_coverage=("positive_coverage", "mean"),
                event_positive_coverage=("event_positive_coverage", "mean"),
                event_close_positive_rate=("event_close_positive_rate", "mean"),
                event_any_closure_rate=("event_any_closure_rate", "mean"),
                defer_rate=("defer_rate", "mean"),
                workload_reduction=("workload_reduction", "mean"),
                overall_coverage=("overall_coverage", "mean"),
                empirical_all_risk=("empirical_all_risk", "mean"),
                certified_all_bound=("certified_all_bound", "mean"),
                projected_hierarchy_violations=("projected_hierarchy_violations", "sum"),
                raw_hierarchy_violations=("raw_hierarchy_violations", "sum"),
                late_premature_closure_risk=("late_premature_closure_risk", "mean"),
                parent_close_positive_precision=("parent_close_positive_precision", "mean"),
                child_close_positive_precision=("child_close_positive_precision", "mean"),
                parent_positive_coverage=("parent_positive_coverage", "mean"),
                child_positive_coverage=("child_positive_coverage", "mean"),
                parent_f1=("parent_f1", "mean"),
                child_f1=("child_f1", "mean"),
            )
            .to_dict("records")
        )
    else:
        summary = []
    write_csv(OUT_DIR / f"f_dfhc_u_summary_{suffix}.csv", summary)

    target = df[df["method"] == "F-DFHC-U full"] if not df.empty else pd.DataFrame()
    lines = [f"# F-DFHC-U assessment ({suffix})", ""]
    if not target.empty:
        lines.append("| Test year | C+ P | P cov. | Overall cov. | Risk | Cert. | H viol. | Late risk | C+ pass | Utility pass |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for row in target.to_dict("records"):
            cplus_passed = row["close_positive_precision"] >= 0.900
            utility_passed = (
                row["close_positive_precision"] >= 0.900
                and row["positive_coverage"] >= 0.050
                and row["overall_coverage"] >= 0.700
                and row["empirical_all_risk"] <= 0.050
                and row["certified_all_bound"] <= 0.080
                and row["projected_hierarchy_violations"] == 0
                and row["late_premature_closure_risk"] <= 0.010
            )
            lines.append(
                f"| {int(row['test_year'])} | {row['close_positive_precision']:.3f} | {row['positive_coverage']:.3f} | "
                f"{row['overall_coverage']:.3f} | {row['empirical_all_risk']:.3f} | {row['certified_all_bound']:.3f} | "
                f"{int(row['projected_hierarchy_violations'])} | {row['late_premature_closure_risk']:.4f} | "
                f"{'yes' if cplus_passed else 'no'} | {'yes' if utility_passed else 'no'} |"
            )
    (OUT_DIR / f"F_DFHC_U_ASSESSMENT_{suffix}.md").write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "full": args.full,
        "seconds": round(time.time() - start, 3),
        "jobs": jobs,
        "test_years": test_years,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "data_file": args.data_file,
        "alpha_all": args.alpha_all,
        "alpha_plus": args.alpha_plus,
        "alpha_hierarchy": args.alpha_hierarchy,
        "alpha_late": args.alpha_late,
        "beta_plus": args.beta_plus,
        "min_cal_cplus": args.min_cal_cplus,
        "min_cal_positive_coverage": args.min_cal_positive_coverage,
        "min_reliability_count": args.min_reliability_count,
        "min_child_reliability_lcb_floor": args.min_child_reliability_lcb_floor,
        "min_parent_positive_floor": args.min_parent_positive_floor,
        "utility_weight": args.utility_weight,
        "delta": args.delta,
        "output_dir": str(OUT_DIR),
    }
    (OUT_DIR / f"run_meta_{suffix}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--label-level", type=int, default=2)
    parser.add_argument("--min-global-support", type=int, default=30)
    parser.add_argument("--min-train-support", type=int, default=20)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--data-file", type=str, default="")
    parser.add_argument("--test-years", type=str, default="")
    parser.add_argument("--max-features", type=int, default=25000)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--alpha-all", type=float, default=0.080)
    parser.add_argument("--alpha-plus", type=float, default=0.100)
    parser.add_argument("--alpha-hierarchy", type=float, default=0.000)
    parser.add_argument("--alpha-late", type=float, default=0.010)
    parser.add_argument("--beta-plus", type=float, default=0.000)
    parser.add_argument("--min-cal-cplus", type=float, default=0.925)
    parser.add_argument("--min-cal-positive-coverage", type=float, default=0.030)
    parser.add_argument("--min-reliability-count", type=int, default=20)
    parser.add_argument("--min-child-reliability-lcb-floor", type=float, default=0.85)
    parser.add_argument("--min-parent-positive-floor", type=float, default=0.72)
    parser.add_argument("--utility-weight", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
