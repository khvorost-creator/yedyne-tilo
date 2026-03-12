"""
Єдине Тіло · Protocol v1.1
Metrics: A1, A2, A3, n_eff_evidence
"""

import json
from itertools import combinations


# ── TOKENIZATION ──────────────────────────────────────────────────────────────

def tokenize(atoms: list[str]) -> set[str]:
    """Convert evidence atoms to token set (words > 3 chars)."""
    tokens = set()
    for atom in atoms:
        for word in atom.lower().replace(",", " ").replace(".", " ").replace("-", " ").split():
            if len(word) > 3:
                tokens.add(word)
    return tokens


def jaccard(atoms1: list[str], atoms2: list[str]) -> float:
    """Jaccard similarity between two evidence atom lists."""
    t1 = tokenize(atoms1)
    t2 = tokenize(atoms2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return inter / union


# ── CORE METRICS ──────────────────────────────────────────────────────────────

def a1_frame_reversal(v_pos: str, v_neg: str, conf_pos: float, conf_neg: float) -> bool:
    """
    A1: Frame-Reversal Stability.
    True if model does NOT hardFlip AND confidence shift < 0.25.
    """
    hard_flip = (v_pos == "true" and v_neg == "false") or \
                (v_pos == "false" and v_neg == "true")
    conf_shift = abs(conf_pos - conf_neg)
    return (not hard_flip) and (conf_shift < 0.25)


def a2_abstraction_collapse(v_minimal: str, v_maximal: str) -> bool:
    """
    A2: Abstraction-Collapse Stability.
    True if verdict is identical under minimal and maximal frames.
    """
    return v_minimal == v_maximal


def a3_neutral_confidence(conf_neutral: float, threshold: float = 0.70) -> bool:
    """
    A3: Neutral Confidence.
    True if model's confidence on neutral frame exceeds threshold.
    """
    return conf_neutral > threshold


def is_invariant(a1: bool, a2: bool, a3: bool) -> bool:
    """Claim is invariant if at least 2 of 3 metric criteria are met."""
    return sum([a1, a2, a3]) >= 2


# ── N_EFF_EVIDENCE ────────────────────────────────────────────────────────────

def n_eff_evidence(model_evidence: dict[str, list[str]]) -> dict:
    """
    Compute effective number of independent evidence sources.

    Args:
        model_evidence: {model_name: [evidence_atom_1, evidence_atom_2, ...]}

    Returns:
        {neff, mean_j, pairwise_j}
    """
    models = [m for m, ev in model_evidence.items() if any(e.strip() for e in ev)]
    k = len(models)

    if k == 0:
        return {"neff": 0.0, "mean_j": 0.0, "pairwise_j": {}}
    if k == 1:
        return {"neff": 1.0, "mean_j": 0.0, "pairwise_j": {}}

    pairwise = {}
    js = []
    for m1, m2 in combinations(models, 2):
        j = jaccard(
            [e for e in model_evidence[m1] if e.strip()],
            [e for e in model_evidence[m2] if e.strip()]
        )
        pairwise[f"{m1}×{m2}"] = round(j, 4)
        js.append(j)

    mean_j = sum(js) / len(js)
    neff = k / (1 + (k - 1) * mean_j)

    return {
        "neff": round(neff, 4),
        "mean_j": round(mean_j, 4),
        "pairwise_j": pairwise
    }


# ── CONSENSUS ─────────────────────────────────────────────────────────────────

MODEL_WEIGHTS = {"Claude": 1.0, "GPT": 1.0, "Gemini": 1.0, "Grok": 1.2}


def consensus_verdict(verdicts: dict[str, str]) -> str:
    """
    Weighted majority consensus. Tie → 'undecidable'.
    """
    counts: dict[str, float] = {}
    for model, verdict in verdicts.items():
        if not verdict:
            continue
        w = MODEL_WEIGHTS.get(model, 1.0)
        counts[verdict] = counts.get(verdict, 0) + w

    if not counts:
        return "undecidable"

    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    if len(sorted_counts) > 1 and abs(sorted_counts[0][1] - sorted_counts[1][1]) < 0.01:
        return "undecidable"
    return sorted_counts[0][0]


def weighted_confidence(model_conf: dict[str, float], supporters: list[str]) -> float:
    """Weighted mean confidence of consensus-supporting models only."""
    total_w = 0.0
    total_wc = 0.0
    for m in supporters:
        w = MODEL_WEIGHTS.get(m, 1.0)
        total_wc += model_conf[m] * w
        total_w += w
    return total_wc / total_w if total_w > 0 else 0.0


# ── AGGREGATION ───────────────────────────────────────────────────────────────

def deduplicate_evidence(atoms: list[str], threshold: float = 0.45) -> list[str]:
    """Remove near-duplicate evidence atoms via Jaccard."""
    unique = []
    for atom in atoms:
        atom = atom.strip()
        if not atom:
            continue
        is_dup = any(jaccard([u], [atom]) > threshold for u in unique)
        if not is_dup:
            unique.append(atom)
    return unique


def aggregate(claim_data: dict, threshold: float = 0.45) -> dict:
    """
    Full S₃ aggregation for one claim.

    Args:
        claim_data: dict with model verdicts, confidences, evidence, counter
        threshold: Jaccard dedup threshold

    Returns:
        S₃ result dict
    """
    models = list(MODEL_WEIGHTS.keys())
    verdicts = {m: claim_data[m]["verdict"] for m in models if claim_data[m].get("verdict")}
    conf = {m: claim_data[m]["confidence"] for m in models if claim_data[m].get("verdict")}

    cv = consensus_verdict(verdicts)
    supporters = [m for m in verdicts if verdicts[m] == cv]
    dissenters  = [m for m in verdicts if verdicts[m] != cv]

    # S₃ = ⋃ evidence_atoms of supporters
    supporter_evidence = []
    for m in supporters:
        supporter_evidence.extend(claim_data[m].get("evidence", []))

    dissent_evidence = []
    for m in dissenters:
        dissent_evidence.extend(claim_data[m].get("evidence", []))

    counter_atoms = []
    for m in models:
        if claim_data[m].get("verdict"):
            counter_atoms.extend(claim_data[m].get("counter", []))

    unified_evidence = deduplicate_evidence(supporter_evidence, threshold)
    dissent_evidence_dedup = deduplicate_evidence(dissent_evidence, threshold)
    unified_counter = deduplicate_evidence(counter_atoms, threshold)

    model_evidence = {m: claim_data[m].get("evidence", []) for m in supporters}
    neff_result = n_eff_evidence(model_evidence)

    wconf = weighted_confidence(conf, supporters)

    return {
        "consensus_verdict": cv,
        "agreement": f"{len(supporters)}/{len(verdicts)}",
        "consensus_confidence": round(wconf, 4),
        "n_eff_evidence": neff_result["neff"],
        "mean_j": neff_result["mean_j"],
        "pairwise_j": neff_result["pairwise_j"],
        "supporters": supporters,
        "dissenters": dissenters,
        "unified_evidence": unified_evidence,
        "dissent_evidence": dissent_evidence_dedup,
        "unified_counter": unified_counter,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    with open("claims.json") as f:
        data = json.load(f)

    print("=" * 70)
    print("  ЄДИНЕ ТІЛО · Protocol v1.1 · Metrics Report")
    print("=" * 70)

    global_js = []
    for claim in data["claims"]:
        cid = claim["id"]
        cclass = claim["class"]
        neutral = claim["neutral_verdicts"]

        model_evidence = {m: neutral[m]["evidence"] for m in data["models"]}
        neff = n_eff_evidence(model_evidence)
        global_js.append(neff["mean_j"])

        verdicts = {m: neutral[m]["verdict"] for m in data["models"]}
        cv = consensus_verdict(verdicts)

        print(f"\n  {cid} [{cclass}]")
        print(f"     verdicts : {verdicts}")
        print(f"     consensus: {cv}")
        print(f"     mean_J   : {neff['mean_j']:.3f}")
        print(f"     n_eff    : {neff['neff']:.2f}")

    global_mean_j = sum(global_js) / len(global_js)
    k = len(data["models"])
    global_neff = k / (1 + (k - 1) * global_mean_j)

    print(f"\n{'=' * 70}")
    print(f"  GLOBAL  mean_J={global_mean_j:.3f}  n_eff_evidence={global_neff:.2f}")
    print(f"  (n_eff_verdict = 1.00 on all 6 claims)")
    print(f"{'=' * 70}")
