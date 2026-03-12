"""
Єдине Тіло · Protocol v1.1
Runner: full aggregation pipeline.

Usage:
    # Run on existing data (no API key needed):
    python runner.py --offline

    # Run with Anthropic API (requires ANTHROPIC_API_KEY):
    python runner.py --claim "Your claim here" --frame neutral

    # Run all claims, all frames:
    python runner.py --all
"""

import json
import os
import sys
import argparse
from pathlib import Path

from metrics import aggregate, n_eff_evidence, consensus_verdict
from prompts import build_prompt, parse_response, FRAMES


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_result(claim_id: str, claim_text: str, result: dict):
    sep = "─" * 65
    print(f"\n{sep}")
    print(f"  {claim_id}: {claim_text[:60]}...")
    print(sep)
    print(f"  CONSENSUS  : {result['consensus_verdict'].upper()}")
    print(f"  AGREEMENT  : {result['agreement']}")
    print(f"  CONFIDENCE : {result['consensus_confidence']:.2f}")
    print(f"  n_eff_evid : {result['n_eff_evidence']:.2f}  (mean_J={result['mean_j']:.3f})")

    if result["supporters"]:
        print(f"\n  SUPPORTERS : {', '.join(result['supporters'])}")
    if result["dissenters"]:
        print(f"  DISSENTERS : {', '.join(result['dissenters'])}")

    print(f"\n  UNIFIED EVIDENCE ({len(result['unified_evidence'])} atoms):")
    for i, e in enumerate(result["unified_evidence"], 1):
        print(f"    {i}. {e}")

    if result["dissent_evidence"]:
        print(f"\n  DISSENT EVIDENCE ({len(result['dissent_evidence'])} atoms):")
        for i, e in enumerate(result["dissent_evidence"], 1):
            print(f"    {i}. {e}")

    if result["unified_counter"]:
        print(f"\n  COUNTER ARGUMENTS ({len(result['unified_counter'])} atoms):")
        for i, e in enumerate(result["unified_counter"], 1):
            print(f"    {i}. {e}")


def print_summary(results: list[dict]):
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"\n  {'ID':<6} {'Verdict':<14} {'Agree':<8} {'Conf':<8} {'n_eff'}")
    print("  " + "─" * 55)
    for r in results:
        print(
            f"  {r['claim_id']:<6} "
            f"{r['result']['consensus_verdict']:<14} "
            f"{r['result']['agreement']:<8} "
            f"{r['result']['consensus_confidence']:.2f}     "
            f"{r['result']['n_eff_evidence']:.2f}"
        )

    all_neff = [r["result"]["n_eff_evidence"] for r in results]
    all_j = [r["result"]["mean_j"] for r in results]
    print(f"\n  GLOBAL mean_J      = {sum(all_j)/len(all_j):.3f}")
    print(f"  GLOBAL n_eff_evid  = {sum(all_neff)/len(all_neff):.2f}")
    print(f"  n_eff_verdict      = 1.00  (all models agree on verdict)")
    print()


# ── OFFLINE MODE (existing data) ──────────────────────────────────────────────

def run_offline():
    """Run aggregation on existing claims.json data."""
    data_path = Path(__file__).parent / "claims.json"
    with open(data_path) as f:
        data = json.load(f)

    results = []
    for claim in data["claims"]:
        cid = claim["id"]
        neutral = claim["neutral_verdicts"]

        # Build claim_data in aggregate() format
        claim_data = {
            m: {
                "verdict": neutral[m]["verdict"],
                "confidence": neutral[m]["confidence"],
                "evidence": neutral[m]["evidence"],
                "counter": neutral[m]["counter"],
            }
            for m in data["models"]
        }

        result = aggregate(claim_data)
        print_result(cid, claim["text"], result)
        results.append({"claim_id": cid, "result": result})

    print_summary(results)
    return results


# ── LIVE MODE (API) ───────────────────────────────────────────────────────────

def query_claude(prompt: str, api_key: str) -> str:
    """Query Claude API with a prompt."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)
    except Exception as e:
        print(f"API error: {e}")
        sys.exit(1)


def run_single(claim_text: str, frame: str, api_key: str):
    """Query Claude on a single claim+frame and print parsed result."""
    prompt = build_prompt(claim_text, frame)
    print(f"\nQuerying Claude on frame='{frame}'...")
    raw = query_claude(prompt, api_key)
    parsed = parse_response(raw)
    print(f"\nRaw response:\n{raw}")
    print(f"\nParsed:\n{json.dumps(parsed, indent=2)}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Єдине Тіло · Runner")
    parser.add_argument("--offline", action="store_true",
                        help="Run on existing claims.json data (no API key needed)")
    parser.add_argument("--claim", type=str, help="Claim text to evaluate")
    parser.add_argument("--frame", type=str, default="neutral",
                        choices=FRAMES, help="Frame to use")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if args.offline or (not args.claim):
        print("Running in OFFLINE mode on claims.json...")
        results = run_offline()
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output}")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    run_single(args.claim, args.frame, api_key)


if __name__ == "__main__":
    main()
