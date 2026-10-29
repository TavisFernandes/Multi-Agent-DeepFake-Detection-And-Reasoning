"""Run the production /analyse pipeline over an external test folder and score it.

Label = fake if any folder in the file's path starts with "fake", else real.
Usage: python training/eval_external.py "<folder>" [--csv out.csv]
"""
import argparse, csv, logging, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.WARNING)
import main  # noqa: E402

EXTS = main._IMAGE_EXTS | main._VIDEO_EXTS | main._AUDIO_EXTS


def run(path: Path) -> dict:
    mtype = main._resolve_media_type(path.name, "")
    agents = main._build_agents(str(path), mtype)
    score, verdict, *_ = main._compute_verdict(agents, mtype, return_details=True)
    scores = {a["agent"]: a.get("confidence") for a in agents if a.get("is_fake") is not None}
    return {"verdict": verdict, "score": score, "agent_scores": scores}


def outcome(truth: str, verdict: str) -> str:
    v = verdict.upper()
    if "DEEPFAKE" in v: return "correct" if truth == "fake" else "WRONG"
    if "AUTHENTIC" in v: return "correct" if truth == "real" else "WRONG"
    return "abstained"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()
    import asyncio; asyncio.run(main.startup_event())  # same model loading as the server
    files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in EXTS)
    rows, tally = [], Counter()
    for p in files:
        rel = p.relative_to(args.folder)
        truth = "fake" if any(part.lower().startswith("fake") for part in rel.parts[:-1]) else "real"
        r = run(p)
        o = outcome(truth, r["verdict"])
        tally[(truth, o)] += 1
        rows.append({"file": str(rel), "truth": truth, **r, "outcome": o})
        print(f"{o:9} {truth:4} {r['verdict']:22} {r['score']:.3f} {rel}", flush=True)
    print("\n" + "\n".join(f"{t:4} {o:9} {n}" for (t, o), n in sorted(tally.items())))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
