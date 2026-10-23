"""
eval_system_fusion_val.py
=========================
System-level check of detector checkpoint combinations under the production fusion (corrected_fixed), built from
the per-clip scores the robustness evals already wrote. fusion_val is development data (used for candidate
selection), so these counts are development evidence, not an independent real-world estimate.

A clip is "fake" unless it is RealVideo-RealAudio. Each scenario pairs a visual condition with an audio format, so
video and audio are degraded together as in a re-encoded upload.

Usage (from veritas/backend/training):
  python eval_system_fusion_val.py OUT.json SPATIAL.json TEMPORAL.json AUDIO.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion_policy import compute_verdict_corrected_fixed  # noqa: E402

SCENARIOS = {  # name -> (visual condition, audio format)
    "native": ("original", "original"),
    "jpeg_q60 + mp3_64k": ("jpeg_q60", "mp3_64k"),
    "down_x0.5 + aac_64k": ("down_x0.5", "aac_64k"),
    "h264_crf32 + aac_64k": ("h264_crf32", "aac_64k"),
    "720p_face25_h264_crf28 + aac_64k": ("720p_face25_h264_crf28", "aac_64k"),
}
SPATIAL = ["spatial_agent_lc.pth", "spatial_agent_lc_v2_aug_s0.pth"]
AUDIO = ["audio_agent_lc.pth", "audio_agent_lc_v2_aug_s1.pth"]


def agent(name, p):
    return {"agent": name, "status": "trained", "confidence": p, "is_fake": p >= 0.5, "has_audio": True}


def main(out, sp_json, te_json, au_json):
    sp, te, au = (json.loads(Path(f).read_text(encoding="utf-8")) for f in (sp_json, te_json, au_json))
    clips = list(te["scores"]["original"])
    fake = {c: not c.startswith("RealVideo-RealAudio") for c in clips}
    report = {"split": "fusion_val", "split_role": "development (used for candidate selection)", "fusion": "corrected_fixed",
              "temporal": te["checkpoint"], "results": {}}
    for s_ck in SPATIAL:
        for a_ck in AUDIO:
            key = f"{s_ck} + {a_ck}"
            report["results"][key] = {}
            print(key)
            for scen, (vc, ac) in SCENARIOS.items():
                n = {"real_DEEPFAKE": 0, "real_UNCERTAIN": 0, "fake_AUTHENTIC": 0, "fake_UNCERTAIN": 0,
                     "n_real": 0, "n_fake": 0}
                for c in clips:
                    agents = [agent("Spatial Analysis", sp["checkpoints"][s_ck]["scores"][vc][c]),
                              agent("Audio Analysis", au["checkpoints"][a_ck]["scores"][ac][c])]
                    if te["scores"][vc][c] == te["scores"][vc][c]:  # NaN = temporal produced no score
                        agents.append(agent("Temporal Analysis", te["scores"][vc][c]))
                    verdict = compute_verdict_corrected_fixed(agents, "video")[1]
                    cls = "fake" if fake[c] else "real"
                    n[f"n_{cls}"] += 1
                    if verdict.startswith("UNCERTAIN"):
                        n[f"{cls}_UNCERTAIN"] += 1
                    elif cls == "real" and verdict == "DEEPFAKE":
                        n["real_DEEPFAKE"] += 1
                    elif cls == "fake" and verdict == "AUTHENTIC":
                        n["fake_AUTHENTIC"] += 1
                report["results"][key][scen] = n
                print(f"  {scen:34s} real->DEEPFAKE={n['real_DEEPFAKE']}/{n['n_real']} "
                      f"real->UNCERTAIN={n['real_UNCERTAIN']} fake->AUTHENTIC={n['fake_AUTHENTIC']}/{n['n_fake']} "
                      f"fake->UNCERTAIN={n['fake_UNCERTAIN']}")
    Path(out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:5])
