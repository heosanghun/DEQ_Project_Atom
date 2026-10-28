#!/usr/bin/env python3
"""codex_grade_local.py - per-item strict/lenient/boxed flags with the LOCKED grader (s5_eval_grader_v2.py, SHA 90cc93e4...).
Same logic as g2v5_s0_grade.py but reads gold solutions from data/dev600_records.jsonl (dev 600 only; no other data is shipped).
Usage: python scripts/codex_grade_local.py <spec.json> <out.json>   spec = {label: [gen_jsonl_path, order_json_path]}
Do NOT modify s5_eval_grader_v2.py. Verify its SHA first (the import below does).
rev2 (2026-10-20, Codex QUESTION-C004-REGRADING): all work is under a __main__ guard (Windows spawn re-import).
rev3 (2026-10-20 17:5x, CC): ROOT CAUSE of the local/server flag differences and of the WinError 6 / EOFError floods:
  math-verify's `timeout` uses signal.alarm on POSIX (server) but a multiprocessing.Process PER CALL on Windows, and that spawn fails here
  (handle duplication), so every symbolic-equivalence check silently raised and was swallowed by the grader -> math-verify effectively OFF
  locally (e.g. 16/3 vs \\frac{16}{3} judged unequal). On Windows this wrapper now replaces `timeout` in math_verify.grader / math_verify.parser
  with a no-op decorator BEFORE the grader is imported. Equivalence logic itself is untouched; only the per-call time limit is removed
  (run the whole script under an outer time limit instead). The grader file is unchanged (SHA verified on import)."""
import json, os, sys, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def _windows_timeout_shim():
    if os.name == "posix":
        return "posix: math-verify signal.alarm timeout kept (same as server)"
    import math_verify.grader as _g, math_verify.parser as _p
    def _no_timeout(timeout_seconds=None, **_kw):
        return lambda f: f
    _g.timeout = _no_timeout; _p.timeout = _no_timeout
    return "windows: math-verify per-call multiprocessing timeout replaced by no-op (equivalence logic unchanged)"


def main():
    print("[LOCAL-SHIM]", _windows_timeout_shim(), flush=True)
    from s7_eval_score_p6_v36 import verify_grader, load_gen_jsonl
    grade = verify_grader()
    ds = {}
    for l in open(os.path.join(ROOT, "data", "dev600_records.jsonl"), encoding="utf-8"):
        if l.strip():
            r = json.loads(l); ds[int(r["id"])] = r
    spec = json.load(open(sys.argv[1])); out = {}
    for label, (path, order) in spec.items():
        o = json.load(open(order)); ids = [int(x) for x in (o["indices"] if isinstance(o, dict) else o)]
        g = load_gen_jsonl(path); res = {}
        for q in ids:
            if q not in g: continue
            s, l, b, _ = grade(g[q].get("generation", ""), ds[q]["solution"])
            res[q] = [int(bool(s)), int(bool(l)), int(bool(b))]
        out[label] = res
        print(f"[S0] {label}: n {len(res)} strict {sum(v[0] for v in res.values())} lenient {sum(v[1] for v in res.values())} boxed {sum(v[2] for v in res.values())}", flush=True)
    json.dump(out, open(sys.argv[2], "w")); print("[S0-DONE]", sys.argv[2])


if __name__ == "__main__":
    main()
