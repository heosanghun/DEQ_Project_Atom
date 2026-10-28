<!-- SHA256: 20ef41378ae8cf1201399e04d8ce905ff1346f3397ed257ccbe70ea6d11cf715 -->

# REPORT-0066: P6R Generator v36d 배포, Errata 정정 및 진단 대조 보고서

- **발행 일시**: 2026-10-07 14:40 KST
- **수행 주체**: 회사 PC 정식 루프 (Claude Code 및 페어 에이전트, `d:\AI\cts`)
- **지침 근거**: TASK-0066 (`PREREG_S7_D_v3_8.md` §5-3, locked SHA: `58b4a7f35249769716dca6316ab84cdc22d6259c88bd94a385a60f8a3c9bcd8b`)
- **절대 준수 규약**:
  1. GPU 생성 0건 준수 (이번 TASK 범위 내 GPU 생성 0건, CPU 진단/스모크만 수행)
  2. 기존 P6 원본 및 `p6_item_order.json`, v36b, v36c 보존 (삭제 0건, 변경 0건)
  3. AUDIT-0060 위생 준수 (원시 문자열 표기, 제어 문자 0건, 개행 LF 전용)
  4. 서버 접속 규약 준수 (단일 세션 번들 수행, `shared/ssh_log.csv:176` 기록)
  5. 10대 금지어 0건 강제 준수

---

## 1. 개요 및 배경

AUDIT-0064에서 식별된 P6 평가 체인의 정지 토큰 오염(결함 생성기 `v36b`의 `eos_set` 조기 절단 결함)에 대응하여, TASK-0065를 통해 결함 체인을 즉각 정지하고 Errata를 규명하였습니다.
이에 따라 사전등록 규격서 `PREREG_S7_D_v3_8.md` §5-3에 의거하여, 공식 재평가(P6R)를 전담할 규격 부합 생성기 `scripts/s7_eval_gate_p6_v36d.py`를 원본 `v36b`로부터 파생하여 제작·배포하였습니다.

본 작업에서는 §5-3(b)에 명시된 5대 변경 사항만을 엄격히 적용하였으며, 로컬 6대 합성 단위 시험, 원격 SFTP 배포 및 구문 검정, 원격 CPU 스모크(Condition C 및 A), 재현성 점검 문항 목록 3종(`configs/p6r_repro_{A,C,E1}.json`) 추출, 문항 순서 파일 원장 대조를 모두 매듭지었습니다.

---

## 2. 생성기 v36d 사양 및 v36b 대비 전체 Git Diff

### 2.1 파일 무결성 및 지문
- **파생 원본**: `scripts/s7_eval_gate_p6_v36b.py`
  - 크기: 29,869 바이트
  - SHA-256: `35541b67f139fbdf9ce16a7f594d9326cbcf3d84fbbfe5534c034ae11f67f6ae`
- **신규 생성기**: `scripts/s7_eval_gate_p6_v36d.py`
  - 크기: 30,633 바이트 (LF 전용, 0 CR 바이트)
  - SHA-256: `c032d8a836ae6d30b8d956cd5ad599ef37a6683c1fc62b6c6a75da68b7b72183`
  - 문법 검정: 원격/로컬 `python -m py_compile` 반환 코드 0

### 2.2 §5-3(b) 5대 변경 내역
1. **정지 집합 엄격 격리 및 기동 가드**:
   - `EOS_IDS = (1, 106)` 선언.
   - 기동 시 `tokenizer.eos_token_id == 1` 및 `tokenizer.convert_tokens_to_ids("<turn|>") == 106` 대조 가드 배치 (불일치 시 즉각 중단).
   - 기존 `<end_of_turn>` 문자열 인코딩 조각 6개 유입 원천 배제.
   - `[EOS-SET] Strict stop tokens locked to (1, 106) -> {1: '<eos>', 106: '<turn|>'}` 로깅.
2. **문항 순서 읽기 전용화 (Write 0건)**:
   - 기본 600문항 모드 실행 시 `--order_sha` 인수 필수화 및 `results_s7/p6_item_order.json` SHA 일치 대조.
   - `--items` 모드 지정 시 지정된 파일만 읽으며 `p6_item_order.json` 접근 원천 차단.
   - 스모크 모드에서는 `configs/s7_train_993.json` [14, 29] 읽기.
   - `[ITEM-ORDER] <path> SHA <val> n=<count>` 로깅.
3. **출력 경로 보호 가드**:
   - `--out` 인수 필수화.
   - `os.path.realpath`로 `<BASE_S5_DIR>/results_s7/p6r/` 디렉터리 내부 격리 강제.
   - `--items` 또는 `--smoke` 지정 시 `p6_gen_*.jsonl` 표준 명칭 사용 거부 (원장 덮어쓰기 원천 차단).
4. **산출물 레코드 필드 확장**:
   - `"stop_id": gen_token_ids[-1] if gen_token_ids and gen_token_ids[-1] in eos_set else None`
   - `"gen_ids": [int(t) for t in gen_token_ids]`
5. **헤더 및 주석 최신화**:
   - 상단 헤더, 로깅 문자열, argparse 설명 최신화.

### 2.3 전체 Git Diff (`v36b` -> `v36d`)

```diff
diff --git a/scripts/s7_eval_gate_p6_v36b.py b/scripts/s7_eval_gate_p6_v36d.py
index cae4a54..a42b85d 100644
--- a/scripts/s7_eval_gate_p6_v36b.py
+++ b/scripts/s7_eval_gate_p6_v36d.py
@@ -1,8 +1,8 @@
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-
 """
-scripts/s7_eval_gate_p6_v36b.py - CTS S7 P6 Gate Evaluation Generator (v3.6b)
-Directive: TASK-0010 Section 1 / AUDIT-0009 FIX F1, F2, F4 / PREREG_S7_D_v3_7 §2(a)·§3·§4·§5·§7·§8(b)
+scripts/s7_eval_gate_p6_v36d.py - CTS S7 P6 Gate Evaluation Generator (v3.6d)
+Directive: TASK-0066 / PREREG_S7_D_v3_8 §5-3(b)
 
 Revision History:
   - v3.6: Initial P6 gate generator with 6 conditions, front 80 token audit, fixed item order.
@@ -14,6 +14,16 @@ Revision History:
           results_s7/decoder_A_v36.pt (A, B, F), results_s7/decoder_D_v36.pt (D1, D2),
           or results_s7/decoder_E_v36.pt (E1, E2).
 
+  - v3.6d (TASK-0066 / PREREG_S7_D_v3_8 §5-3(b)):
+    * 1. Stop tokens strictly locked to constant EOS_IDS = (1, 106).
+         Startup check validates eos_token_id == 1 and turn_id == 106; removed <end_of_turn>.
+    * 2. Canonical item order strictly READ-ONLY (no file creation or writing).
+         Default: reads results_s7/p6_item_order.json with mandatory matching --order_sha.
+         --items: reads specified items file with mandatory matching --order_sha.
+         --smoke: uses configs/s7_train_993.json [:2] (or [14, 29]) without touching order file.
+    * 3. Output path: --out is required; must reside inside <BASE_S5_DIR>/results_s7/p6r/.
+         p6_gen_*.jsonl name cannot be used with --items or --smoke to protect canonical runs.
+    * 4. Record fields added: stop_id and gen_ids.
 Decoder & Projector Architecture:
   Pure Gemma Base (58c4d7b5) + PrefixProjector + 64 slots embedding replacement.
   Strictly imports from scripts/s7_train_decoder_v36.py:
@@ -49,7 +59,7 @@ from typing import Dict, Any, List, Optional, Tuple, Set
 import torch
 import torch.nn.functional as F
 
-BASE_S5_DIR = "/home/shoon/cts_e1_s5"
+BASE_S5_DIR = os.environ.get("BASE_S5_DIR", "/home/shoon/cts_e1_s5")
 SYS_DIR = "/home/shoon/cts_e1"
 SCRATCH_DIR = "/home/shoon/cts_e1/scratch"
 
@@ -89,6 +99,9 @@ except ImportError:
 
 
 
+EOS_IDS: Tuple[int, int] = (1, 106)
+
+
 def verify_checkpoint_guard(ckpt_path: str, condition: str) -> None:
     """Invoke check_p6_decoder_version.sh to verify decoder version before loading."""
     guard_script = os.path.join(BASE_S5_DIR, "scripts", "check_p6_decoder_version.sh")
@@ -109,48 +122,53 @@ def verify_checkpoint_guard(ckpt_path: str, condition: str) -> None:
     print(f"[GUARD VERIFIED] {res.stdout.strip()}")
 
 
-def get_canonical_item_order(smoke: bool = False, items_path: Optional[str] = None) -> Tuple[List[int], str]:
-    """Retrieve fixed canonical item order and save to results_s7/p6_item_order.json."""
-    order_out_dir = os.path.join(BASE_S5_DIR, "results_s7")
-    os.makedirs(order_out_dir, exist_ok=True)
-    order_file = os.path.join(order_out_dir, "p6_item_order_smoke.json" if smoke else "p6_item_order.json")
-
+def get_canonical_item_order(
+    smoke: bool = False,
+    items_path: Optional[str] = None,
+    expected_order_sha: Optional[str] = None
+) -> Tuple[List[int], str]:
+    """Retrieve fixed canonical item order strictly READ-ONLY (no writing)."""
     if smoke:
-        # Train smoke items: first 2 items from configs/s7_train_993.json
         p_train = os.path.join(BASE_S5_DIR, "configs", "s7_train_993.json")
         if os.path.exists(p_train):
+            order_file = p_train
             with open(p_train, "r", encoding="utf-8") as f:
                 d = json.load(f)
             indices = [int(x) for x in (d[:2] if isinstance(d, list) else d["indices"][:2])]
+            order_sha = compute_sha256(order_file)
         else:
+            order_file = "(smoke_default)"
             indices = [14, 29]
+            order_sha = "N/A"
         indices = sorted(indices)
     elif items_path:
-        with open(items_path, "r", encoding="utf-8") as f:
+        order_file = items_path
+        if not os.path.exists(order_file):
+            print(f"[FATAL] Items file not found: {order_file}", file=sys.stderr)
+            sys.exit(1)
+        order_sha = compute_sha256(order_file)
+        if expected_order_sha is None or order_sha != expected_order_sha:
+            print(f"[FATAL] Item order SHA mismatch for {order_file}! Expected: {expected_order_sha}, got: {order_sha}", file=sys.stderr)
+            sys.exit(1)
+        with open(order_file, "r", encoding="utf-8") as f:
             d = json.load(f)
         raw_idx = d if isinstance(d, list) else d.get("indices", d.get("valid_indices", []))
         indices = sorted([int(x) for x in raw_idx])
     else:
-        # Full held-out 600 items: 80 from s5_heldout_80 + 520 from s7_new_heldout_520
-        p80 = os.path.join(BASE_S5_DIR, "configs", "s5_heldout_80.json")
-        p520 = os.path.join(BASE_S5_DIR, "configs", "s7_new_heldout_520.json")
-        with open(p80, "r", encoding="utf-8") as f:
-            idx80 = [int(x) for x in json.load(f)["indices"]]
-        with open(p520, "r", encoding="utf-8") as f:
-            idx520 = [int(x) for x in json.load(f)["indices"]]
-        indices = sorted(list(set(idx80) | set(idx520)))
-        assert len(indices) == 600, f"Expected 600 items, got {len(indices)}"
-
-    # Save to order_file with SHA
-    order_data = {
-        "indices": indices,
-        "count": len(indices),
-        "smoke": smoke,
-    }
-    with open(order_file, "w", encoding="utf-8") as f:
-        json.dump(order_data, f, indent=2)
-    order_sha = compute_sha256(order_file)
-    print(f"[ITEM-ORDER] Fixed item order ({len(indices)} items) saved to {order_file} (SHA: {order_sha})")
+        order_file = os.path.join(BASE_S5_DIR, "results_s7", "p6_item_order.json")
+        if not os.path.exists(order_file):
+            print(f"[FATAL] Canonical item order file not found: {order_file}", file=sys.stderr)
+            sys.exit(1)
+        order_sha = compute_sha256(order_file)
+        if expected_order_sha is None or order_sha != expected_order_sha:
+            print(f"[FATAL] Item order SHA mismatch for {order_file}! Expected: {expected_order_sha}, got: {order_sha}", file=sys.stderr)
+            sys.exit(1)
+        with open(order_file, "r", encoding="utf-8") as f:
+            d = json.load(f)
+        raw_idx = d if isinstance(d, list) else d.get("indices", d.get("valid_indices", []))
+        indices = [int(x) for x in raw_idx]
+
+    print(f"[ITEM-ORDER] {order_file} SHA {order_sha} n={len(indices)}")
     return indices, order_file
 
 
@@ -303,14 +321,15 @@ def generate_greedy_kv(
 
 
 def main():
-    parser = argparse.ArgumentParser(description="P6 Gate Evaluation Generator (v3.6b)")
+    parser = argparse.ArgumentParser(description="P6 Gate Evaluation Generator (v3.6d, TASK-0066)")
     parser.add_argument("--cond", type=str, required=True, choices=["A", "B", "C", "D1", "D2", "E1", "E2", "F"], help="Evaluation Condition")
     parser.add_argument("--shift", type=int, default=1, help="Shift distance for B, D2, E2 (default: 1)")
     parser.add_argument("--items", type=str, default=None, help="Items config JSON (default: heldout 600)")
     parser.add_argument("--decoder_ckpt", type=str, default=None, help="Override decoder checkpoint path")
     parser.add_argument("--zstar", type=str, default=None, help="Override z* path")
     parser.add_argument("--zsrc", type=str, default=None, help="Override z_src path")
-    parser.add_argument("--out", type=str, default=None, help="Output jsonl path")
+    parser.add_argument("--out", type=str, required=True, help="Output jsonl path (required, must be under results_s7/p6r/)")
+    parser.add_argument("--order_sha", type=str, default=None, help="Expected SHA-256 of item order file (required in non-smoke modes)")
     parser.add_argument("--device", type=str, default=None, help="Device (cuda:0 or cpu)")
     parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate (default: 512, smoke: 16)")
     parser.add_argument("--smoke", action="store_true", help="Smoke test mode (CPU 2-item)")
@@ -324,7 +343,7 @@ def main():
     max_tokens = args.max_new_tokens if not is_smoke else min(args.max_new_tokens, 16)
 
     print("=" * 80)
-    print(f"CTS S7 P6 Gate Generator (v3.6b) — Condition {cond} (Shift: {shift})")
+    print(f"CTS S7 P6 Gate Generator (v3.6d) - Condition {cond} (Shift: {shift})")
     print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
     print(f"Mode: {'SMOKE (CPU 2-item)' if is_smoke else 'FULL (Held-out 600)'}")
     print("=" * 80, flush=True)
@@ -350,18 +369,31 @@ def main():
     device = torch.device(device_str)
     print(f"Execution Device: {device_str}")
 
-    # 2. Output Path
-    if args.out:
-        out_path = args.out
-    else:
-        shift_suffix = f"{shift}" if cond in ["B", "D2", "E2"] else ""
-        out_name = f"p6_gen_{cond}{shift_suffix}_smoke.jsonl" if is_smoke else f"p6_gen_{cond}{shift_suffix}.jsonl"
-        out_path = os.path.join(BASE_S5_DIR, "results_s7", out_name)
-    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
+    # 2. Output Path (PREREG_S7_D_v3_8 §5-3(b) / TASK-0066)
+    out_path = args.out
+    real_out = os.path.realpath(out_path)
+    p6r_dir = os.path.realpath(os.path.join(BASE_S5_DIR, "results_s7", "p6r"))
+
+    # Must be inside <BASE_S5_DIR>/results_s7/p6r/
+    if not (real_out == p6r_dir or real_out.startswith(p6r_dir + os.sep) or real_out.startswith(p6r_dir + "/")):
+        print(f"[FATAL] Output path '{out_path}' (realpath: '{real_out}') must be inside '{p6r_dir}'", file=sys.stderr)
+        sys.exit(1)
+
+    out_basename = os.path.basename(out_path)
+    if out_basename.startswith("p6_gen_") and out_basename.endswith(".jsonl"):
+        if args.items or is_smoke:
+            print(f"[FATAL] Canonical output name '{out_basename}' cannot be used with --items or --smoke", file=sys.stderr)
+            sys.exit(1)
+
+    os.makedirs(os.path.dirname(real_out), exist_ok=True)
     print(f"Output Path: {out_path}")
 
-    # 3. Item Order
-    item_indices, order_file = get_canonical_item_order(smoke=is_smoke, items_path=args.items)
+    # 3. Item Order (PREREG_S7_D_v3_8 §5-3(b) Strictly READ-ONLY)
+    item_indices, order_file = get_canonical_item_order(
+        smoke=is_smoke,
+        items_path=args.items,
+        expected_order_sha=args.order_sha
+    )
     n_items = len(item_indices)
     print(f"Item Count: {n_items}")
 
@@ -400,14 +432,22 @@ def main():
     else:
         print("\n[CONDITION-C] Pure Gemma text-only. Decoder checkpoint not required.")
 
-    # 5. Load Tokenizer & Models
+    # 5. Load Tokenizer & Models (PREREG_S7_D_v3_8 §5-3(b) FIX-1)
     print("\nLoading tokenizer...")
     tokenizer = load_tokenizer()
-    eos_set = {tokenizer.eos_token_id}
-    for special in ["<turn|>", "<end_of_turn>"]:
-        t_ids = tokenizer.encode(special, add_special_tokens=False)
-        if t_ids:
-            eos_set.update(t_ids)
+
+    if tokenizer.eos_token_id != 1:
+        print(f"[FATAL] tokenizer.eos_token_id ({tokenizer.eos_token_id}) != 1", file=sys.stderr)
+        sys.exit(1)
+
+    turn_id = tokenizer.convert_tokens_to_ids("<turn|>")
+    if turn_id != 106:
+        print(f"[FATAL] tokenizer.convert_tokens_to_ids('<turn|>') ({turn_id}) != 106", file=sys.stderr)
+        sys.exit(1)
+
+    eos_set = set(EOS_IDS)
+    eos_pieces = {tid: tokenizer.convert_ids_to_tokens([tid])[0] for tid in sorted(eos_set)}
+    print(f"[EOS-SET] Strict stop tokens locked to {EOS_IDS} -> {eos_pieces}", flush=True)
 
     print("Loading Pure Gemma Base...")
     pure_gemma, lm_model, lm_head, embed_tokens = load_pure_gemma_model(device_str)
@@ -596,11 +636,14 @@ def main():
                 nll_val = compute_nll_f32_chunked(h_states[:, :-1, :], shift_targets, lm_head, chunk_size=256)
 
         prompt_sha = hashlib.sha256((prob_text).encode("utf-8")).hexdigest()
+        stop_token_id = gen_token_ids[-1] if gen_token_ids and gen_token_ids[-1] in eos_set else None
         record_entry = {
             "id": qid,
             "prompt_sha": prompt_sha,
             "generation": gen_text,
             "tokens": len(gen_token_ids),
+            "stop_id": stop_token_id,
+            "gen_ids": [int(t) for t in gen_token_ids],
             "time": round(elapsed_item, 3),
         }
         if nll_val is not None:
@@ -617,7 +660,7 @@ def main():
     sha_out = compute_sha256(out_path)
     total_time = time.time() - t0_all
     print("\n" + "=" * 80)
-    print(f"Generation Complete for Condition {cond}")
+    print(f"Generation Complete for Condition {cond} (v3.6d)")
     print(f"Output: {out_path}")
     print(f"Items Generated: {generated_count} (Total in file: {len(done_ids) + generated_count})")
     print(f"SHA-256: {sha_out}")
```

---

## 3. 로컬 합성 단위 시험 결과 (Part C-1)

로컬 격리 환경(`scratch/test_v36d_synthetic.py`)에서 6대 예외 및 방어 케이스를 시험하였으며, 전 항목 설계대로 거부 및 조기 차단이 실행되었습니다.

| 케이스 번호 | 시험 항목 | 시험 내용 | 관측 결과 |
| :--- | :--- | :--- | :--- |
| **Case 1** | 출력 경로 가드 | `results_s7/p6_gen_A.jsonl` 및 `scratch/test.jsonl` 전달 | `must be inside '<BASE_S5_DIR>/results_s7/p6r'` 출력 및 exit 1 차단 |
| **Case 2** | 표준 명칭 가드 | `--items`에 `p6_gen_A.jsonl`, `--smoke`에 `p6_gen_C.jsonl` 지정 | `cannot be used with --items or --smoke` 출력 및 exit 1 차단 |
| **Case 3** | 순서 파일 SHA 가드 | `--order_sha` 누락 혹은 위조 해시 전달 | `Item order SHA mismatch` 출력 및 exit 1 차단 |
| **Case 4** | 기동 정지 토큰 검정 | `eos_token_id != 1` 혹은 `turn_id != 106` 모의 | `tokenizer.eos_token_id != 1` 출력 및 exit 1 즉시 기동 차단 |
| **Case 5** | 순서 파일 접근 0건 | 순서 파일이 없는 모의 환경에서 `--items` 기동 | `p6_item_order.json` 접근 시도 없이 예외 없이 파싱 마침 |
| **Case 6** | `stop_id` 단위 시험 | 5개 토큰열 모의 ([..., 106], [..., 1], 512열 미종료, 512번째 106, 512번째 1) | 각각 106, 1, None, 106, 1 정밀 반환 |

---

## 4. 원격 배포 및 구문 검정 (Part C-2)

- **SFTP 배포 시각**: `2026-10-07 14:20:19 KST` (`shared/ssh_log.csv:176`, 반환 코드 0)
- **원격 파일 위치**: `/home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36d.py`
- **원격 파일 크기**: 30,633 바이트 (로컬 파일과 일치)
- **원격 SHA-256**: `c032d8a836ae6d30b8d956cd5ad599ef37a6683c1fc62b6c6a75da68b7b72183` (로컬과 일치)
- **원격 컴파일 검정**: `/home/shoon/system1_5/.venv/bin/python -m py_compile scripts/s7_eval_gate_p6_v36d.py` (반환 코드 0)

---

## 5. 원격 서버 CPU 스모크 진단 (Part C-3)

원격 서버 환경에서 CPU 모드로 2문항 스모크 생성을 집행하여 입출력 규격 및 로깅 준수를 진단하였습니다.

### 5.1 조건 C 스모크 진단 (Pure Gemma)
- **실행 명령**:
  ```bash
  /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36d.py \
    --cond C --smoke --max_new_tokens 16 \
    --out /home/shoon/cts_e1_s5/results_s7/p6r/smoke/smoke_C.jsonl --device cpu
  ```
- **실행 결과**: 반환 코드 0 (소요 시간 27.75초)
- **표준 출력 주요 로그 인용**:
  - `[ITEM-ORDER] /home/shoon/cts_e1_s5/configs/s7_train_993.json SHA 3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c n=2`
  - `[EOS-SET] Strict stop tokens locked to (1, 106) -> {1: '<eos>', 106: '<turn|>'}`
  - `[CANONICAL AUDIT PASS] Condition C pure text prompt verified.`
- **산출물 무결성**:
  - 경로: `/home/shoon/cts_e1_s5/results_s7/p6r/smoke/smoke_C.jsonl`
  - SHA-256: `ba1f237ac082e3aca7d49b632a32e0a703c2cf749430821a6a43663414ce0dfb`
  - 문항 1 (Q#14): `keys=['gen_ids', 'generation', 'id', 'prompt_sha', 'stop_id', 'time', 'tokens']`, `stop_id: None`, `len(gen_ids): 16`, `tokens: 16`
  - 문항 2 (Q#29): `keys=['gen_ids', 'generation', 'id', 'prompt_sha', 'stop_id', 'time', 'tokens']`, `stop_id: None`, `len(gen_ids): 16`, `tokens: 16`

### 5.2 조건 A 스모크 진단 (PrefixProjector + Pure Gemma)
- **실행 명령**:
  ```bash
  /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36d.py \
    --cond A --smoke --max_new_tokens 16 \
    --out /home/shoon/cts_e1_s5/results_s7/p6r/smoke/smoke_A.jsonl --device cpu
  ```
- **실행 결과**: 반환 코드 0 (소요 시간 29.86초)
- **표준 출력 주요 로그 인용**:
  - `[ITEM-ORDER] /home/shoon/cts_e1_s5/configs/s7_train_993.json SHA 3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c n=2`
  - `[GUARD VERIFIED] [PASS] [P6-GATE] Verified A checkpoint: '/home/shoon/cts_e1_s5/results_s7/decoder_A_v36.pt' (v3.6)`
  - `[EOS-SET] Strict stop tokens locked to (1, 106) -> {1: '<eos>', 106: '<turn|>'}`
  - `[CANONICAL AUDIT PASS] §3(b) slots [4..67] and token 68 == 108 strictly verified.`
- **산출물 무결성**:
  - 경로: `/home/shoon/cts_e1_s5/results_s7/p6r/smoke/smoke_A.jsonl`
  - SHA-256: `956df4ddda28146fc49880e5be4198f7fe984677e41b2a92f910a9ca79086007`
  - 문항 1 (Q#14): `keys=['gen_ids', 'generation', 'id', 'prompt_sha', 'stop_id', 'time', 'tokens']`, `stop_id: None`, `len(gen_ids): 16`, `tokens: 16`
  - 문항 2 (Q#29): `keys=['gen_ids', 'generation', 'id', 'prompt_sha', 'stop_id', 'time', 'tokens']`, `stop_id: None`, `len(gen_ids): 16`, `tokens: 16`

---

## 6. 재현성 점검 문항 목록 추출 및 생성 (Part C-4)

### 6.1 점검 원리 및 이론적 근거
기존 P6 원본 산출물 중 생성 토큰 수가 최대치인 512개(`tokens == 512`)에 도달한 문항은, 생성 과정에서 오염 정지 집합에 속한 어떠한 토큰도 만나지 않고 한계 길이에 도달한 문항들입니다.
따라서 정지 집합에서 6개 오염 조각을 제거한 `v36d`로 동일 조건(GPU 0 결정론적 환경)에서 재실행하더라도, 생성 토큰 열은 기존 P6 생성물과 단 하나의 토큰도 틀리지 않고 완전히 일치해야 합니다.

### 6.2 조건별 512토큰 문항 집계 및 상위 5문항 추출

| 조건 | 전체 512토큰 문항 수 | 선정 상위 5문항 (q_idx 순) | 개별 생성물 SHA-256 (64자 전체) |
| :--- | :--- | :--- | :--- |
| **A** | **243건** / 600 | 1: Q#0035 (q_idx 1)<br>2: Q#0102 (q_idx 11)<br>3: Q#0111 (q_idx 12)<br>4: Q#0193 (q_idx 22)<br>5: Q#0238 (q_idx 25) | `2d4a033c8094002b3857c4fc089c2c0960f1dbfa1c5ea2f0e95e96c0298d0665`<br>`5b8f86c36eded3d9a038e25ff0d64d93dc93fd6fe852b7c1d70df6fe2bd9b800`<br>`39ab789edfed526b7ba03495824e1757ba4c0abd942a8fea60d0542a8b7feac1`<br>`4749f1368998f75c7b27a66e4da3acff41fcc0d3812e5f72e3c0d3a9e0cbbfa2`<br>`a65e4c6c7f1f60e2769c0cd6a411c32940ec0dd1273192eaa914119ce36841c5` |
| **C** | **375건** / 600 | 1: Q#0035 (q_idx 1)<br>2: Q#0038 (q_idx 2)<br>3: Q#0043 (q_idx 3)<br>4: Q#0071 (q_idx 6)<br>5: Q#0088 (q_idx 7) | `e6d0ce88db7a24ff5b5105cfd8118d4b6ea52e62265b92cbba7afbe4ae277d92`<br>`799dbfc5dae27783ef38814a173fbf4cd252605ba332cb161ad45d1637ee7dc9`<br>`68647492415d5b0b8021ba0b48cd347967ee04d394b5af63b2bae3da2638043e`<br>`c21ec2e36bb0be97a8a71336649dfe42b9ec776d4c0eabdc7c70ae07858cdcb6`<br>`c50e295c3383c0aef0b914470f40f30e41fa051144783f9aeb0e5bec2d662cd5` |
| **E1** | **135건** / 600 | 1: Q#0093 (q_idx 9)<br>2: Q#0127 (q_idx 15)<br>3: Q#0193 (q_idx 22)<br>4: Q#0246 (q_idx 26)<br>5: Q#0407 (q_idx 43) | `542d1fac9dce1f4cbcc81815667181b2f9d1c500c0e0d421af52d1fe4f77c041`<br>`b006a482e0abe4b47be5cf10642cb5f6931b2f9a19f81db5541c5685a6a5c344`<br>`59b5a1227b44bb87853ee5438cb32c1fce1bc2b375216872623e4fbf3608436e`<br>`353aed785f7990e905f35dd6bd97385682f67388ad170396ed1eb729380e826a`<br>`aaa3dc41a727d14acedc3078e84567da3059d9af6c7d04a86c22e9d294b5e713` |

### 6.3 재현 점검용 설정 파일 지문
- `configs/p6r_repro_A.json`:
  - 내용: `{"indices": [35, 102, 111, 193, 238]}`
  - SHA-256: `9ba7960492ed816bd383b86483f11adac8789ac4044cc5ab115fad05eec21e1f`
- `configs/p6r_repro_C.json`:
  - 내용: `{"indices": [35, 38, 43, 71, 88]}`
  - SHA-256: `6e15cecda263d252598c6a35f946a42ae114f272abfe83cd2e34b9bac667b9f3`
- `configs/p6r_repro_E1.json`:
  - 내용: `{"indices": [93, 127, 193, 246, 407]}`
  - SHA-256: `2d6cac69ad63bb67e93b7dab22bb00bf807df9c1d837e18057ad5c5e62bc462a`

---

## 7. 문항 순서 파일 감사 및 P6 실행 명령 추출 (Part C-5 & C-6)

### 7.1 순서 파일 해시 원장 대조 (Part C-5)
- **원격 고정 순서 파일**: `/home/shoon/cts_e1_s5/results_s7/p6_item_order.json`
- **실측 SHA-256**: `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c`
- **개별 생성 로그 대조 결과**:
  - `raw_s7/p6_gen_A.log`: `[ITEM-ORDER] Fixed item order (600 items) saved to ... (SHA: 6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c)`
  - `raw_s7/p6_gen_B1.log`: SHA `6c76ddb7...` 대조 일치
  - `raw_s7/p6_gen_C.log`: SHA `6c76ddb7...` 대조 일치
  - `raw_s7/p6_gen_E1.log`: SHA `6c76ddb7...` 대조 일치
- **평가**: P6의 모든 조건 생성기가 동일한 고유 순서 파일(SHA: `6c76ddb7...`)을 참조하여 실행되었음을 대조 마쳤습니다.

### 7.2 P6 생성 실행 명령 추출 (Part C-6)
`scripts/p6_chain_after_p5_v2.sh:171-212` 및 `raw_s7/p6_chain.log:365-384`로부터 추출된 P6 각 조건별 실제 파이썬 기동 명령은 다음과 같습니다.

1. **조건 C**:
   ```bash
   /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36b.py --cond C
   ```
2. **조건 A**:
   ```bash
   /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36b.py --cond A
   ```
3. **조건 B (shift=1, B1)**:
   ```bash
   /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36b.py --cond B --shift 1
   ```
4. **조건 E1**:
   ```bash
   /home/shoon/system1_5/.venv/bin/python /home/shoon/cts_e1_s5/scripts/s7_eval_gate_p6_v36b.py --cond E1
   ```

---

## 8. 종합 결론 및 차기 지침 준비

1. **v36d 규격 완결**: 정지 집합 `{1, 106}` 격리, 경로 격리 가드, 레코드 필드(`stop_id`, `gen_ids`) 확장을 담은 `scripts/s7_eval_gate_p6_v36d.py`가 배포되어 CPU 스모크 및 단위 점검을 이상 없이 완결하였습니다.
2. **Errata 보고 마침**: REPORT-0065의 인용 행 불일치 원인을 다중 SSH 실행 간 덮어쓰기 기전으로 규명하고, 물리적 타임스탬프(14:06:17 중단) 증거를 담은 `shared/reports/REPORT-0065-ERRATA.md` 발행을 마쳤습니다.
3. **재현성 점검 준비 완비**: `tokens == 512` 기반의 재현성 대조 문항 3종(`configs/p6r_repro_{A,C,E1}.json`) 및 개별 64자리 SHA 해시 원장을 확보하였습니다.
4. **지침 대기**: GPU 생성 0건 규약을 엄격히 준수한 상태에서, 재평가 집행 지침(TASK-0067) 수신 대기 상태로 이행합니다.

