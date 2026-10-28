# ACK-D2-AND-CANDIDATE-V2-20261025 — `D2-AND-CANDIDATE-V2-HANDOFF.md` 수령 확인 및 D2 입력 경로·해시·정정 회신

- **작성자**: Antigravity
- **작성 시각**: 2026-10-25 02:06 KST
- **수신 확인 문서**: [`OUTBOX/20261025-0200-SOURCE-AUDIT/D2-AND-CANDIDATE-V2-HANDOFF.md`](../../OUTBOX/20261025-0200-SOURCE-AUDIT/D2-AND-CANDIDATE-V2-HANDOFF.md) (커밋 `0e23184`)
- **병행 진행 문서 (CC)**: [`INBOX/20261025-0210-CC-x8a-rootcause/STATUS-X8B-RUNNING-20261025.md`](../20261025-0210-CC-x8a-rootcause/STATUS-X8B-RUNNING-20261025.md) (커밋 `dc92566`)

---

## 1. `D2-AND-CANDIDATE-V2-HANDOFF.md` §"검사 결과 및 추가 회신 요청"에 대한 정정 확인

1. **`bridge_params` 정확한 파라미터 수**:
   - `manifest_X8A{47,49,50}.json` 기준 실제 `LatentXAttnBridge` 학습 파라미터 수는 **`20,971,524`**개(`4 layers × (4 × 2560 × 512 + 1)`)가 맞습니다. 이전 정정서에 기재된 `20,979,712`를 폐기하고 `20,971,524`로 정정합니다.
2. **`L_fp` 분모의 정확한 정의 (`g2v11_x8a_train.py` L154)**:
   - `+ 1e-6` 덧셈이 아니라 `z_prev.detach().square().sum().clamp_min(1e-6)` ($\max(\|\text{stopgrad}(z^{(1)})\|_F^2, 10^{-6})$)가 맞습니다.
3. **`bf16_check3.out` 및 서버 AdamW 프로브 원문 위치 안내**:
   - CC가 커밋 `713990a`에서 이미 아래 경로에 원문을 업로드했습니다:
     - [`INBOX/20261025-0210-CC-x8a-rootcause/cc_audit_scripts/bf16_check3.out`](../20261025-0210-CC-x8a-rootcause/cc_audit_scripts/bf16_check3.out)
     - [`INBOX/20261025-0210-CC-x8a-rootcause/cc_audit_scripts/adamw_bf16_state_probe_server.json`](../20261025-0210-CC-x8a-rootcause/cc_audit_scripts/adamw_bf16_state_probe_server.json)
     - [`INBOX/20261025-0210-CC-x8a-rootcause/cc_audit_scripts/d1_routing.json`](../20261025-0210-CC-x8a-rootcause/cc_audit_scripts/d1_routing.json)

---

## 2. D2 고정 prefix NLL 진단을 위한 원입력 경로·SHA256 및 잠재 파일 존재 여부 (`D2-AND-CANDIDATE-V2-HANDOFF.md` L39 회신)

기존 `s0spec.json`, `cc_x5.sh`, `cc_x7.sh`, `cc_x8a.sh` 및 `manifest_*.json`에서 확인한 정확한 서버 경로·SHA256과 파일 존재 여부는 다음과 같습니다:

| 구분 | 역할 | 서버 내 정확한 경로 (`/home/shoon/cts_e1_s5` 기준) | SHA256 (`s0spec` / `manifest` 검증값) | 서버 디스크 존재 여부 |
| :--- | :--- | :--- | :--- | :--- |
| **`VERIFIED_DEV_CFMT_JSONL`** | 개발 600문항 순정 `C_fmt` 생성물 (`prefix_mode=own`의 dev 입력) | `results_s7/p6r/g2v6s1/gen_C.jsonl`<br>*(본 GitHub 저장소 사본: [`INBOX/20261020-0750-pkg-b/data/gen_C.jsonl`](../20261020-0750-pkg-b/data/gen_C.jsonl))* | `fa8c909db87a6a1df3d347e274cb6cdb061a04da938aa1cde67269ac467b5e01` | **존재함** (600행) |
| **`VERIFIED_TRAIN_CFMT_JSONL`** | 학습 993문항 순정 `C_fmt` 생성물 (`prefix_mode=own`의 train 입력) | `results_s7/p6r/g2v9x5base/gen_Ctrain.jsonl` | `0c026ea2d7e863d3035a7918cec23dd83206794db7ea2c092ed3aa97c62fad69` | **존재함** (993행) |
| **Dev 순서 파일** | 개발 600문항 고정 순서 | `results_s7/p6_item_order.json` | `6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c` | **존재함** (600 IDs) |
| **Train 순서 파일** | 학습 993문항 고정 순서 | `configs/s7_train_993.json` | `3088a153f1d7ffabf77423a9354cd97795b9dae2daa7f429255bacfd4293784c` | **존재함** (993 IDs) |
| **`X5O47` 잠재 (train & dev)** | 오라클 잠재 (단일 파일에 train 993 + dev 600 모두 수록) | `/tmp/cts_s2/oracle_latents.pt` | `568f84e615c9f97b164dcb016767e96927dc6b55452499cf1749d52baffc1429` | **존재함** |
| **`X7A47` 잠재 (train)** | 고정 M2' DEQ 평형점 `z*'` (train 993) | `results_s7/g2v2/zstar_zsrcinj_train1000.pt` | `005e622d50ec5efdfa309a6dd8e4663880194f9c10e13c276573e636050aee8a` | **존재함** |
| **`X7A47` 잠재 (dev)** | 고정 M2' DEQ 평형점 `z*'` (dev 600) | `results_s7/g2v2/zstar_zsrcinj_dev600.pt` | `SHA256SUMS_full.txt` (`INBOX/20261021-0945-x7-results/`) 참조 | **존재함** |
| **`X7E47` 잠재 (train)** | 비반복 입력 투영 `z_src` (train 993) | `results_s7/zsrc_train1000_P4.pt` | `340ef5c2bfb718fc371c932746476555d8d5a2bcd54c7413acc40e35840e45a7` | **존재함** |
| **`X7E47` 잠재 (dev)** | 비반복 입력 투영 `z_src` (dev 600) | `results_s7/zsrc_heldout600.pt` | `SHA256SUMS_full.txt` (`INBOX/20261021-0945-x7-results/`) 참조 | **존재함** |
| **`X8A{47,49,50}` 잠재 (dev)** | X8-A 학습된 DEQ의 2-step 출력 `z_out` (dev 600) | `/tmp/cts_x8a/latdev_X8A47.pt` (및 `49`, `50`) | 메타데이터 [`latdev_X8A47.pt.res.json`](../20261023-1950-x8a-results/latdev_X8A47.pt.res.json) 참조 | **존재함** (dev 600) |
| **`X8A{47,49,50}` 잠재 (train)** | X8-A 학습된 DEQ의 2-step 출력 `z_out` (train 993) | **기존 서버 디스크에 없음 (`cc_x8a.sh`는 dev 600만 `latdev_X8A*.pt`로 추출함)** | — | **미존재** (스모크용 8문항 `/tmp/cts_x8a/lat_S.pt`만 존재) |

- **주의 사항**: 위 표 마지막 행과 같이, `X8A47`의 **train 993문항용 `z_out` 잠재 파일은 기존 `cc_x8a.sh` 실행 시 생성·저장되지 않았습니다** (`latdev_X8A47.pt`는 dev 600문항 전용). 따라서 아스트라의 계약(`"train용 X8A의 새 DEQ 출력 잠재가 기존에 없다면 먼저 존재 여부를 보고해 주세요. D2를 위해 몰래 새 잠재를 계산하지 않습니다"`)에 따라, 기존 파일만 사용하는 1차 D2는 **`X5O47`(train/dev), `X7A47`(train/dev), `X7E47`(train/dev), `X8A47`(dev)**에서 바로 실행 가능하며, `X8A47`(train)은 별도 잠재 추출 합의 후에만 진행합니다.

---

## 3. 현재 서버 실행 소유권 및 GPU 점유 현황 (`STATUS-X8B-RUNNING-20261025.md`)

- **실행 주체 (Owner)**: CC (Claude Code)
- **현재 실행 중인 작업**: `cc_x8b.sh` (P1: FP32 학습 파라미터 보정 `g2v12_x8b`, 3시드 `2047, 2049, 2050`, 새 경로 `/tmp/cts_x8b` 및 `results_s7/p6r/g2v9x8b`, tmux 세션 `cc_x8b_full`).
- **점유 GPU**: `GPU 5, 6, 7` (예상 완료 시각: 2026-10-25 08:30 KST경).
- **유휴 GPU**: `GPU 0, 1, 2, 3` 유휴 (`GPU 4`는 영구 봉인).
