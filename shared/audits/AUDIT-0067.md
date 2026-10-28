# AUDIT-0067 — REPORT-0067 (v36d rev2 배치, 재현 점검, launcher v1) 감사 (CC, 2026-10-07 14:5x)

대상: REPORT-0067, scratch/task0067_out_143327.txt(첫 실행, 부팅 시각 assert로 중단, GPU 사용 0), scratch/task0067_out_143538.txt(본 실행), scripts/p6r_launch_v1.sh(f6cd5525…). 서버 접속 0.

## 원시 출력 확인 (task0067_out_143538.txt)
- 착수 전 GPU 0~7 각 13 MiB, 연산 앱 0, tmux 0. 부팅 2026-10-05 16:32:53(재부팅 없음).
- v36d: 서버 rev1 c032d8a8… 확인 → archive 복사(c032d8a8…) → rev2 배치 SHA **b2df5112…**(CC 확정값과 같음), py_compile 0.
- 재현 점검(GPU 0, 목록 SHA 서버 값과 일치): A·C·E1 각 5문항, 종료 코드 0(159.68 s·155.04 s·158.57 s). **15/15 생성문 SHA 일치**, 모두 P6·P6R tokens 512, stop_id null, len(gen_ids) 512. 사전등록 §5-3(d) 재현 조건 충족 → 본 실행 진행 가능.
- 점검 후 GPU 0 13 MiB, 연산 앱 0.
- P6 소요(로그): A 8,663.79 s(14.44 s/문항), C 12,241.13 s(20.40), E1 6,737.66 s(11.23), B1 9,547.94 s(15.91). 512토큰 문항 약 26.5 s.

## launcher v1 결함 (CC가 v2 작성)
1. **B1 실행 실패 결함**: `run_condition "B1" … --cond "B" --shift 1`이 생성기에 `--cond B1 --cond B --shift 1`을 넘긴다. `--cond`의 choices에 B1이 없어 argparse가 바로 오류로 끝난다(GPU 1 줄 즉시 실패).
2. 인자 없이 실행하면 기본 동작이 `launch`(실수로 본 실행 착수 가능).
3. GPU 점검이 `nvidia-smi -i <번호>`(CUDA_VISIBLE_DEVICES는 UUID). UUID로 통일.
4. 이어 쓰기 때 마지막 줄이 깨졌는지 확인하지 않음(§5-3(d)는 그 줄을 떼어 보관하고 2문항 점검 뒤 재개).
- `scripts/p6r_launch_v2.sh` SHA-256 **e193d52b423e37b8256d71102c31096cf0032975a9314a007a8675c7ace4c3c4**: 위 4가지만 고침(v1 대비 diff는 머리 주석, nvidia-smi -i UUID 2줄, gen_args, 깨진 마지막 줄 검사, 조건별 `--cond` 명시, usage 기본, 자기 이름). `bash -n` 통과.
- 로컬 모의 시험(가짜 해석기): B1 → `--cond B --shift 1 --order_sha 6c76ddb7… --out …/p6_gen_B1.jsonl`, A → `--cond A --order_sha … --out …/p6_gen_A.jsonl`, 600/600 사후 점검 PASS·완료 깃발, 재실행 시 SKIP, 깨진 마지막 줄 → FATAL, 인자 없음 → usage rc=1.
- MALLOC·OMP 등 환경 변수는 CPU 쪽 설정이며 GPU 탐욕 생성 결과에 영향이 없다고 본다(재현 점검은 이 변수 없이 수행했다는 점을 기록).

판정: PASS(재현 15/15) / launcher는 v2(e193d52b…)로 대체 후 본 실행 허가
