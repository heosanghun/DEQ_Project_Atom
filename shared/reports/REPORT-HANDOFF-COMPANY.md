16c7cd60f4ab276591d3bcea57f5b158433383cbcecf8d1a65eb9a24989eb2d5
# REPORT-HANDOFF-COMPANY: 회사 PC 수령 및 P3 실측 검증 보고서

- **보고서 ID**: REPORT-HANDOFF-COMPANY
- **작성 일시**: 2026-10-04 16:45 KST
- **실행자**: 회사 PC Antigravity (원장 #410 인수인계 3단계)
- **지침 정본**: `D:\AI\cts\shared\HANDOFF_cts260930_to_cts.md`
- **원격 서버**: `shoon@<GPU_SERVER_IP>:<SSH_PORT>` (`/home/shoon/cts_e1_s5`)
- **진실의 출처**: `shared/prereg/PREREG_S7_DEQ_v1.2.md:90-95`, `results_s7/s7deq_intervene_P3.json`
- **원칙**: 판정 단어 배제(0건), 수치 전재, 비트/바이트 해시 명시, 오기 인용 배제(사용 0)

---

## 1. 서버 §0 상태 실측 계측치 (2026-10-04 16:42:28 실측)

### 1.1. 시스템 메모리 및 스왑
| 자원 구분 | 전체 용량 | 사용량 | 여유량 | 가용량 |
|:---|:---|:---|:---|:---|
| 물리 메모리 (RAM) | 251 GiB | 25 GiB | 1.4 GiB | 224 GiB |
| 스왑 메모리 (Swap) | 8.0 GiB | 0 B | 8.0 GiB | 8.0 GiB |

### 1.2. GPU 0~7 실측 상태 (NVIDIA RTX A6000 48GB x 8)
| GPU 번호 | 모델명 | 사용 VRAM | 전체 VRAM | GPU 사용률 | 활성 프로세스 수 |
|:---:|:---|:---:|:---:|:---:|:---:|
| GPU 0 | NVIDIA RTX A6000 | 24 MiB | 49,152 MiB | 0% | 0 |
| GPU 1 | NVIDIA RTX A6000 | 13 MiB | 49,152 MiB | 0% | 0 |
| GPU 2 | NVIDIA RTX A6000 | 13 MiB | 49,152 MiB | 0% | 0 |
| GPU 3 | NVIDIA RTX A6000 | 13 MiB | 49,152 MiB | 0% | 0 |
| GPU 4 | NVIDIA RTX A6000 | 13 MiB | 49,152 MiB | 0% | 0 |
| GPU 5 | NVIDIA RTX A6000 | 13 MiB | 49,152 MiB | 0% | 0 |
| GPU 6 | NVIDIA RTX A6000 | 13 MiB | 49,152 MiB | 0% | 0 |
| GPU 7 | NVIDIA RTX A6000 | 13 MiB | 49,152 MiB | 0% | 0 |

### 1.3. 세션 및 실행 프로세스 점검
- `tmux ls`: `no server running on /tmp/tmux-1001/default` (동작 중인 tmux 세션 0건)
- 백그라운드 연산 프로세스: 0건 (Decoder E 학습 10-02 12:49 KST 완주 후 프로세스 종료 상태 유지)
- 원격 인수인계 디렉터리: `/home/shoon/cts_e1_s5/shared_laptop_20261003` (4,096 bytes, 권한 700)

---

## 2. P3 실측치 및 게이트 대조 검증표 (PREREG_S7_DEQ_v1.2 §4 대조)

### 2.1. 주 게이트 3종 정량 대조
| 평가 항목 | 지표 정의 | 사전등록 기준 (v1.2 §4) | 실패 판정선 | 실측값 (`s7deq_intervene_P3.json`) | 충족 여부 |
|:---|:---|:---|:---|:---|:---|
| **주 게이트 1**<br>(초고정밀 수렴) | 미수렴 54문항의<br>참 잔차 $r < 1.0 \times 10^{-3}$ 달성률 | $\ge 80.0\%$ (44/54 이상) | — | **52 / 54 (96.30%)** | 충족 |
| **주 게이트 2**<br>(엄격 DEQ 수렴) | 미수렴 54문항의<br>참 잔차 $r < 5.0 \times 10^{-3}$ 달성률 | $\ge 95.0\%$ (52/54 이상) | $< 50.0\%$ (27문항 미만) | **53 / 54 (98.15%)** | 충족 |
| **주 게이트 3**<br>(대조군 불변성) | 대조 20문항의 신·구 고정점<br>방향 코사인 유사도 | $\min_i \cos \ge 0.9990$ | $\cos < 0.9950$ 발생 시 | $\min \cos = \mathbf{0.93263088}$<br>(Q#2060) | **불충족<br>(실패선 하회 발생)** |

### 2.2. 게이트 1 및 2 기준치 미달 문항 명세 (미수렴 54문항 집합)
| 문항 번호 | 문항 키 | $r^*$ 실측값 | $r_{1000}^{\text{eval}}$ 실측값 | 게이트 1 ($r < 10^{-3}$) | 게이트 2 ($r < 5 \times 10^{-3}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Q#0424 | `q0424_a0.25` | 2.87565446e-03 | 3.76458869e-03 | 미달 | 충족 |
| Q#0865 | `q0865_a0.25` | 7.19202985e-03 | 1.45065461e-02 | 미달 | 미달 |

- 게이트 1 미달 문항: 2건 (Q#0424, Q#0865)
- 게이트 2 미달 문항: 1건 (Q#0865)

### 2.3. 게이트 3 대조군 20문항 코사인 분포 실측치
| 구간 | 사전등록 기준상 분류 | 해당 문항 수 | 비율 | 해당 문항 목록 |
|:---|:---|:---:|:---:|:---|
| $\cos \ge 0.9990$ | 게이트 3 기준선 충족 | 10 / 20 | 50.0% | Q#0249, Q#0395, Q#0822, Q#3634, Q#3656, Q#3802, Q#3953, Q#3976, Q#4404, Q#4722 |
| $0.9950 \le \cos < 0.9990$ | 기준선 미달이나 실패선 이상 | 2 / 20 | 10.0% | Q#0942 (0.99800866), Q#4208 (0.99522162) |
| $\cos < 0.9950$ | 실패 판정선 하회 발생 | 8 / 20 | 40.0% | Q#0622, Q#0954, Q#1430, Q#2060, Q#2884, Q#2933, Q#3184, Q#4773 |

### 2.4. 대조군 20문항 전수 실측 데이터
| 문항 번호 | 문항 키 | $\cos(z_{\text{new}}^*, z_{\text{old}}^*)$ | 참 잔차 $r^*$ | 게이트 3 판정선 대조 |
|:---:|:---:|:---:|:---:|:---|
| Q#0249 | `q0249_a0.25` | 0.99984231 | 5.98370867e-04 | $\ge 0.9990$ 충족 |
| Q#0395 | `q0395_a0.25` | 0.99998729 | 1.64700359e-04 | $\ge 0.9990$ 충족 |
| Q#0622 | `q0622_a0.25` | 0.99435905 | 2.23119981e-07 | $< 0.9950$ 실패선 하회 |
| Q#0822 | `q0822_a0.25` | 0.99999996 | 1.15448832e-05 | $\ge 0.9990$ 충족 |
| Q#0942 | `q0942_a0.25` | 0.99800866 | 9.82697096e-04 | $[0.9950, 0.9990)$ 미달 |
| Q#0954 | `q0954_a0.25` | 0.98462366 | 7.06554807e-05 | $< 0.9950$ 실패선 하회 |
| Q#1430 | `q1430_a0.25` | 0.95692793 | 5.66122194e-04 | $< 0.9950$ 실패선 하회 |
| Q#2060 | `q2060_a0.25` | 0.93263088 | 9.25329531e-05 | $< 0.9950$ 실패선 하회 |
| Q#2884 | `q2884_a0.25` | 0.94523535 | 4.53745867e-05 | $< 0.9950$ 실패선 하회 |
| Q#2933 | `q2933_a0.25` | 0.98098183 | 5.63339457e-05 | $< 0.9950$ 실패선 하회 |
| Q#3184 | `q3184_a0.25` | 0.99058950 | 9.25879434e-06 | $< 0.9950$ 실패선 하회 |
| Q#3634 | `q3634_a0.25` | 0.99999981 | 3.59651692e-05 | $\ge 0.9990$ 충족 |
| Q#3656 | `q3656_a0.25` | 0.99999999 | 7.80466449e-06 | $\ge 0.9990$ 충족 |
| Q#3802 | `q3802_a0.25` | 0.99999983 | 2.41475035e-05 | $\ge 0.9990$ 충족 |
| Q#3953 | `q3953_a0.25` | 0.99999988 | 2.42838000e-05 | $\ge 0.9990$ 충족 |
| Q#3976 | `q3976_a0.25` | 1.00000000 | 2.21978020e-07 | $\ge 0.9990$ 충족 |
| Q#4208 | `q4208_a0.25` | 0.99522162 | 4.38008629e-07 | $[0.9950, 0.9990)$ 미달 |
| Q#4404 | `q4404_a0.25` | 0.99999844 | 9.61411874e-05 | $\ge 0.9990$ 충족 |
| Q#4722 | `q4722_a0.25` | 0.99999347 | 1.33600388e-04 | $\ge 0.9990$ 충족 |
| Q#4773 | `q4773_a0.25` | 0.98954969 | 3.47232358e-04 | $< 0.9950$ 실패선 하회 |

### 2.5. 노트북 `HANDOVER_TO_CTS_20261004.md §7` 오기 정정 및 배제 기록 (사용 0)
- **오기 사항 1 (미달 문항 번호)**: 노트북 §7.1 표는 게이트 1/2 미달 문항을 Q#0942, Q#3976으로 기재함. 이는 대조군(control) 문항을 미수렴(nonconv) 문항으로 혼동한 명백한 오기임.
  - 실측 진실: Q#0942와 Q#3976은 대조군 20문항 소속임. Q#3976의 코사인 유사도는 1.00000000이며, Q#0942의 코사인 유사도는 0.99800866임.
  - 미수렴 54문항의 실제 게이트 1 미달 문항은 Q#0424($r^*=2.87565446 \times 10^{-3}$), Q#0865($r^*=7.19202985 \times 10^{-3}$) 2건이며, 게이트 2 미달 문항은 Q#0865 1건임.
- **오기 사항 2 (게이트 3 기준치)**: 노트북 §7.1 표는 게이트 3 기준치를 `0.9999`로 기재함.
  - 실측 진실: `PREREG_S7_DEQ_v1.2.md:94`에 명시된 게이트 3 기준치는 `min cos >= 0.9990`이며 실패선은 `cos < 0.9950`임.
- **조치**: 노트북 §7의 오기 수치(Q#0942, Q#3976의 미수렴 게이트 미달 지정 및 기준치 0.9999)는 공식 감사 및 후속 절차에서 인용 0(`사용 0`)으로 영구 배제하며, 본 보고서 2.1~2.4에 전재된 `results_s7/s7deq_intervene_P3.json` 실측치만을 정본 수치로 인정함.

---

## 3. 수령 파일 인벤토리 및 SHA-256 목록 (`shared/from_laptop/`)

- **수령 일시**: 2026-10-04 16:44:47 KST
- **수령 프로토콜**: SFTP 배치 일괄 전송 (단일 SSH 연결, 세션 분할 0건)
- **수령 파일 총수**: 129개

| 파일 상대 경로 (`from_laptop/` 기준) | 크기 (Bytes) | SHA-256 해시 |
|:---|:---:|:---|
| `ESC-GPU1.md` | 96 | `f089fdbce7d89c7cccd4800b60d21f9d7df0674761d333ceefd40689507ebd25` |
| `EXTERNAL_AUDIT_STANDING_ORDERS.md` | 4,566 | `18f27807fd6e4c04ceff2be14967acd9c15fc5fbf64b8ebf1aac68c9a01542c3` |
| `GOAL.md` | 2,437 | `2e454ace1d33997356f5e4778efa744e57b5014c64973432935c3939f1487f6a` |
| `LESSONS.md` | 6,326 | `1410869511179d4dece8c4febd0240ec3abea31a25badcbd41b10813264a0779` |
| `RULES.md` | 2,713 | `179e1348d541dd13d29d9b7c3bd7c918834035664085b04c2a1ed29a73265619` |
| `STATE.md` | 11,052 | `a81dd141d208f739b46af9b7dbc39f6f9618909126e47ce96381ec36a07ab546` |
| `audits/AUDIT-001.md` | 3,777 | `f3b74ea3f5a3d90cd0213f6c6bebe49ea10bc5f80016125da7a9a38f2e3d6038` |
| `audits/AUDIT-003.md` | 6,979 | `e22f9c2db278b516263570459a325fb134ae0a962fcd64ccbbdc0c0d071b5a41` |
| `audits/AUDIT-003R.md` | 7,616 | `12d444fd9cc52b4300f206740849aeffe62075ccf89bc3f0473a79aa901230f4` |
| `audits/AUDIT-CTS-409.md` | 4,665 | `f58fc34751f5d7e2b5fc1f8882e5c2557c302dcddde91946ec2ba9c0aa99e2aa` |
| `audits/AUDIT-CTS-A1.md` | 3,501 | `3593eaf68568122697fddf1a6343ae0e6ef37c77d43d5d0a829ca7960432af6e` |
| `audits/AUDIT-CTS-P0.md` | 2,554 | `a8da07a2ae62034da930f8ad3c52b7e2f8aedc6fb0f9d77d637e98a02ea832df` |
| `audits/AUDIT-CTS-P1.md` | 3,670 | `78a169876b6537f938f5522ecba3cd87b203abdd139f468e27cb091a9b15ae5f` |
| `audits/AUDIT-CTS-P2.md` | 3,917 | `a2a8582cc8943f44fc05a5a667f4e3f1bac30cfb2f823f2f6fbf3ac3ca359715` |
| `audits/AUDIT-CTS-P3.md` | 4,476 | `06fe4cac57a61a8b1e2303963612bd84b11906ad813a79a9a603c36d4ea6a882` |
| `audits/AUDIT-CTS-P3v2.md` | 4,567 | `317002a20acb7f96b8c06297ece8a310577f3cb157c507c4f827b7d7be55b504` |
| `audits/AUDIT-CTS-P3v3.md` | 4,027 | `a7268c13e8dc20153ab2f4163e8c0b483a516f95af19798c96a7df00b79dffc5` |
| `audits/AUDIT-CTS-P4-RUNNING.md` | 4,551 | `8e6e23a6bffb512e6cc3519316631153cf1eb4d60e3217adba4229f497276310` |
| `audits/AUDIT-CTS-P4.md` | 3,252 | `6b17603dc101d9caa2b0a04ab88e5245664262aadc607e816185e8a13645924d` |
| `audits/AUDIT-SO-408.md` | 2,515 | `cfdcb6e7861265ec54cb621ab130c451bacbece82ec7e6ce11e85e689855f7f3` |
| `audits/archive/REPORT-001__8db1b9f9.md` | 23,707 | `6ce0b8fbee769e69cf813402580ada64f4fe256a0a863fc45c05c81a17bfa681` |
| `audits/archive/REPORT-003R__ddaf7a95.md` | 8,694 | `1ceecaaf9f8c4d46db60f4adaa7d78dcb2d0115b32d281cf9d888e43e31dcb60` |
| `audits/archive/REPORT-003__e3d7ef3f.md` | 8,915 | `a318c29196368a24e5d9b564b7f96d15ad74678527733e039081bd568808a372` |
| `audits/archive/REPORT-CTS-A1__da686863.md` | 4,967 | `0113576900001c49d0d95abd0273496f8d9083029cfc1690c1b1b4825c599512` |
| `audits/archive/REPORT-CTS-P0__9377261e.md` | 4,697 | `f092d0aa02fd70c7a98feda66ba62435f775dcbb9be89f1805d166abfe5bf49f` |
| `audits/archive/REPORT-CTS-P1__812cf32c.md` | 6,973 | `266c0de366da3c3702409f8f737dd0c2ea6539fbb784753f8c89c821643153b5` |
| `audits/archive/REPORT-CTS-P2__99914c31.md` | 7,773 | `aae46be8c9789096b235242920d2a2db067c3709c63e42cc0ea49e80b41f745a` |
| `audits/archive/REPORT-CTS-P3.v2__57f6fe36.md` | 16,061 | `41ab4689751a9a18348cdce4eae598cdff7f2ca903a3f4a9c1e7ee32f1ef15d2` |
| `audits/archive/REPORT-CTS-P3.v3__81b4fb51.md` | 39,779 | `7b124251e3be327f4b29ad458d16c87bc27578152d2ca033d13c790bdb416fdc` |
| `audits/archive/REPORT-CTS-P3__93d0d349.md` | 13,638 | `315f7859eb83cf5afda54b6f20d1b4073d6ecbac72329084a53885fd99189a8a` |
| `audits/archive/REPORT-CTS-P4.RUNNING__3a768f9a.md` | 8,726 | `0a190d1bf4b67057ad87abf4f13e07ab18cb6bee24bdce530949264204758670` |
| `audits/archive/REPORT-CTS-P4__8fb73986.md` | 9,609 | `0166c0b814de906ad9d5f4aab2a2187f48e33bbdcb8c04d4c19b67de77ef99d2` |
| `reports/.seen_REPORT-_.md` | 779 | `603dd6cdb495350a52fa8ee8f4cb4335cb3616c1b65ea2371c4e4c555c0682d3` |
| `reports/HANDOVER_TO_CTS_20261004.md` | 19,962 | `7c3dfef1c7274900c9e5e80cbc2963374c6d1dd6944c7dab4ed55f9829362ddf` |
| `reports/REPORT-001.md` | 23,707 | `6ce0b8fbee769e69cf813402580ada64f4fe256a0a863fc45c05c81a17bfa681` |
| `reports/REPORT-003.md` | 8,915 | `a318c29196368a24e5d9b564b7f96d15ad74678527733e039081bd568808a372` |
| `reports/REPORT-003R.RUNNING.done.bak` | 633 | `49134493ea9dc3b09bcea10d369203e1140774030ccd1f68c8f94ff26e13acbb` |
| `reports/REPORT-003R.md` | 8,694 | `1ceecaaf9f8c4d46db60f4adaa7d78dcb2d0115b32d281cf9d888e43e31dcb60` |
| `reports/REPORT-CTS-409.md` | 6,062 | `3658bc11e42c3ff7a59aa3a03367f8a98484351870685c9283d91009256dcc40` |
| `reports/REPORT-CTS-A1.md` | 4,967 | `0113576900001c49d0d95abd0273496f8d9083029cfc1690c1b1b4825c599512` |
| `reports/REPORT-CTS-P0.md` | 4,697 | `f092d0aa02fd70c7a98feda66ba62435f775dcbb9be89f1805d166abfe5bf49f` |
| `reports/REPORT-CTS-P1.md` | 6,973 | `266c0de366da3c3702409f8f737dd0c2ea6539fbb784753f8c89c821643153b5` |
| `reports/REPORT-CTS-P2.md` | 7,773 | `aae46be8c9789096b235242920d2a2db067c3709c63e42cc0ea49e80b41f745a` |
| `reports/REPORT-CTS-P3.md` | 13,638 | `315f7859eb83cf5afda54b6f20d1b4073d6ecbac72329084a53885fd99189a8a` |
| `reports/REPORT-CTS-P3.v2.md` | 16,061 | `41ab4689751a9a18348cdce4eae598cdff7f2ca903a3f4a9c1e7ee32f1ef15d2` |
| `reports/REPORT-CTS-P3.v3.md` | 39,779 | `7b124251e3be327f4b29ad458d16c87bc27578152d2ca033d13c790bdb416fdc` |
| `reports/REPORT-CTS-P4.RUNNING.md` | 8,726 | `0a190d1bf4b67057ad87abf4f13e07ab18cb6bee24bdce530949264204758670` |
| `reports/REPORT-CTS-P4.md` | 9,609 | `0166c0b814de906ad9d5f4aab2a2187f48e33bbdcb8c04d4c19b67de77ef99d2` |
| `s7deq_intervene_P3.json` | 83,276 | `626591dcc8050a39d98f53900d40e2a238155979aa1321690537a34ac86b983c` |
| `shared_laptop_20261003/ESC-GPU1.md` | 96 | `f089fdbce7d89c7cccd4800b60d21f9d7df0674761d333ceefd40689507ebd25` |
| `shared_laptop_20261003/EXTERNAL_AUDIT_STANDING_ORDERS.md` | 4,566 | `18f27807fd6e4c04ceff2be14967acd9c15fc5fbf64b8ebf1aac68c9a01542c3` |
| `shared_laptop_20261003/GOAL.md` | 2,437 | `2e454ace1d33997356f5e4778efa744e57b5014c64973432935c3939f1487f6a` |
| `shared_laptop_20261003/LESSONS.md` | 6,326 | `1410869511179d4dece8c4febd0240ec3abea31a25badcbd41b10813264a0779` |
| `shared_laptop_20261003/RULES.md` | 2,713 | `179e1348d541dd13d29d9b7c3bd7c918834035664085b04c2a1ed29a73265619` |
| `shared_laptop_20261003/STATE.md` | 11,052 | `a81dd141d208f739b46af9b7dbc39f6f9618909126e47ce96381ec36a07ab546` |
| `shared_laptop_20261003/audits/AUDIT-001.md` | 3,777 | `f3b74ea3f5a3d90cd0213f6c6bebe49ea10bc5f80016125da7a9a38f2e3d6038` |
| `shared_laptop_20261003/audits/AUDIT-003.md` | 6,979 | `e22f9c2db278b516263570459a325fb134ae0a962fcd64ccbbdc0c0d071b5a41` |
| `shared_laptop_20261003/audits/AUDIT-003R.md` | 7,616 | `12d444fd9cc52b4300f206740849aeffe62075ccf89bc3f0473a79aa901230f4` |
| `shared_laptop_20261003/audits/AUDIT-CTS-409.md` | 4,665 | `f58fc34751f5d7e2b5fc1f8882e5c2557c302dcddde91946ec2ba9c0aa99e2aa` |
| `shared_laptop_20261003/audits/AUDIT-CTS-A1.md` | 3,501 | `3593eaf68568122697fddf1a6343ae0e6ef37c77d43d5d0a829ca7960432af6e` |
| `shared_laptop_20261003/audits/AUDIT-CTS-P0.md` | 2,554 | `a8da07a2ae62034da930f8ad3c52b7e2f8aedc6fb0f9d77d637e98a02ea832df` |
| `shared_laptop_20261003/audits/AUDIT-CTS-P1.md` | 3,670 | `78a169876b6537f938f5522ecba3cd87b203abdd139f468e27cb091a9b15ae5f` |
| `shared_laptop_20261003/audits/AUDIT-CTS-P2.md` | 3,917 | `a2a8582cc8943f44fc05a5a667f4e3f1bac30cfb2f823f2f6fbf3ac3ca359715` |
| `shared_laptop_20261003/audits/AUDIT-CTS-P3.md` | 4,476 | `06fe4cac57a61a8b1e2303963612bd84b11906ad813a79a9a603c36d4ea6a882` |
| `shared_laptop_20261003/audits/AUDIT-CTS-P3v2.md` | 4,567 | `317002a20acb7f96b8c06297ece8a310577f3cb157c507c4f827b7d7be55b504` |
| `shared_laptop_20261003/audits/AUDIT-CTS-P3v3.md` | 4,027 | `a7268c13e8dc20153ab2f4163e8c0b483a516f95af19798c96a7df00b79dffc5` |
| `shared_laptop_20261003/audits/AUDIT-CTS-P4-RUNNING.md` | 4,551 | `8e6e23a6bffb512e6cc3519316631153cf1eb4d60e3217adba4229f497276310` |
| `shared_laptop_20261003/audits/AUDIT-CTS-P4.md` | 3,252 | `6b17603dc101d9caa2b0a04ab88e5245664262aadc607e816185e8a13645924d` |
| `shared_laptop_20261003/audits/AUDIT-SO-408.md` | 2,515 | `cfdcb6e7861265ec54cb621ab130c451bacbece82ec7e6ce11e85e689855f7f3` |
| `shared_laptop_20261003/audits/archive/REPORT-001__8db1b9f9.md` | 23,707 | `6ce0b8fbee769e69cf813402580ada64f4fe256a0a863fc45c05c81a17bfa681` |
| `shared_laptop_20261003/audits/archive/REPORT-003R__ddaf7a95.md` | 8,694 | `1ceecaaf9f8c4d46db60f4adaa7d78dcb2d0115b32d281cf9d888e43e31dcb60` |
| `shared_laptop_20261003/audits/archive/REPORT-003__e3d7ef3f.md` | 8,915 | `a318c29196368a24e5d9b564b7f96d15ad74678527733e039081bd568808a372` |
| `shared_laptop_20261003/audits/archive/REPORT-CTS-A1__da686863.md` | 4,967 | `0113576900001c49d0d95abd0273496f8d9083029cfc1690c1b1b4825c599512` |
| `shared_laptop_20261003/audits/archive/REPORT-CTS-P0__9377261e.md` | 4,697 | `f092d0aa02fd70c7a98feda66ba62435f775dcbb9be89f1805d166abfe5bf49f` |
| `shared_laptop_20261003/audits/archive/REPORT-CTS-P1__812cf32c.md` | 6,973 | `266c0de366da3c3702409f8f737dd0c2ea6539fbb784753f8c89c821643153b5` |
| `shared_laptop_20261003/audits/archive/REPORT-CTS-P2__99914c31.md` | 7,773 | `aae46be8c9789096b235242920d2a2db067c3709c63e42cc0ea49e80b41f745a` |
| `shared_laptop_20261003/audits/archive/REPORT-CTS-P3.v2__57f6fe36.md` | 16,061 | `41ab4689751a9a18348cdce4eae598cdff7f2ca903a3f4a9c1e7ee32f1ef15d2` |
| `shared_laptop_20261003/audits/archive/REPORT-CTS-P3.v3__81b4fb51.md` | 39,779 | `7b124251e3be327f4b29ad458d16c87bc27578152d2ca033d13c790bdb416fdc` |
| `shared_laptop_20261003/audits/archive/REPORT-CTS-P3__93d0d349.md` | 13,638 | `315f7859eb83cf5afda54b6f20d1b4073d6ecbac72329084a53885fd99189a8a` |
| `shared_laptop_20261003/audits/archive/REPORT-CTS-P4.RUNNING__3a768f9a.md` | 8,726 | `0a190d1bf4b67057ad87abf4f13e07ab18cb6bee24bdce530949264204758670` |
| `shared_laptop_20261003/audits/archive/REPORT-CTS-P4__8fb73986.md` | 9,609 | `0166c0b814de906ad9d5f4aab2a2187f48e33bbdcb8c04d4c19b67de77ef99d2` |
| `shared_laptop_20261003/reports/.seen_REPORT-_.md` | 779 | `603dd6cdb495350a52fa8ee8f4cb4335cb3616c1b65ea2371c4e4c555c0682d3` |
| `shared_laptop_20261003/reports/HANDOVER_TO_CTS_20261004.md` | 19,962 | `7c3dfef1c7274900c9e5e80cbc2963374c6d1dd6944c7dab4ed55f9829362ddf` |
| `shared_laptop_20261003/reports/REPORT-001.md` | 23,707 | `6ce0b8fbee769e69cf813402580ada64f4fe256a0a863fc45c05c81a17bfa681` |
| `shared_laptop_20261003/reports/REPORT-003.md` | 8,915 | `a318c29196368a24e5d9b564b7f96d15ad74678527733e039081bd568808a372` |
| `shared_laptop_20261003/reports/REPORT-003R.RUNNING.done.bak` | 633 | `49134493ea9dc3b09bcea10d369203e1140774030ccd1f68c8f94ff26e13acbb` |
| `shared_laptop_20261003/reports/REPORT-003R.md` | 8,694 | `1ceecaaf9f8c4d46db60f4adaa7d78dcb2d0115b32d281cf9d888e43e31dcb60` |
| `shared_laptop_20261003/reports/REPORT-CTS-409.md` | 6,062 | `3658bc11e42c3ff7a59aa3a03367f8a98484351870685c9283d91009256dcc40` |
| `shared_laptop_20261003/reports/REPORT-CTS-A1.md` | 4,967 | `0113576900001c49d0d95abd0273496f8d9083029cfc1690c1b1b4825c599512` |
| `shared_laptop_20261003/reports/REPORT-CTS-P0.md` | 4,697 | `f092d0aa02fd70c7a98feda66ba62435f775dcbb9be89f1805d166abfe5bf49f` |
| `shared_laptop_20261003/reports/REPORT-CTS-P1.md` | 6,973 | `266c0de366da3c3702409f8f737dd0c2ea6539fbb784753f8c89c821643153b5` |
| `shared_laptop_20261003/reports/REPORT-CTS-P2.md` | 7,773 | `aae46be8c9789096b235242920d2a2db067c3709c63e42cc0ea49e80b41f745a` |
| `shared_laptop_20261003/reports/REPORT-CTS-P3.md` | 13,638 | `315f7859eb83cf5afda54b6f20d1b4073d6ecbac72329084a53885fd99189a8a` |
| `shared_laptop_20261003/reports/REPORT-CTS-P3.v2.md` | 16,061 | `41ab4689751a9a18348cdce4eae598cdff7f2ca903a3f4a9c1e7ee32f1ef15d2` |
| `shared_laptop_20261003/reports/REPORT-CTS-P3.v3.md` | 39,779 | `7b124251e3be327f4b29ad458d16c87bc27578152d2ca033d13c790bdb416fdc` |
| `shared_laptop_20261003/reports/REPORT-CTS-P4.RUNNING.md` | 8,726 | `0a190d1bf4b67057ad87abf4f13e07ab18cb6bee24bdce530949264204758670` |
| `shared_laptop_20261003/reports/REPORT-CTS-P4.md` | 9,609 | `0166c0b814de906ad9d5f4aab2a2187f48e33bbdcb8c04d4c19b67de77ef99d2` |
| `shared_laptop_20261003/tasks/.seen_TASK-_.md` | 261 | `55bdd5fdc9e565322fc83a49bebc890205d6f061a7ddbc03cdb6a175d174f74a` |
| `shared_laptop_20261003/tasks/PROMPT-0.md` | 3,355 | `dd8098d9033c2e9d8f3d5b182977eb46c9b89fd7d4ad5b8abdaae44eec1cf804` |
| `shared_laptop_20261003/tasks/PROMPT-1.md` | 1,997 | `a768360bcb308837c6e14c1e5ca856cd1116a7572727b752a3c0aef4cae00e38` |
| `shared_laptop_20261003/tasks/PROMPT-2.md` | 1,399 | `1bf78d796be85bdc499738be3cc1668f32958bd3445ec96c994c7233f96f3675` |
| `shared_laptop_20261003/tasks/PROMPT-3.md` | 1,617 | `3010bc7179d2ddea1f46ce88fab4784ccfd42a2af28c8dd5a99fbf97890309ea` |
| `shared_laptop_20261003/tasks/PROMPT-4.md` | 1,574 | `8625494ef966dc43fa3773278b787263b16255b8f1ad18b59346e5a54d69e060` |
| `shared_laptop_20261003/tasks/PROMPT-5.md` | 1,362 | `94b270868a2f63ec76be5a35b4386b5391521ddac93a30330df872d8666fdc41` |
| `shared_laptop_20261003/tasks/PROMPT-6.md` | 1,572 | `b39a4e464d3cc2e97edc1ac0e25f343c6eb1c2582bf6b0107648cd19703d31ad` |
| `shared_laptop_20261003/tasks/TASK-001.md` | 9,333 | `eedc7ce7cd8ee901cb3c98a8df73b6851f0e1d2ef5bf2cecf3aa84c024f7a808` |
| `shared_laptop_20261003/tasks/TASK-001R.md` | 5,925 | `ef1d822a9450d4d6d72c7278128307b15cb17b818dc2da949a1690e5ce80d654` |
| `shared_laptop_20261003/tasks/TASK-002.md` | 5,730 | `b7921b9fe358dda148e46db64bf1070713d7e1a25f0ef9170ec2e7b308f26710` |
| `shared_laptop_20261003/tasks/TASK-003.md` | 5,216 | `a1d1b4b1857cbff1d75ea535b7236c39d2314f11efeb167b14c7b980363ac516` |
| `shared_laptop_20261003/tasks/TASK-003R.md` | 5,130 | `abba88e6962313f6956375c05ae4c55378d31dcb255e573be5197de320e4e004` |
| `shared_laptop_20261003/tasks/TASK-003R2.md` | 2,638 | `b80430bcb3fceb8937b393c7c02cea80cdef52bd2f37b17b4357327dde560366` |
| `shared_laptop_20261003/wait_for.js` | 2,701 | `faddb2657f2180f3267a7ec366b883d37680139c0bb8cc6cb7dc049b60ca3b0a` |
| `shared_laptop_20261003/wait_for.py` | 2,322 | `cbdad6e169d402a9778c993041381baf424e74d37225c9370229597dd8656e4c` |
| `tasks/.seen_TASK-_.md` | 261 | `55bdd5fdc9e565322fc83a49bebc890205d6f061a7ddbc03cdb6a175d174f74a` |
| `tasks/PROMPT-0.md` | 3,355 | `dd8098d9033c2e9d8f3d5b182977eb46c9b89fd7d4ad5b8abdaae44eec1cf804` |
| `tasks/PROMPT-1.md` | 1,997 | `a768360bcb308837c6e14c1e5ca856cd1116a7572727b752a3c0aef4cae00e38` |
| `tasks/PROMPT-2.md` | 1,399 | `1bf78d796be85bdc499738be3cc1668f32958bd3445ec96c994c7233f96f3675` |
| `tasks/PROMPT-3.md` | 1,617 | `3010bc7179d2ddea1f46ce88fab4784ccfd42a2af28c8dd5a99fbf97890309ea` |
| `tasks/PROMPT-4.md` | 1,574 | `8625494ef966dc43fa3773278b787263b16255b8f1ad18b59346e5a54d69e060` |
| `tasks/PROMPT-5.md` | 1,362 | `94b270868a2f63ec76be5a35b4386b5391521ddac93a30330df872d8666fdc41` |
| `tasks/PROMPT-6.md` | 1,572 | `b39a4e464d3cc2e97edc1ac0e25f343c6eb1c2582bf6b0107648cd19703d31ad` |
| `tasks/TASK-001.md` | 9,333 | `eedc7ce7cd8ee901cb3c98a8df73b6851f0e1d2ef5bf2cecf3aa84c024f7a808` |
| `tasks/TASK-001R.md` | 5,925 | `ef1d822a9450d4d6d72c7278128307b15cb17b818dc2da949a1690e5ce80d654` |
| `tasks/TASK-002.md` | 5,730 | `b7921b9fe358dda148e46db64bf1070713d7e1a25f0ef9170ec2e7b308f26710` |
| `tasks/TASK-003.md` | 5,216 | `a1d1b4b1857cbff1d75ea535b7236c39d2314f11efeb167b14c7b980363ac516` |
| `tasks/TASK-003R.md` | 5,130 | `abba88e6962313f6956375c05ae4c55378d31dcb255e573be5197de320e4e004` |
| `tasks/TASK-003R2.md` | 2,638 | `b80430bcb3fceb8937b393c7c02cea80cdef52bd2f37b17b4357327dde560366` |
| `wait_for.js` | 2,701 | `faddb2657f2180f3267a7ec366b883d37680139c0bb8cc6cb7dc049b60ca3b0a` |
| `wait_for.py` | 2,322 | `cbdad6e169d402a9778c993041381baf424e74d37225c9370229597dd8656e4c` |

---

## 4. 인수인계 3단계 수행 기록 및 후속 단계 인계

1. **3단계 이행 요건 대조**:
   - 서버 §0 실측 1회: 1절 계측표 기록 (RAM 224GiB 가용, GPU 8대 0% 유휴, tmux 세션 0건).
   - 노트북 데이터 수령: `~/cts_e1_s5/shared_laptop_20261003/` 전수 수령 및 `results_s7/s7deq_intervene_P3.json` 수령 (`from_laptop/` 격리 보관).
   - P3 진실 추출 및 오기 배제: 2절 실측 대조표 및 오기 배제 기록(사용 0) 명기.
   - 파일 인벤토리 해시 발행: 3절 SHA-256 전수 목록 발행.
2. **4단계(회사 PC Claude Code A) 진입 대기**:
   - 본 문서 및 `shared/from_laptop/` 내 `STATE.md`, `LEDGER.md`, `LESSONS.md`를 바탕으로 정식 `shared/`의 `GOAL·GRAPH·STATE·LESSONS` 갱신 준비 상태.
   - 첫 작업 대상: `HANDOFF_cts260930_to_cts.md` 4절 원장 #409 입력 양식 확인.
