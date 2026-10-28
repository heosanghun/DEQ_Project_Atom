#!/usr/bin/env python3
r"""
loop_driver.py — CTS 자율 루프 드라이버 (0 토큰 감시 + 파일 기반 A/B/EXE 호출)
설치: D:\AI\cts\shared\loop_driver.py   실행: python loop_driver.py   (Windows 작업 스케줄러로 로그인 시 자동 시작 권장)
역할 호출은 Claude Code 헤드리스( `claude -p` )를 기본으로 하며, EXE 역할은 Antigravity(inbox 파일 방식)로도 둘 수 있다.
※ `claude -p`의 정확한 플래그는 설치본에서 `claude --help`로 확인 후 CLAUDE_CMD를 맞출 것.
"""
import os, re, sys, time, json, glob, subprocess, datetime, pathlib

ROOT = pathlib.Path(__file__).resolve().parent            # D:\AI\cts\shared
SSH  = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-p", "10022", "shoon@<GPU_SERVER_IP>"]  # 키 인증만 (비밀번호 파일 금지)
CYCLE_SEC       = 600            # 10분
STALL_MIN       = 30             # 파이프라인 로그 무변화 허용(분)
ESC_STALL_HOURS = 3
EXE_MODE        = os.environ.get("CTS_EXE_MODE", "claude")   # "claude" | "antigravity"
CLAUDE_CMD      = ["claude", "-p", "--output-format", "text"]  # 필요 시 --model, --allowedTools 추가

def now(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
def read(p):
    p = ROOT / p
    return p.read_text(encoding="utf-8") if p.exists() else ""
def write(p, s):
    p = ROOT / p; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(s, encoding="utf-8")
def log(msg):
    write(f"logs/driver_{datetime.date.today()}.log", read(f"logs/driver_{datetime.date.today()}.log") + f"{now()} {msg}\n")

# ---------- 1. 서버 건강 (0 토큰) ----------
def server_health():
    cmd = ("free -g | awk '/Mem/{print \"AVAIL_GB=\"$7}'; "
           "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\\n' ';'; echo; "
           "tmux ls 2>/dev/null | cut -d: -f1 | tr '\\n' ','; echo; "
           "stat -c %Y /home/shoon/cts_e1_s5/raw_s7/s7_zstar_pipeline.log 2>/dev/null; "
           "tail -n 1 /home/shoon/cts_e1_s5/raw_s7/s7_zstar_pipeline.log 2>/dev/null")
    try:
        out = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=60)
        if out.returncode != 0: return {"ok": False, "why": (out.stderr or out.stdout)[:200]}
        lines = out.stdout.strip().splitlines() + ["", "", "", ""]
        avail = int(re.sub(r"\D", "", lines[0]) or 0)
        gpus, tmux, mtime, last = lines[1], lines[2], lines[3], lines[4]
        age_min = (time.time() - int(mtime)) / 60 if mtime.strip().isdigit() else 9999
        return {"ok": True, "avail_gb": avail, "gpus": gpus, "tmux": tmux, "log_age_min": round(age_min), "last": last[:120]}
    except Exception as e:
        return {"ok": False, "why": str(e)[:200]}

# ---------- 2. 상태판 상단 갱신 ----------
def update_state(h, node, esc_open):
    top = [
        "# STATE.md — 상단 10줄만 사람이 본다 (스크립트+A 갱신)",
        f"현재 노드: {node}",
        f"서버: {'OK' if h['ok'] else '무응답 — ' + h.get('why','')} " + (f"| available {h['avail_gb']} GB | GPU {h['gpus']} | tmux {h['tmux']}" if h["ok"] else ""),
        f"파이프라인: 로그 최신화 {h.get('log_age_min','?')}분 전 | {h.get('last','')}" if h["ok"] else "파이프라인: 미확인",
        f"ESC 대기: {', '.join(esc_open) if esc_open else '없음'}",
        f"갱신: {now()}",
        "--- 이하 상세 (에이전트용) ---",
    ]
    body = read("STATE.md").split("--- 이하 상세 (에이전트용) ---", 1)
    detail = body[1] if len(body) > 1 else ""
    write("STATE.md", "\n".join(top) + detail)

# ---------- 3. 파일 상태 ----------
def ids(folder, prefix):
    return sorted(int(m.group(1)) for f in glob.glob(str(ROOT / folder / f"{prefix}-*.md")) if (m := re.search(r"-(\d{4})\.md$", f)))
def pending():
    t, r, a = set(ids("tasks","TASK")), set(ids("reports","REPORT")), set(ids("audits","AUDIT"))
    return {"exe": sorted(t - r), "audit": sorted(r - a), "plan": (max(a) if a and max(a) >= max(t, default=0) else None)}
def open_escalations():
    out = []
    for f in glob.glob(str(ROOT / "escalate" / "ESC-*.md")):
        if "답:" not in pathlib.Path(f).read_text(encoding="utf-8"): out.append(pathlib.Path(f).stem)
    return out

# ---------- 4. 역할 호출 (토큰 사용 지점) ----------
def call_claude(role_prompt_file, context_files, extra=""):
    ctx = "\n\n".join(f"<<FILE {c}>>\n{read(c)}" for c in context_files)
    prompt = read(role_prompt_file) + "\n\n" + ctx + "\n\n" + extra
    try:
        out = subprocess.run(CLAUDE_CMD + [prompt], capture_output=True, text=True, timeout=3600, cwd=str(ROOT.parent))
        return out.stdout
    except Exception as e:
        log(f"claude call failed: {e}"); return ""

def run_exe(n):
    task = f"tasks/TASK-{n:04d}.md"
    if EXE_MODE == "antigravity":
        write(f"inbox/EXE/TASK-{n:04d}.md", read(task)); log(f"TASK-{n:04d} → inbox/EXE (Antigravity)"); return
    out = call_claude("prompts/EXE_PROMPT.md", ["STATE.md", "RULES.md", task], f"REPORT-{n:04d}.md 본문만 출력.")
    if out.strip(): write(f"reports/REPORT-{n:04d}.md", out)
def run_audit(n):
    out = call_claude("prompts/B_PROMPT.md", ["RULES.md", "GOAL.md", f"tasks/TASK-{n:04d}.md", f"reports/REPORT-{n:04d}.md"], f"AUDIT-{n:04d}.md 본문만 출력.")
    if out.strip(): write(f"audits/AUDIT-{n:04d}.md", out); write("LEDGER.md", read("LEDGER.md") + f"\n{now()} AUDIT-{n:04d} " + out.splitlines()[-1][:200])
def run_plan(last_audit):
    nxt = max(ids("tasks","TASK"), default=0) + 1
    out = call_claude("prompts/A_PROMPT.md", ["STATE.md", "GOAL.md", "GRAPH.md", "RULES.md", f"audits/AUDIT-{last_audit:04d}.md"], f"다음 TASK-{nxt:04d}.md 본문만 출력. ESC가 필요하면 ESC-NNNN.md 본문을 대신 출력.")
    if out.strip().startswith("# ESC"): write(f"escalate/ESC-{nxt:04d}.md", out)
    elif out.strip(): write(f"tasks/TASK-{nxt:04d}.md", out)

# ---------- 5. 메인 루프 ----------
def main():
    last_progress = time.time()
    while True:
        h = server_health(); esc = open_escalations(); p = pending()
        node = "N0 SERVER_RECOVER" if not h["ok"] else read("STATE.md").split("현재 노드:",1)[-1].splitlines()[0].strip() if "현재 노드:" in read("STATE.md") else "?"
        update_state(h, node, esc); log(f"health={h} pending={p} esc={esc}")
        if not h["ok"]:
            if not esc and (time.time() - last_progress) > ESC_STALL_HOURS * 3600:
                write("escalate/ESC-SERVER.md", read("templates/ESC.md").replace("<한 줄>", "서버 SSH 무응답 3시간 — 관리자 조치 요청 허용?"))
        else:
            last_progress = time.time()
            if h["log_age_min"] > STALL_MIN: log("pipeline stall suspected")
            if p["exe"]:   run_exe(p["exe"][0])
            elif p["audit"]: run_audit(p["audit"][0])
            elif p["plan"] is not None: run_plan(p["plan"])
        time.sleep(CYCLE_SEC)

if __name__ == "__main__":
    main()
