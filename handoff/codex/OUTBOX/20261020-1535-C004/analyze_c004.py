"""C004 CPU audit. Existing files are read-only; all outputs use exclusive creation."""
import sys, json, hashlib, random, statistics, re, platform, importlib.metadata
from pathlib import Path
import numpy as np

ROOT = Path("D:/AI/cts")
OUT = Path(__file__).resolve().parent
PKG = ROOT / "handoff/codex/INBOX/20261020-1320-x5-results"
DATA = ROOT / "handoff/codex/data"
ORDER = DATA / "p6_item_order.json"
EXPECTED_ORDER = "6c76ddb7cc4a1ee901291eee76ed99763ec5ebfec202cb30ba17f9bcee487e7c"
EXPECTED_GRADER = "90cc93e4b76704b209291ac43999a3f829ac79f7234c60362f95bccadd12fa13"
GEN = {"C_fmt": DATA/"gen_C.jsonl", "X5O47": PKG/"gen_X5O47.jsonl", "shuf_X5O47": PKG/"genB1_X5O47.jsonl"}
PAIRS = [("X5O47","shuf_X5O47"),("X5O47","C_fmt"),("shuf_X5O47","C_fmt")]

def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def write(name, obj):
    with (OUT/name).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def ids(p):
    d=read(p)
    return [int(x) for x in (d["indices"] if isinstance(d,dict) else d)]

def jsonl(p):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

def prepare():
    assert sha(ORDER)==EXPECTED_ORDER
    for p in [ROOT/"scripts/s5_eval_grader_v2.py",ROOT/"handoff/codex/scripts/s5_eval_grader_v2.py"]:
        assert sha(p)==EXPECTED_GRADER
    order=ids(ORDER);train=ids(ROOT/"configs/s7_train_993.json")
    assert len(order)==len(set(order))==600
    assert len(train)==len(set(train))==993 and not set(order)&set(train)
    records=jsonl(DATA/"dev600_records.jsonl")
    assert len(records)==600 and {int(r["id"]) for r in records}==set(order)
    schemas={}
    for label,p in GEN.items():
        rows=jsonl(p)
        assert len(rows)==600 and len({int(r["id"]) for r in rows})==600
        assert {int(r["id"]) for r in rows}==set(order)
        schemas[label]={"rows":len(rows),"keys":sorted(rows[0]),"id_order_equal": [int(r["id"]) for r in rows]==order}
    cc=read(PKG/"x5_items.json")
    assert set(cc)==set(GEN)
    for label in GEN:
        assert set(map(int,cc[label]))==set(order)
        assert all(len(v)==3 and all(x in [0,1] for x in v) for v in cc[label].values())
    declared=[]
    for line in (PKG/"SHA256SUMS.txt").read_text().splitlines():
        value,name=line.split(" *",1);p=ROOT/name
        declared.append({"path":name,"declared":value,"declared_chars":len(value),"actual_sha256":sha(p),"full_match":value==sha(p)})
    master=(PKG/"cts_x5_master.log").read_text(encoding="utf-8")
    log_hashes=[]
    for value,name in re.findall(r"^([0-9a-f]{64})\s+(.+)$",master,re.M):
        base=Path(name).name
        candidates=[PKG/base,ROOT/"scripts"/base,ROOT/"handoff/codex/scripts"/base]
        for p in candidates:
            if p.is_file():
                log_hashes.append({"logged_path":name,"local_path":str(p),"expected":value,"actual":sha(p),"match":value==sha(p)})
                assert value==sha(p)
                break
    sources=list(p for p in PKG.iterdir() if p.is_file())+list(GEN.values())+[ORDER,DATA/"dev600_records.jsonl",ROOT/"configs/s7_train_993.json",ROOT/"AGENTS.md",ROOT/"handoff/codex/AGENTS.md",ROOT/"handoff/codex/README.md",ROOT/"handoff/codex/CONTEXT.md",ROOT/"shared/proposals/X_INTERFACE_PLAN_20261019.md"]
    sources += [ROOT/"handoff/codex/INBOX"/n for n in ["TASK-C002.md","ANSWER-C003-20261020-0900.md","ANSWER-C003b-20261020-0930.md","ANSWER-C004Q-20261020-1255.md"]]
    sources += [ROOT/f"{folder}/{name}" for folder in ["scripts","handoff/codex/scripts"] for name in ["s5_eval_grader_v2.py","g2v10_train.py","cc_x5.sh"]]
    sources += [ROOT/"handoff/codex/scripts"/n for n in ["codex_grade_local.py","s7_eval_score_p6_v36.py"]]
    manifest=[{"path":str(p.relative_to(ROOT)),"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(set(sources))]
    write("INPUT_MANIFEST.json",manifest)
    write("preflight.json",{"train_n":len(train),"dev_n":len(order),"intersection":0,"schemas":schemas,"package_hash_file":declared,"server_log_hash_comparisons":log_hashes,"canonical_order_path_exists":(ROOT/"results_s7/p6_item_order.json").exists(),"actual_order":str(ORDER),"versions":{"python":sys.version,"platform":platform.platform(),**{x:importlib.metadata.version(x) for x in ["numpy","sympy","math-verify"]}}})
    write("s0spec_local.json",{label:[str(p),str(ORDER)] for label,p in GEN.items()})
    print("PREPARED: 3 x 600 generation rows; 600 dev IDs; grader/order hashes verified.")

def analyze():
    order=sorted(ids(ORDER)); server=read(PKG/"x5_items.json");local=read(OUT/"x5_items_local.json")
    for label in GEN:
        assert set(map(int,local[label]))==set(order)
    # Same resampled item indices for every metric; identical per-metric seed
    # and order-statistic CI endpoints as the CC script, independently coded.
    rng=random.Random(2026)
    indices=np.array([[rng.randrange(600) for _ in range(600)] for _ in range(10000)],dtype=np.int32)
    stats={}
    for source,d in [("server",server),("local",local)]:
        arrays={label:np.array([d[label][str(q)] for q in order],dtype=np.int64) for label in GEN}
        aggregates={}
        for label,a in arrays.items():
            counts=a.sum(0).tolist()
            aggregates[label]={"n":len(a),"strict":counts[0],"lenient":counts[1],"boxed":counts[2],"rates_pct":(a.mean(0)*100).tolist(),"strict_among_boxed_pct":100*int(np.sum(a[:,0]*a[:,2]))/int(a[:,2].sum())}
        differences={}
        overlaps={}
        for left,right in PAIRS:
            key=left+" - "+right;differences[key]={};overlaps[key]={}
            for idx,metric in [(0,"strict"),(1,"lenient")]:
                x=arrays[left][:,idx];y=arrays[right][:,idx];delta=x-y
                boot=np.sort(delta[indices].mean(1)*100)
                differences[key][metric]={"delta_pp":float(delta.mean()*100),"ci95_pp":[float(boot[250]),float(boot[9749])],"B":10000,"seed":2026,"resampling_unit":"item","n_seeds":1}
                overlaps[key][metric]={"both":int(np.sum((x==1)&(y==1))),"left_only":int(np.sum((x==1)&(y==0))),"right_only":int(np.sum((x==0)&(y==1))),"neither":int(np.sum((x==0)&(y==0)))}
        stats[source]={"aggregates":aggregates,"differences":differences,"overlaps":overlaps}
    gens={label:{int(r["id"]):r for r in jsonl(p)} for label,p in GEN.items()}
    gold={int(r["id"]):r for r in jsonl(DATA/"dev600_records.jsonl")}
    # Official extraction functions only; grading is exclusively the unchanged
    # local wrapper. These calls annotate discrepancies, not regrade them.
    sys.path.insert(0,str(ROOT/"handoff/codex/scripts"))
    import s5_eval_grader_v2 as grader
    mismatch=[];directions={}
    for label in GEN:
        directions[label]={m:{"server_only":0,"local_only":0,"ids":[]} for m in ["strict","lenient","boxed"]}
        for q in order:
            a=server[label][str(q)];b=local[label][str(q)]
            if a==b:continue
            text=gens[label][q]["generation"]
            mismatch.append({"label":label,"id":q,"server":a,"local":b,"pred_boxed":grader._extract_boxed(text),"pred_lenient":grader.lenient_extract(text),"gold_extracted":grader.extract_answer(gold[q]["solution"]),"generation":text,"gold_solution":gold[q]["solution"]})
            for i,m in enumerate(["strict","lenient","boxed"]):
                if a[i]!=b[i]:
                    directions[label][m]["server_only" if a[i] else "local_only"]+=1;directions[label][m]["ids"].append(q)
    lengths={}
    for label,g in gens.items():
        vals=[g[q]["tokens"] for q in order]
        stops=[g[q].get("stop_id") for q in order]
        lengths[label]={"n":len(vals),"mean_tokens":statistics.mean(vals),"median_tokens":statistics.median(vals),"min_tokens":min(vals),"max_tokens":max(vals),"quantiles_linear":np.quantile(vals,[.25,.75,.9,.95]).tolist(),"tokens_eq512":sum(v==512 for v in vals),"tokens_lt512":sum(v<512 for v in vals),"stop_ids":{str(v):stops.count(v) for v in set(stops)},"eos_count":sum(s in [1,106] for s in stops),"missing_stop_id":sum("stop_id" not in g[q] for q in order)}
    man=read(PKG/"manifest_X5O47.json");tel=man["telemetry"]
    recomputed_loss=statistics.mean(x["loss"] for x in tel[-200:])
    recomputed_ans=statistics.mean(x["ans_ce"] for x in tel[-200:] if x["ans_n"])
    manifest_checks={"telemetry_n":len(tel),"steps_contiguous": [x["step"] for x in tel]==list(range(man["steps"])),"finite_telemetry":all(np.isfinite(x[k]) for x in tel for k in ["loss","ans_ce"]),"ans_n_counts":{str(n):sum(x["ans_n"]==n for x in tel) for n in set(x["ans_n"] for x in tel)},"last200_loss":recomputed_loss,"last200_ans_ce":recomputed_ans,"loss_difference":recomputed_loss-man["final_loss_last200"],"ans_ce_difference":recomputed_ans-man["final_ans_ce_last200"],"replaced_plus_appended":sum(man["target_counts"][k] for k in ["replaced","appended"]),"budget_truncated_pct_of_retained":100*man["target_counts"]["budget_truncated"]/man["n_items_trained"],"budget_truncated_pct_of_listed":100*man["target_counts"]["budget_truncated"]/man["n_items_listed"],"visits_minus_steps_times_batch":man["n_sample_visits"]-man["steps"]*man["batch_size"],"train_sha_matches":man["items_sha256"]==sha(ROOT/"configs/s7_train_993.json"),"dev_sha_matches":man["dev_order_sha256"]==sha(ORDER),"raw_train_targets_reconstructed":False,"excluded_id_available":False,"training_counts_source":"reported manifest; aggregate consistency checked, not independent replay"}
    changed=[r["path"] for r in read(OUT/"INPUT_MANIFEST.json") if sha(ROOT/r["path"])!=r["sha256"]]
    write("statistics.json",stats);write("discrepancies.json",{"directions":directions,"rows":mismatch})
    write("generation_lengths.json",lengths);write("training_manifest_checks.json",manifest_checks)
    write("input_integrity_after.json",{"checked":len(read(OUT/"INPUT_MANIFEST.json")),"changed":changed})
    assert not changed
    print(json.dumps({"statistics":stats,"mismatch_item_label_pairs":len(mismatch),"directions":directions,"lengths":lengths,"manifest_checks":manifest_checks},ensure_ascii=False,indent=2))

if __name__=="__main__":
    {"prepare":prepare,"analyze":analyze}[sys.argv[1]]()
