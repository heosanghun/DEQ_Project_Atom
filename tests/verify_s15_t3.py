import os
import sys
import torch
import json
from pathlib import Path

# Add system1_5 to path
sys.path.insert(0, "/home/shoon/system1_5")

from system1_5.backbone.gemma_loader import load_backbone
from system1_5.backbone.deq_wrapper import DEQGemmaWrapper
from system1_5.config import DEFAULT_CONFIG
from system1_5.fwp import FastWeightProgrammer
from data.finqa_subset import load_finqa_subset
from system1_5.backbone.attention_overlay import build_specs_for_blocks

def main():
    print("=== STARTING SYSTEM 1.5 T3 WIRING VERIFICATION ===", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    cfg = DEFAULT_CONFIG
    cfg.backbone.deq_wrap_blocks = 6
    cfg.backbone.use_quant = True # Enable bitsandbytes 4-bit quantization to match checkpoints
    cfg.backbone.model_id = "/home/shoon/System2.5_deq/hf_cache/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
    
    # 1. Load checkpoints
    fwp_ckpt = "checkpoints/fwp/fwp_step_50.pt"
    deq_ckpt = "checkpoints/stage1_ddp_5gpu_sealed.pt"
    
    if not os.path.exists(fwp_ckpt) or not os.path.exists(deq_ckpt):
        print(f"Error: Checkpoints not found! fwp={fwp_ckpt}, deq={deq_ckpt}", flush=True)
        sys.exit(1)
        
    print(f"Loading checkpoints: FWP={fwp_ckpt}, DEQ={deq_ckpt}", flush=True)
    fwp_meta = torch.load(fwp_ckpt, map_location="cpu")
    deq_meta = torch.load(deq_ckpt, map_location="cpu")
    
    # Update config from FWP metadata
    fwp_cfg_dict = fwp_meta.get("fwp_cfg", fwp_meta.get("config", None))
    if fwp_cfg_dict is not None:
        from dataclasses import replace
        cfg.fwp = replace(cfg.fwp, **fwp_cfg_dict)
        
    # 2. Build model and load state dicts
    from scripts.finetune_deq import _apply_lora
    model, tokenizer, info = load_backbone(cfg.backbone, cfg.tokens)
    model, mode = _apply_lora(model, cfg)
    wrapper = DEQGemmaWrapper(model, cfg.backbone, cfg.solver)
    
    sd = deq_meta.get("state_dict", deq_meta.get("deq_state", deq_meta))
    wrapper.load_state_dict(sd, strict=False)
        
    wrapper.freeze_base()
    wrapper.eval()

    fwp = FastWeightProgrammer(info.d_model, cfg.backbone.deq_wrap_blocks, cfg.fwp).to(device)
    if "fwp_state" in fwp_meta:
        fwp.load_state_dict(fwp_meta["fwp_state"], strict=True)
    else:
        fwp.load_state_dict(fwp_meta, strict=True)
    fwp.eval()

    # 3. Load 10 samples from FinQA subset
    examples = load_finqa_subset(10)
    print(f"Loaded {len(examples)} samples from FinQA subset.", flush=True)
    
    diff_results = []
    
    for idx, ex in enumerate(examples):
        print(f"\nProcessing sample {idx} (id={ex.qid})...", flush=True)
        ids = tokenizer(ex.question, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Full DEQ: standard baseline forward without FWP overlays
            base_out = wrapper.forward(ids["input_ids"], attention_mask=ids["attention_mask"])
            z_star = base_out.z_star
            base_logits = base_out.logits[:, -1, :]
            
            # FWP computation
            z_summary = wrapper.extract_z_summary(z_star)
            fws = fwp(z_summary, eps=0.002)
            
            # CFD Mode: forward with fast weights overlay (Delta W)
            overlay_dict = fwp.to_overlay_dict(fws)
            specs_cfd = build_specs_for_blocks(wrapper.deq_blocks(), overlay_dict)
            cfd_out = wrapper.forward(
                ids["input_ids"],
                attention_mask=ids["attention_mask"],
                overlay_specs=specs_cfd,
                cfd_mode=True,
                z_init=z_star,
            )
            cfd_logits = cfd_out.logits[:, -1, :]
            
            # Compute difference in logit space
            max_logit_diff = (base_logits - cfd_logits).abs().max().item()
            mean_logit_diff = (base_logits - cfd_logits).abs().mean().item()
            
            # Top-1 token comparisons
            base_tok = base_logits[0].argmax().item()
            cfd_tok = cfd_logits[0].argmax().item()
            
            base_word = tokenizer.decode([base_tok])
            cfd_word = tokenizer.decode([cfd_tok])
            
            token_changed = base_tok != cfd_tok
            
            print(f"  Max Logit Diff: {max_logit_diff:.6f}")
            print(f"  Base Top-1 Token: {repr(base_word)} ({base_tok})")
            print(f"  CFD Top-1 Token:  {repr(cfd_word)} ({cfd_tok})")
            print(f"  Token Changed: {token_changed}", flush=True)
            
            diff_results.append({
                "sample_index": idx,
                "sample_id": ex.qid,
                "max_logit_diff": max_logit_diff,
                "mean_logit_diff": mean_logit_diff,
                "base_token_id": base_tok,
                "cfd_token_id": cfd_tok,
                "base_token": base_word,
                "cfd_token": cfd_word,
                "token_changed": token_changed
            })
            
    # Output json report
    out_dir = Path("/home/shoon/system1_5/evidence/rm02_stage2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "t3_wiring_diff.json"
    
    summary = {
        "total_samples": len(examples),
        "total_changed": sum(1 for r in diff_results if r["token_changed"]),
        "max_logit_diff_across_all": max(r["max_logit_diff"] for r in diff_results),
        "results": diff_results
    }
    
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print(f"\nT3 verification summary: {summary['total_changed']}/{summary['total_samples']} tokens changed.", flush=True)
    print(f"Saved T3 report to {out_file}", flush=True)
    
    # Return exit code based on changes: if there is at least one difference, z* wiring is live (EXIT_OK)
    if summary["total_changed"] > 0 or summary["max_logit_diff_across_all"] > 1e-4:
        print("[T3-RESULT] z* wiring is ACTIVE and FUNCTIONAL. Outputs are distinct.", flush=True)
        sys.exit(0)
    else:
        print("[T3-RESULT] z* wiring is INACTIVE. Outputs are identical (diff=0).", flush=True)
        sys.exit(2)

if __name__ == "__main__":
    main()
