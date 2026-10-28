import torch
from pathlib import Path

def main():
    p = Path("/home/shoon/system1_5/checkpoints/stage1_ddp_5gpu_sealed.pt")
    if not p.is_file():
        print(f"File not found: {p}")
        return
        
    c = torch.load(p, map_location="cpu")
    print(f"Root keys: {list(c.keys())}")
    
    sd = c.get("state_dict", c)
    print(f"State dict class: {type(sd)}")
    if isinstance(sd, dict):
        keys = list(sd.keys())
        print(f"Number of keys: {len(keys)}")
        print(f"Top 20 keys: {keys[:20]}")
        print("\nTop 5 shapes:")
        for k in keys[:5]:
            print(f"  {k}: {sd[k].shape}")
            
if __name__ == "__main__":
    main()
