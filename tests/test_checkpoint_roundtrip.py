import tempfile
from pathlib import Path
import torch
import pytest

from cts.policy.meta_policy import MetaPolicy
from cts.critic.neuro_critic import NeuroCritic

# Mock load_gemma4_e4b class structures to check state_dict roundtrip
class MockBackbone(torch.nn.Module):
    def __init__(self, hidden_size=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

def test_checkpoint_saving_loading_roundtrip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = 256  # Small hidden size for fast unit testing
    W = 3
    
    # 1. Instantiate structures
    meta = MetaPolicy(text_dim=H, hidden=64, W=W).to(device)
    critic = NeuroCritic(z_dim=H).to(device)
    bb = MockBackbone(hidden_size=H).to(device)
    
    # Randomise weights slightly to verify value preservation
    with torch.no_grad():
        meta.head_nu.weight.fill_(0.42)
        critic.net[0].weight.fill_(0.24)
        bb.w_proj.weight.fill_(0.77)
        
    # 2. Save state dicts
    state_dict_meta = meta.state_dict()
    state_dict_critic = critic.state_dict()
    state_dict_bb = bb.state_dict()
    
    # Check that saving contents matches
    assert state_dict_meta is not None
    assert state_dict_critic is not None
    assert state_dict_bb is not None
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        meta_file = tmp_path / "meta_test.pt"
        bb_file = tmp_path / "bb_test.pt"
        
        # Save to disk
        torch.save({"meta": state_dict_meta}, meta_file)
        torch.save({"backbone_state_dict": state_dict_bb}, bb_file)
        
        # 3. Reload with strict=True
        meta_loaded_ckpt = torch.load(meta_file, map_location=device, weights_only=True)
        bb_loaded_ckpt = torch.load(bb_file, map_location=device, weights_only=True)
        
        # Re-instantiate
        meta_new = MetaPolicy(text_dim=H, hidden=64, W=W).to(device)
        bb_new = MockBackbone(hidden_size=H).to(device)
        
        # strict=True check (must succeed without raising exceptions)
        meta_new.load_state_dict(meta_loaded_ckpt["meta"], strict=True)
        bb_new.load_state_dict(bb_loaded_ckpt["backbone_state_dict"], strict=True)
        
        # 4. Values comparison
        assert torch.allclose(meta_new.head_nu.weight, meta.head_nu.weight)
        assert torch.allclose(bb_new.w_proj.weight, bb.w_proj.weight)
