"""tests/test_s6_damping.py - Unit test for S6 Damping Factor Pipeline
Directive: EXE-260926-36 Task 2 (System 2 / 원장 #348)

Requirement:
- 단위 테스트 1건: alpha 인자가 compute_zstar 호출 kwargs에 그대로 전달.
- Trajectory 4-way classification sanity tests.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

BASE_S5_DIR = "/home/shoon/cts_e1_s5"
SYS_DIR = "/home/shoon/cts_e1"
SCRATCH_DIR = "/home/shoon/cts_e1/scratch"

for p in [BASE_S5_DIR, SYS_DIR, SCRATCH_DIR, os.path.join(SYS_DIR, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.s6_damping_run import classify_trajectory_4way, run_damping_experiment


class TestS6Damping(unittest.TestCase):
    def test_alpha_passed_to_compute_zstar_kwargs(self):
        """Verify that alpha argument is passed directly into compute_zstar kwargs."""
        test_alpha = 0.35

        mock_zstar = torch.zeros(64, 256)
        mock_rho_fp = 0.001
        mock_nfe = 5
        mock_log = {
            "rho_traj": [0.01, 0.008, 0.005, 0.003, 0.001],
            "solver": "picard",
            "checkpoints_100": {},
        }

        # Mock get_model and get_dataset so no GPU/weights are loaded
        mock_bb = MagicMock()
        mock_bb.routing_matrix.return_value = torch.zeros(10, 10)
        mock_dataset = {31: {"prompt": "Solve 2+2"}}

        with patch("scripts.s6_damping_run.get_model", return_value=mock_bb), \
             patch("scripts.s6_damping_run.get_dataset", return_value=mock_dataset), \
             patch("scripts.s6_damping_run.compute_zstar", return_value=(mock_zstar, mock_rho_fp, mock_nfe, mock_log)) as mock_cz:

            dummy_out = "/tmp/test_s6_out.pt"
            dummy_raw = "/tmp/test_s6_raw"

            res = run_damping_experiment(
                alpha=test_alpha,
                items=[31],
                out_pt_path=dummy_out,
                raw_dir=dummy_raw,
                config_path=os.path.join(BASE_S5_DIR, "configs", "zstar_canonical.yaml") if os.path.exists(os.path.join(BASE_S5_DIR, "configs", "zstar_canonical.yaml")) else "dummy.yaml",
                device_str="cpu",
                max_nfe=5,
                tol=0.0,
                resume=False,
                run_id="test_unit",
            )

            # Assert compute_zstar was called
            self.assertTrue(mock_cz.called)
            call_kwargs = mock_cz.call_args.kwargs

            # Verify alpha parameter is strictly in kwargs and equals test_alpha
            self.assertIn("alpha", call_kwargs)
            self.assertEqual(call_kwargs["alpha"], test_alpha)
            self.assertEqual(call_kwargs["tol"], 0.0)
            self.assertEqual(call_kwargs["max_nfe"], 5)

            # Clean up dummy files
            for p in [dummy_out, dummy_out.replace(".pt", "_summary.json"), os.path.join(dummy_raw, "s6_a0.35_test_unit.log")]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    def test_classify_trajectory_4way(self):
        """Verify 4-way trajectory classification logic."""
        # 1. Monotonic decrease
        traj_dec = [float(1.0 / (i + 1)) for i in range(200)]
        self.assertEqual(classify_trajectory_4way(traj_dec), "monotonic_decrease")

        # 2. Increase (상승)
        traj_inc = [float(0.01 * (1.0 + 0.01 * i)) for i in range(200)]
        self.assertEqual(classify_trajectory_4way(traj_inc), "increase")

        # 3. Oscillation
        traj_osc = [0.01 + 0.005 * ((-1) ** i) for i in range(200)]
        self.assertEqual(classify_trajectory_4way(traj_osc), "oscillation")

        # 4. Plateau
        traj_plat = [0.005 + 1e-6 * (i % 2) for i in range(200)]
        self.assertEqual(classify_trajectory_4way(traj_plat), "plateau")


if __name__ == "__main__":
    unittest.main()
