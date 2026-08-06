# HRMPPO v2 horizon protocol correction

The v2 configuration declares a predictive Safety Shield horizon of 5, while the historical trainer and evaluator defaults were horizon 1. The defaults are now aligned to the declared configuration and covered by `tests/scripts/test_hrmppo_v2_protocol.py`.

The strongest preserved v2 checkpoint was re-evaluated on 270 held-out episodes (30 seeds × 3 worlds × 3 profiles) with horizon 5 and step budget 1.0. The corrected run verified CUDA inference and zero invalid actions, but mission success collapsed to 0.0 while mean safety cost fell to 12.2898. This is a protocol-corrected negative result, not a replacement approval.

Historical horizon-1 reports remain immutable. No gate was lowered and no checkpoint or demonstration was overwritten.
