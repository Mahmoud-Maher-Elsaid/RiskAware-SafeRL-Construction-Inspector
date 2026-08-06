# HRMPPO v2 horizon protocol correction

The v2 configuration declares a predictive Safety Shield horizon of 5, while the historical trainer and evaluator defaults were horizon 1. The defaults are now aligned to the declared configuration and covered by `tests/scripts/test_hrmppo_v2_protocol.py`.

The strongest preserved v2 checkpoint was re-evaluated on 270 held-out episodes (30 seeds × 3 worlds × 3 profiles) with horizon 5 and step budget 1.0. The corrected run verified CUDA inference and zero invalid actions, but mission success collapsed to 0.0 while mean safety cost fell to 12.2898. This is a protocol-corrected negative result, not a replacement approval.

Historical horizon-1 reports remain immutable. No gate was lowered and no checkpoint or demonstration was overwritten.

A 1,000-step CUDA smoke and a resumed 1,500-step continuation also passed with horizon 5 and the 8 GB memory guard. These are implementation/resume evidence only, not replacement-performance evidence.

A distinct 30,000-step horizon-5 recovery from the strongest preserved horizon-1 checkpoint also completed safely, but its 270-episode paired evaluation produced 0.0 mission success, 0.20262 recall/coverage, and 10.9741 mean safety cost. It is therefore preserved as a failed recovery candidate and is not installed or authorized for production.
