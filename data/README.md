# Data

Datasets and trained checkpoints are intentionally not committed. This keeps
the repository code-only, matching the upstream project organization.

Use the nominal `.fjs` files and the health-overlay generator with
`train_ramp.py --train-dir ... --validation-dir ... --test-dir ...`. The
industrial Steel-FJSP bundle must be placed at the path expected by
`ramp/steel_data.py`; its manifest is checked before evaluation.
