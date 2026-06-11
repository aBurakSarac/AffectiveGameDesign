# Archived Pipeline Scripts

These scripts were active during Phase 1 / early Phase 2 (Mar 15-22, 2026).
They are superseded by newer tools but preserved for reference.

Archived: Mar 27, 2026

| Script | Superseded by | Notes |
|---|---|---|
| `compare_3tools.py` | `formula_benchmark.py` | Hardcoded to S01 paths, MP+HS+DF only |
| `compare_all_sessions.py` | Per-session annotation workflow | Used old sessions.json registry |
| `compare_fear_moments.py` | `formula_benchmark.py` | Hardcoded to S01 game paths |
| `run_all_tools_sequential.py` | `test_mp_hs.py` + `test_fusion.py` | Sequential 4-tool runner |
| `eval_ground_truth.py` | `formula_benchmark.py` | Old P/R eval against 36-event GT |
| `calculate_missing_stats.py` | One-time utility | Stats for Mar 15 CSVs (job done) |
| `emotion_comparison_report.py` | `formula_benchmark.py` | One-time fear feature comparison |
| `improved_fear_detection.py` | `test_mp_hs.py` compute_tension() | Prototype for dual-type detection |
| `update_march15_metadata.py` | One-time utility | Interactive metadata fix (job done) |
| `validate_hs_crop.py` | Validation complete | Haar->MP crop r=0.935 (Mar 23) |

To reuse any script, move it back to `Pipeline/` (imports assume that location).
