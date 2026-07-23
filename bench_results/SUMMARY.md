# Empirical throughput — Arc Pro B60 / Qwen3.6-27B-MTP Q4

Method: `POST /completion`, `cache_prompt:false`, metrics from `timings.*_per_second`, ~1k prompt, `n_predict=64`.

**Caveat:** early rows used best-of-2. Gen tok/s under MTP is noisy (~20% spread on identical production config over 5 runs: ~19–24 tok/s) and prompt-sensitive. Treat absolute per-row gen numbers as approximate; relative gaps only mean something when well outside that noise band (e.g. n_max 5–6 dropoff). Prefer `--repeats 5+` going forward (`REPEATS` default is now 5).

| config | prompt tok/s | gen tok/s | ctx |
|---|---:|---:|---:|
| A_prod_manual | 340.50 | 16.86 | 32768 |
| B_fit_on_minimal | 339.33 | 20.03 | 120832 |
| C_fit_on_keep_batch | 339.58 | 20.98 | 120832 |
| D_fa_auto_sycl | 336.62 | 19.89 | 32768 |
| E_fa_off_f16v | 339.59 | 24.59 | 32768 |
| F_batch_b512 | 337.81 | 16.89 | 32768 |
| F_batch_b1024 | 340.20 | 20.78 | 32768 |
| F_batch_b2048 | 337.87 | 16.21 | 32768 |
| F_batch_b4096 | 337.56 | 19.24 | 32768 |
| G_mtp_nmax1 | 340.82 | 19.65 | 32768 |
| G_mtp_nmax2 | 340.57 | 18.89 | 32768 |
| G_mtp_nmax3 | 340.24 | 19.46 | 32768 |
| G_mtp_nmax4 | 336.31 | 19.57 | 32768 |
| G_mtp_nmax5 | 333.59 | 13.72 | 32768 |
| G_mtp_nmax6 | 334.86 | 15.25 | 32768 |
| H_no_mtp | 374.96 | 16.30 | 32768 |
| I_ctx131k_fa_on | 338.78 | 16.46 | 131072 |
| J_ctx131k_fa_off_f16v | ERR | ERR | — |

## Conclusions (reconciled)

- **Prompt-eval** is ~flat (~334–375) across fit/manual/batch; no-MTP is the main prompt bump (~375).
- **draft-mtp n_max 1–4**: good band; **n_max 5–6 regresses gen** well outside the ~20% same-config noise band. **Keep auto draft-mtp; pin n_max=3; warn if >4.**
- **Flash-attn on SYCL is NOT required** for q8 V-cache. Production serves q8 V with no `--flash-attn` flag. Earlier "hard abort" conclusion conflated Vulkan (or explicit `-fa off`) onto SYCL — **claim dropped**. FA inject remains **Vulkan-only** for quantized KV.
- Gen absolute tok/s in best-of-2 rows is **not precise** (same prod config can run ~19–24). Do not rank configs on small gen gaps without ≥5 repeats.
- fit on does not beat manual on prompt; it auto-sizes larger ctx.
- batch `-b` sweep: no clean winner under this noise model — no default change.
