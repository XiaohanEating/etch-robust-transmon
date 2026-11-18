# Etch-robust transmon capacitors

Core code, electromagnetic simulation records, model interactions, and native CAD projects for reducing transmon frequency sensitivity to capacitor etch bias. The manuscript and publication figure files are excluded.

## Contents

- Thirty complete searches: Qwen3-VL-Plus, Kimi-K2.5, and the endpoint-reported GPT-6-Astra configuration; two starting families; five repeats each; nine attempted candidate slots per search.
- Actual top-view model input images, requests, visible replies, candidate parameters, rejection records, capacitance matrices, adaptive-pass histories, convergence flags, and retained geometries.
- Five historical classical baselines, shape exploration, calibrated annular and edited candidates, matched references, stricter-convergence checks, and four held-out points per reference.
- A separately identified nominal common-readout Purcell pilot.
- The 374-file recorded implementation, library resources, regression tests, and 990 original native Ansys projects with individual hashes.

**These data are electromagnetic simulations and analytic predictions. No fabrication or laboratory device-measurement dataset is included.** Historical names containing measured refer to solver observations, not fabricated devices.

## Offline reproduction

Run `python tools/reproduce.py`, then `python tools/verify_release.py` from the repository root. Both use the Python 3.11 standard library, without API calls or Ansys. Outputs go to `verification/`. The first recomputes capacitance, frequency, drift, final selection, medians, structural comparisons, and held-out shifts. The second verifies provenance hashes, 90 actual input images, native projects, and publication exclusions.

| Configuration | Double-pad median reduction | Interdigitated median reduction |
|---|---:|---:|
| Qwen | 25.1% | 25.8% |
| Kimi | 22.1% | 48.2% |
| GPT configuration | 32.7% | 54.6% |

Edited annular drift is 0.002507 at the 0.1% extraction target, 40.6% below matched double-pad and 85.6% below matched interdigitated. The calibrated unedited ring provides nearly all of the gain. Its small difference from the edited candidate remains unresolved.

## Repository layout

- `robust_design_lab/`: geometry, search, extraction, circuit analysis, tests, and selected raw result records.
- `scripts/cross_model/` and `scripts/cross_model_gpt/`: recorded service adapters and drivers.
- `src/qiskit_metal/`: recorded local Metal implementation and runtime resources.
- `data/`: numerical index, structural provenance, solver exports, and native projects.
- `environment/`: observed package versions and external solver requirements.
- `provenance/`: original and published hashes, plus transformations.
- `tools/`: offline replay, validation, and project extraction.

Full field reproduction requires separately licensed Ansys Electronics Desktop 2022 R1 on Windows and user-owned model service access. No solver software, model weights, API credentials, or complete adaptive mesh caches are redistributed. See [reproduction instructions](docs/reproduction.md).

## Evidence boundaries

Historical classical comparisons and current model searches have different protocols and must not be pooled. Candidate budgets are shared across current model configurations; inference settings differ. Model names identify service-reported configurations, and no fixed API service seed was used. Strict visual-input attribution and complete-device frequency/Purcell joint qualification remain unfinished. No measured T1 or total coherence-time equivalence is claimed. See [data and limitations](docs/data_and_limits.md).

Apache-2.0; see LICENSE and NOTICE. Original library notices are retained. One historical scheduler replaces its workstation interpreter path with sys.executable; other frozen source hashes are retained. JSON metadata are sanitized while scientific numbers and visible proposals remain. This is an identified public repository; any anonymous review link must be prepared separately.
