# Reproduction

## Offline numeric replay

From the root run `python tools/reproduce.py` and `python tools/verify_release.py`. Both are standard-library-only and make no field solves or model requests. Expected outputs are committed under verification.

## Geometry and regression checks

Use an isolated Python 3.11 environment. Installation metadata are supplied in pyproject.toml; observed versions are in environment/requirements-recorded.txt. Install the repository editable with its research extra in that separate environment. This distribution supplies the recorded quantum-metal library, so do not install over an unrelated working environment.

Set PYTHONNOUSERSITE=1, PYTHONUTF8=1, QISKIT_METAL_HEADLESS=1, and PYTHONPATH to the repository root plus src. See verification/commands.txt for the release-time test invocation.

## Native fields

Run `python tools/extract_projects.py` to restore original projects at their repository-relative paths. Every file is hash-checked; a different existing file is never overwritten. Open with a separately licensed Ansys Electronics Desktop 2022 R1. Full mesh caches are excluded; a fresh solve generates them. Matrix histories, exported matrices, and native profiles preserve recorded evidence.

## Explicit new model searches

Use `python -m robust_design_lab.experiments.cross_model --help` for the original Qwen/Kimi driver. It requires an output directory and successful image preflight records. Use `python -m scripts.cross_model_gpt.batch --help` for the recorded GPT driver. It additionally requires original-batch records and successful paired image checks.

The Alibaba adapter reads a local credential CSV/XLSX specified by ROBUST_ALIYUN_CREDENTIALS. Compatible-service and GPT adapters accept a local two-line HTTPS endpoint/key file through their credentials option. Store credentials outside the repository. Published endpoint metadata are redacted; the request profiles, model identifiers, visible responses, exact input images, and usage remain.

Start new runs into new output directories. Released original source identities are provenance; privacy sanitization changes JSON hashes. Original and published file hashes are recorded separately. Paid model calls and field solves require explicit user action and external access.
