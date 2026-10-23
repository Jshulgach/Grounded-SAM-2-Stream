# Contributing to Grounded Stream

Please open an issue describing a reproducible bug or proposed feature before a large change. Use synthetic fixtures or consented sample video; do not include household footage, tokens, camera passwords, or private network configuration.

1. Create a branch from main.
2. Install the service and dev extras in a virtual environment.
3. Run pytest, Ruff, and the wheel build documented in README.
4. Add tests for failure, cancellation and recovery when changing runtime behavior.
5. Update API contracts and migration notes when changing observable behavior.
6. State exactly which real devices and models were tested; mock tests do not prove live-camera behavior.

Keep the base package importable without Torch, OpenCV, Transformers or an LLM server. Keep framework dependencies optional and preserve independent use outside Homebase.

This project does not require the unrelated Facebook CLA referenced by the original template. Contributions follow the repository LICENSE; preserve existing third-party notices.

For security or conduct issues, contact a repository maintainer using the contact method on their GitHub profile. If private vulnerability reporting is enabled, use the repository Security tab. Do not publish private recordings, credentials or exploit details in an ordinary issue.
