# Milestone and Maintenance Checklist

## Completed

- [x] M2.1: Core pipeline and baseline explainability.
- [x] M2.2: Workforce analytics and expanded tests.
- [x] M2.3: Consistency fixes and dashboard integration.
- [x] M2.4: Final baseline and privacy-safe export behavior.
- [x] M2.5: Bounded optimization, comparison, persistence, and version selection.
- [x] M2.6: Monitoring, drift detection, status, history, exports, and browser evidence.
- [x] M2.7: Deployment configuration, local startup docs, health, dependencies, and release readiness.

## Post-2.7 Backlog

- [ ] Add CI workflow for pytest, Ruff, compilation, and secret scanning.
- [ ] Add lock/version pinning for fully reproducible hosted builds.
- [ ] Add automated browser smoke tests for monitoring upload and run.
- [ ] Add persisted upload-file size limits and clearer ingestion diagnostics.
- [ ] Add optional object-backed monitoring history.
- [ ] Review UI on narrow mobile and tablet layouts.
- [ ] Continue the approved UI/UX professionalization workflow.
- [ ] Add a formal deployment release checklist and rollback runbook.

## Repository Cleanup Record

- M2.1-M2.7 implementation knowledge is consolidated into this docs set.
- Root milestone plans and QA reports are historical artifacts, not runtime inputs.
- Browser evidence and synthetic monitoring files are release-validation evidence and should be regenerated or stored outside the source tree if needed later.
- The original reference notebook is experimental source material and is not imported by runtime or tests.
