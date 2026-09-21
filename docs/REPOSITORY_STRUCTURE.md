# Repository structure

This repository keeps the Image2EditablePPT source, tests, schemas, and operational documentation together so future work can happen from one GitHub checkout.

```text
backend/     FastAPI service, OCR/CV/Vision pipeline, reconstruction, PPTX export, tests
frontend/    Vite + React editor, canvas, settings, comparison view, and export UI
shared/      Scene and layout schemas shared by backend and frontend
docs/        Architecture, API, development, references, and repository guidance
scripts/     Local setup, start, and test helpers
```

The following folders remain local runtime storage and are intentionally not versioned:

- `uploads/` — user-uploaded source images
- `outputs/` — generated PPTX, renders, scene graphs, and QA artifacts
- `temp/` and `work/` — transient processing data
- dependency folders, Python caches, model caches, and log files

API keys are stored by the application in the operating-system settings directory, not in this repository. `.env.example` contains placeholders only.
