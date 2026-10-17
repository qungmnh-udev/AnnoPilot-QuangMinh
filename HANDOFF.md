# AnnoPilot Development Handoff

## Training UI and automatic checkpoint integration — 2026-10-16

User explicitly expanded scope to in-app model training, full retirement of YOLOv8 and CPU/CUDA support. This supersedes the historical no-training scope below. See `doc/training.md`.

Model QC now has a compact TrainingPanel (model, presets, reviewed-label confirmation; optional technical settings). Backend persists jobs/progress in SQLite and launches an isolated worker process, with cancel, heartbeat/dead-worker recovery and a unique global training slot. Supports YOLO26n/s, RT-DETR-l and Faster R-CNN. Data snapshots exclude registered Golden images and byte-identical duplicates, split train/val without overlap, and validate class/geometry. A completed checkpoint is reloaded/inferred, then registered automatically for that dataset/family; source predictions are marked STALE and must be rerun. Registry selection is used by YOLO queue/frame QC, triad and optical experiments. Original weights and older checkpoint versions remain selectable. Do not count validation metrics as independent Golden quality.

Removed old training/export scripts, YOLOv8 weights (including historical train-run weights) and ONNX decoder. Imports/evaluation/ensemble reject legacy model sources; existing audit/history DB rows are not relabeled or deleted. All three inference adapters now use CPU/CUDA selection, including moving TorchVision model/input to the actual device.

Real isolated smoke passed on CPU for all four supported model choices: one training epoch, export, registration, and actual QC inference using the generated checkpoint/hash. Artifacts: `.runtime/training-smoke-a46570e2-d006-4938-b6c0-5ff7dfa21dfa/smoke.json` (n/RT-DETR/Faster R-CNN) and `.runtime/training-smoke-d4ed6676-1558-4a14-8b80-7cfae94c274b/smoke.json` (s). Synthetic fixtures are technical feasibility evidence only, not model-quality validation.

NVIDIA RTX 3050 Laptop 4096 MiB is present, but current torch is `2.14.1+cpu`. Outbound pip connection to download.pytorch.org is blocked with WinError 10013; no CUDA wheel installed or real GPU training verified. `Install-GPU.ps1` installs a CUDA wheel pair and verifies a GPU tensor when run outside that network restriction. Do not claim GPU verification from device-routing mocks.

Validation: 120 backend tests passed and final frontend production build passed; 13 training/device tests were rerun successfully after the final explicit-device adjustment. Summary: `backend/benchmarks/training-smoke-summary.json`. Browser HTTP preview was reachable in an isolated fixture, but Edge renderer could not complete page initialization in this sandbox; no full browser training-flow pass is claimed. Temporary preview servers/browser were stopped. GPU setup PowerShell syntax passed; CUDA install remains blocked by network access.

Last verified: 2026-10-16, Asia/Saigon.

## CVAT two-way flow audit and corrections (2026-10-15)

See `doc/cvat-data-flow.md` for the exact workflow and limits. Initial dataset/media import is still manual CVAT images XML + images/ZIP upload. Live sync updates existing samples only. The isolated perturbation experiment reports do not feed operational Prediction rows; use ordinary WBF before pushing pre-labels.

Added shared `services/cvat_api.py`: validates Job ownership/range, reads Task frame metadata, maps image names uniquely, checks image dimensions, and follows label pagination. Task scope stays Task scope; push no longer selects the first job. Segment IDs are no longer parsed as Job IDs; only explicit job metadata/URLs qualify. Config can clear a Job ID and partial updates no longer reset the server URL.

Live attributes become dicts with names; raw shape IDs/attributes remain in metadata. File sync recognizes declaration-free XML, rejects empty/incomplete/unmatched snapshots, matches names rather than unrelated numeric frame IDs, and distinguishes file snapshots from live verification in origin/review notes. Sync/push validation errors return 422; server failures return 502 with database rollback.

Append remains create-only: exact duplicate boxes are skipped; known local Accept wrong-class changes are rejected if the old class box is still on CVAT. Replace overwrites unrotated rectangles only for selected dataset frames, preserves other frames/non-rectangle shapes/tags, and refuses video-track round-trip. Pre-label push rejects unfused multiple detectors and stale/inactive fused provenance; synthetic/inactive raw sources are excluded. Reviewer edits in CVAT -> Save -> live pull/verify is the preferred correction flow. Pull before making local edits, not over local edits that have not been pushed.

UI now contains a CVAT flow explainer, clearer source/mode names, and explicit manual-resolution labels. Full backend suite: **103 passed**. Frontend TypeScript/Vite build passed to `.runtime/build-cvat-final-a`. New regression tests use a stateful fake CVAT to exercise push -> simulated remote edit -> pull -> retry, mapping, labels, attributes, partial replace, track rejection and config clearing. **No real team CVAT server round-trip or external push was performed in this audit.** No production dataset was changed. Restart the API to use the new code; verify with a designated demo Task before marking live CVAT QA complete.

## Real WBF and perturbation experiment (2026-10-15)

User prioritized real Multi-Model WBF, repeated inference, and Gaussian + brightness + mild blur. Added an isolated experiment service, validated request schema, async API/job reports, a panel at the top of the Multi-Model WBF modal, and `backend/scripts/benchmark_wbf_perturbations.py`. See `doc/wbf-perturbation-experiment.md` for protocol and commands.

Six arms compare one/multiple physical detectors under one original pass, repeated clean passes, and original + perturbed passes. Default 4 views, sigma 8, brightness +15%, blur radius .7, seed 42, confidence .35, fusion IoU .55, equal model weights. Fuse temporal views per physical model before cross-model fusion; identical repeated outputs preserve distinct overlapping detections. Repeated views never become independent detector votes. Added a lock around cached triad inference to avoid mutable-predictor races.

Experiment jobs snapshot the selected images/annotations/optional independent Golden, save raw per-view predictions/provenance/config/timings and metrics under `.runtime/wbf-experiments`, and never change operational annotation/prediction/issue rows or push CVAT. Reports persist; active jobs do not resume after backend restart. Without independent Golden, quality metrics remain null; agreement with the single-pass baseline is consistency only. CLI without annotation snapshot does not invent QC errors.

Verified: standard `/ensemble/run-triad` executed real YOLO26n/Faster R-CNN/RT-DETR on one image in an isolated SQLite database, persisted all three physical sources plus WBF (116 raw detections, 62 fused, default triad weights). Experiment POST/status/report endpoints also completed six real-inference arms on one image in another isolated database. CPU checkpoints/dependencies were already present. Production database was not opened by these smoke scripts.

Software validation after the identical-pass fix: full pytest **89 passed**. Used workspace-local `--basetemp` and `-p no:cacheprovider` because default user Temp/cache paths denied access. TypeScript/Vite production build passed with `npm.cmd run build -- --outDir ../.runtime/build-wbf-20261015`; default `dist/assets` cleanup was denied by existing filesystem permissions. No browser automation was available; UI was compiled, real execution was verified through API and CLI. Restart API before using new endpoints; no servers were left running by these checks.

Final real run completed on 15 demo images: **540 measured detector calls + 3 warm-up calls**. Single original / repeated clean / repeated perturbed produced **277 / 277 / 290** boxes; multi-model WBF original / repeated clean / repeated perturbed produced **1022 / 1022 / 1148** boxes. Clean-repeat predictions exactly matched their single-original controls for all 15 frames. Raw report: `.runtime/wbf-experiments/demo-real-15-final.json`; compact manifest/summary: `backend/benchmarks/wbf-perturbations-summary.json`; interpretation: `doc/wbf-experiment-results.md`. Times were measured during the working session with some early concurrent checks, so treat them as operational observations. This is inference feasibility/consistency evidence, not independent Golden quality validation. Ordinary queue still has no perturbation stage, durable resume, per-frame automatic CVAT issue push, or export-blocking Quality Gate.

## Current direction and implementation

The user approved Model-assisted QC for missing objects and wrong classes: pre-label → human edit in CVAT → prioritized QC → review/sync → acceptance. CVAT is the annotation source of truth. Baseline is one pretrained YOLO26n (default) or YOLO26s inference; pilot uses the same selected YOLO plus other detectors for cross-verification. YOLOv8 predictions remain historical evidence with their original IDs. WBF generates proposed pre-labels; consensus is not ground truth. The previous difficulty/sampling direction below is historical.

Implemented in this change:

- Shared class-agnostic candidate matching, one vote per model, retained original evidence, weighted class selection and strict Tier 1 thresholds.
- QC reads original predictions and retains disagreement even when annotation matches fusion; unmatched candidates remain eligible for Missing Object QC.
- Golden Benchmark/evaluator API and UI: paired baseline/pilot runs on identical input snapshots, per-error recall, precision/F1, separate class suggestion accuracy, measured review efficiency and inference/QC durations.
- Persisted acceptance events (submit/return/resubmit/accept) and issue review history. Retry does not duplicate accepted boxes; reviewed decisions survive prediction replacement by spatial reconciliation.
- Per-frame commits during multi-model inference, UI polling, explicit inference error/coverage reporting; synthetic predictions are restricted to sandbox datasets and never used as real evaluation evidence.
- CVAT push excludes pending/false-positive additions, prioritizes fused pre-labels, deduplicates exact existing shapes, refuses arbitrary class fallback. Live verification requires current CVAT data, preserves resolved issue history, normalizes rectangles and interpolates track keyframes. CVAT image IDs are retained in new imports.
- Additive database migration, optional model dependencies in `backend/requirements-models.txt`, updated README and Vietnamese guide.

Verification: **76 backend tests passed** (one existing Starlette/httpx deprecation warning), frontend `npm.cmd run build` passed, `git diff --check` passed. Tests use temporary storage; the production `annopilot.db` remained unchanged. Actual YOLO26n/s CPU inference was benchmarked on 15 demo images, and an actual YOLO26s inference through the QC API passed on a temporary DB. Full three-model inference, browser interaction and live CVAT server verification were not performed; other orchestration tests use mocks.

YOLO26 migration: official `.pt` weights downloaded to `backend/models`, ignored in Git. Shared YOLO service resolves paths independent of cwd, serializes Predictor access and selects CUDA only if installed PyTorch supports it. UI dropdown selects n/s for queue, frame audit, detector and ensemble M1. Other YOLO variants become inactive without deleting historical predictions; automatic QC/ensemble excludes inactive sources. Optional model requirements now require Ultralytics >=8.4.174.

Hardware: i5-12450H, 15.71 GiB RAM, RTX 3050 Laptop 4 GiB. Installed torch is `2.14.1+cpu`, so GPU remains unused. Measured CPU mean at imgsz 640: YOLO26n 85.40 ms/image, YOLO26s 165.08 ms/image; isolated process peak RSS 395.6/481.7 MiB. Report: `backend/benchmarks/yolo26-hardware.json`; protocol and rerun command: `doc/yolo26-hardware.md`. These numbers measure inference feasibility, not QC recall/precision or full ensemble throughput.

Restart the backend to initialize new tables/column. Read `doc/project-direction.md` and `doc/huong-dan-su-dung.md` for the current protocol. Remaining work: independently verify a real golden set, run quality experiments, calibrate risk/thresholds and measure review/acceptance outcomes. Durable queue resume, automatic per-frame CVAT push/issues, TTA in the operational queue and export-blocking Quality Gate remain future work; the separate real-model perturbation experiment above is now implemented.

## Historical handoff (2026-10-03)

## 1. Project Goal
Dataset → Import/normalize → Media + Annotation Analysis → Difficulty → Dataset Health → Smart Sampling → Workload → Reviewer Priority → Human Review → CSV Export.
Difficulty estimates review effort/complexity, never correctness. Human review alone records No Issue or Issue Found. Production data must come from user uploads.

## 2. Working Directory
F:\VinAI\Workspace\Tool\AnnoPilot. Initially empty; no existing source, handoff or project/inspected-parent AGENTS instructions found. Project built directly here, no nested checkout. Runtime paths are portable.

## 3. Architecture
React19/Vite6/TypeScript/Tailwind4/Recharts/Lucide frontend. Python/FastAPI/Pydantic/SQLAlchemy/SQLite backend. Local Python3.9.6; Docker Python3.11. Flexible JSON geometry, four parser strategies and four BaseDifficultyAnalyzer strategies. Shared deterministic sampling, greedy effort balancing, independent reviewer queues. Synchronous local CPU analysis; no external AI/cloud/job infrastructure.

## 4. Current Project Structure
- frontend/src: App.tsx, api.ts, types.ts, main.tsx, style.css
- frontend: package.json/lock, vite.config.ts, tsconfig.json, postcss.config.js, index.html, Dockerfile, nginx.conf
- backend/app: main.py, database/, models/, schemas/, parsers/, analyzers/, services/pipeline.py, routers/api.py
- backend/tests: conftest.py, test_parsers.py, test_analysis.py, test_api.py, ui_fixture/labels.xml and README.md
- backend: requirements.txt, pytest.ini, Dockerfile
- root: README.md, HANDOFF.md, Start-AnnoPilot.ps1, docker-compose.yml, .gitignore
- uploads/, exports/: gitkeep plus ignored user runtime files
Ignored: venv, node_modules, npm cache, builds, databases, test runtime stores, logs, uploads/exports.

## 5. Implemented
- Unified dataset/sample/annotation/analysis model with BBOX_2D, POLYGON, MASK, KEYPOINT, CUBOID_3D and MIXED tasks. Attributes, nullable occlusion and source metadata preserved: models/ and parsers/.
- CVAT, COCO, YOLO detection, KITTI strategies; invalid individual shapes warned/skipped; malformed/empty documents handled.
- Multipart/ZIP uploads, bounded extraction, traversal/symlink rejection, format/task detection, media matching, missing media retained: services/pipeline.py, routers/api.py.
- OpenCV blur/exposure/contrast/edges/entropy, task-specific annotation features, availability/reasons, weight renormalization and actual contributions: analyzers/.
- Configurable weight groups and thresholds; save recalculates all datasets atomically. Revision mismatch exposes Needs Regeneration for existing plans.
- Seven connected frontend pages: real statistics/charts, filters/sort/20-row client pagination, scaling SVG box/polygon/keypoint/label overlays, properties and scoring explanation.
- Count/percentage seeded QC selection, deduplication, explicit coverage fallback; reviewer creation/removal guards; greedy QC-subset balancing; independent priority queues.
- No Issue, Issue Found, Skip, persisted notes, next PENDING sample. Historical results retained across regeneration.
- Three UTF-8-BOM CSV exports with requested columns and formula-text escaping.
- Dataset deletion confirmation, cascades, complete upload-folder cleanup via additive storage_key migration.
- Dockerfiles/Compose/Nginx, local hidden-process start script and comprehensive README.

## 6. Partially Implemented
| Feature | What works | Remaining / next action / files |
| --- | --- | --- |
| COCO RLE masks | Original RLE retained; count/rarity available | Validated decoder, geometry metrics and overlay. parsers/, analyzers/, App.tsx. Never claim current rendering. |
| 3D sensors | KITTI parse/score/QC/metadata review; BIN/PCD retained/matched | Decode point cloud, calibration transforms/density and optional viewer. parsers/, analyzers/, pipeline.py, App.tsx. |
| Import progress | Busy/loading messages, disabled actions | Optional actual stage progress/background job for large imports; no invented percentages. |
| Large dataset browsing | Fetch all summaries, paginate20 client rows | Server-side filtering/pagination if needed. |
| Docker verification | Compose config validates, proxy/healthcheck/volumes implemented | Unpause engine, free3000/8000, build/start and smoke-test. |
These limits are described accurately in README/UI; core local workflow works.

## 7. Not Started
Future/non-goals: CVAT video tracks/skeletons/raster masks, YOLO pose/segmentation, articulated pose viewer, advanced 3D viewer, login/RBAC and internet deployment hardening. Do not add auto-label/model training/chatbots to MVP.

## 8. Supported Annotation Tasks
| Task | Status | Notes |
| --- | --- | --- |
| BBox2D | TESTED | CVAT/COCO/YOLO, tiny/overlap/known occlusion/rarity |
| Polygon segmentation | TESTED | CVAT/COCO, area/vertices/envelope crowding; RLE partial |
| Keypoint | TESTED | CVAT points/COCO visibility; unknown visibility stays unknown |
| Cuboid3D | TESTED | KITTI geometry/metadata/available scoring; calibrated sensor features unavailable |
| Mixed | TESTED | Per-task analyzers combined by annotation counts |

## 9. Supported Formats
| Format | Status | Tasks | Limits |
| --- | --- | --- | --- |
| CVAT images XML | TESTED | Boxes/polygons/points/mixed | Attributes/occlusion; video tracks unsupported |
| COCO JSON | TESTED | Boxes/polygons/keypoints | RLE retained without decoder; multiple rings count as separate normalized shapes |
| YOLO TXT | TESTED | Detection BBox | Five fields, matching image required, optional classes.txt |
| KITTI TXT | TESTED | Cuboid3D |15+ fields; rectified camera frame/bottom-center preserved; DontCare excluded |
Unit parser and multipart upload/analysis integration tests exercised all implemented formats. JPG/JPEG/PNG decoded; BIN/PCD retained but not decoded/visualized.

## 10. Difficulty Implementation
Source: backend/app/analyzers/__init__.py. Defaults/validation: backend/app/schemas/__init__.py.
Each feature: normalized[0,1] value, raw, available, reason_if_unavailable. Score =100*available weighted sum/available weight sum. Missing is excluded, not zero. All missing =None.
Visual: Laplacian blur1/(1+variance/100) .30; brightness deviation .25; low contrast .20; Canny density/.2 .15; histogram entropy/8 .10. Longest image side capped1600 for compute.
BBox: count capped20 .20; tiny .25; known occlusion .20; unique IoU pairs .20; rare .15.
Polygon: count .20; tiny .20; vertices capped50 .25; envelope crowding .20; rare .15. Shoelace area; envelope crowding is a proxy, not polygonIoU. Occlusion metadata informational. RLE geometry metrics unavailable.
Keypoint: count .20; known missing .25; known occluded .25; visible pose-envelope crowding .15; rare .15. Unknown visibility not invented.
Cuboid: count .20; known occlusion .20; source truncation .20; center pairs within5m .15; rare .15; sparsity .10 unavailable. Density/distance/projection have explicit unavailable reasons.
Tiny ratio<.01; overlapIoU>.30; rare frequency<.05 across ALL dataset annotations; configurable. Pairwise metrics above2000 instances unavailable to bound CPU cost.
Mixed score: count-weighted mean of available task scores. Overall .70 annotation/.30 visual, renormalized. Measured empty annotation list scores0. Clamp0–100; EASY<40, MEDIUM<70, HARD>=70, UNAVAILABLE when no available weighted source. Stored contributions reconstruct actual score within rounding tolerance; tested with images and all formats.
Full formula table in README. Never use score as error probability.

## 11. Database
Default root annopilot.db (ignored), persisted on restart. Tables: datasets/samples/annotations/analysis_results/settings/sampling_runs/sample_selections/reviewers/review_assignments. ReviewResult represented by assignment status/note fields. Dataset storage_key enables folder cleanup; additive migration preserves prior schema.
Production API returned[] after final restart; no fixtures/reviewers were seeded. Browser test stores are isolated under backend/tests/ui_fixture; pytest temporary stores are removed on completion. Docker named database volume is separate from local DB. Back up database and uploads together.

## 12. API
Router backend/app/routers/api.py; health/api/health; Swagger/docs. Covers datasets/upload/delete/samples/media/settings/recalculate/sampling/reviewers/balance/assignments/queues/status/notes/exports. README route matrix is accurate. Empty browser media fields tolerated; nonempty non-file media rejected. Invalid requests get readable422/409/404 messages.

## 13. Frontend
App.tsx contains seven views and upload/detail dialogs. api.ts handles backend errors, types.ts payloads, style.css responsive dark navy/cyan layout. SVG overlays scale with image stage. Features/contributions and unavailable reasons exposed. Errors visible within dialogs. Chart code split into separate production chunk.
Chrome browser test used separate frontend3001/API8001 with SQLite/uploads/exports under backend/tests. Verified landing, upload dialog, CVAT fixture import WITHOUT media, real computed dashboard, two-sample QC selection, reviewer addition, balancing, priority queue, Issue Found+note, next sample, persistence after reload, No Issue and queue completion. Production empty landing and all settings groups inspected on3000. No production mock data.
Chrome production tab left open as deliverable at http://localhost:3000. Screenshot saved outside repo in chat visualization folder.

## 14. Tests
Executed 2026-10-03 about11:17 Bangkok:
Command: cd backend; .venv/Scripts/python.exe -m pytest -q
Result:39 passed,0 failed,1.66s.
Command: .venv/Scripts/python.exe -m pip check
Result:No broken requirements found.
Coverage: parsers/finite geometry, malformed imports/ZIP, image matching, visual features/blur, tiny/rarity/unique overlap, polygon/keypoint/cuboid availability, missing-weight renormalization, mixed deterministic scoring/contributions, budget/deduplication/fallback, greedy balancing/unknown-score rejection, all-format uploads, full review/export persistence, saved-results preservation, stale plans and empty optional media fields.
Earlier greedy test expected a tighter-than-guaranteed optimum; corrected to documented LPT behavior (120vs150) and largest-job bound. Empty-media regression exposed multipart string-vs-file handling; fixed. Latest full run has no failures.

## 15. Frontend Build
Final npm run build:PASS, TypeScript+Vite,7.46s. MainJS254.79kB, chart chunk423kB, CSS20.42kB. No oversized/empty chunk warning in final build. Latest npm installation reported0 vulnerabilities after Tailwind4.3.3 upgrade removed vulnerable Tailwind3 braces/micromatch chain. package-lock committed.
Vite/esbuild needed sandbox escalation due spawnEPERM; source build succeeded. No source-build failure remains.

## 16. Known Bugs
No known failing core-workflow tests. Scope limits in sections6/7 and README. No RLE/advanced3D rendering claimed. Synchronous analysis and whole-summary client fetch can be slow at scale. Local trusted-workspace use only, no auth. Reviewer removal/rebalancing blocked if it would erase saved reviews; start new subset to rebalance.

## 17. Current Blocking Issue
Docker runtime verification ONLY. docker version returned 'Docker Desktop is manually paused. Unpause it through the Whale menu or Dashboard.' docker compose config --quiet passed. Did not alter user's pause state/restart their containers.
Exact next action:unpause Docker, free ports3000/8000, docker compose up --build; check health/proxy/Swagger. All local core implementation/install/test/build/run work verified. No approval-review rejection occurred.

## 18. Exact Next Steps
- [ ] After Docker unpause, verify container build/start and routes localhost3000,/api/health,localhost8000/docs. Local MVP already works.
- [ ] Upload user-owned real data and evaluate heuristics at domain scale; user provided no dataset in this session.
- [ ] Optional future:validated RLE/calibrated point-cloud analysis/server pagination.
Completed:backend,parsers,analyzers,frontend,sampling,workload,review/export,configuration,pytest,frontendbuild,localSwagger/proxy,README/handoff. Do not rebuild these from scratch.

## 19. Commands
New machine: .\Start-AnnoPilot.ps1 -Install
Next starts: .\Start-AnnoPilot.ps1
Manual backend:cd backend; python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip; python -m pip install -r requirements.txt; uvicorn app.main:app --reload --host 127.0.0.1 --port 8000.
Manual frontend:cd frontend; npm ci; npm run dev.
Tests:cd backend; .venv/Scripts/python.exe -m pytest -q.
Build:cd frontend; npm run build.
Docker root:docker compose config --quiet; docker compose up --build.
Servers left running:frontend process14984 on3000; API wrapper23832/Python child25024 on8000 at verification. Verify command lines if IDs change before stopping. Test-only servers were stopped. Ignored logs backend/server*.log and frontend/server*.log.

## 20. Important Design Decisions
1. Human judgment/source data authoritative; difficulty never writes findings.
2. JSON geometry plus parser/analyzer strategies keep QC pipeline independent of task/format.
3. Missing visibility/calibration stays unavailable. Available weights renormalize; balancing refuses unavailable effort rather than treating it as zero load.
4. Revisioned sampling/assignment preserves history. Historical review export includes prior runs; sample may appear once per run.
5. Greedy balancing deterministic, not globally optimal. Saved reviews prevent destructive reviewer removal/rebalance; start new run.
6. Count caps/CPU/file extraction limits prevent runaway local resource use. No invented advanced metrics.
7. All fixtures only in isolated backend/tests storage; production remains empty.
8. greenlet3.1.1 pinned because original resolver selected source build requiring absent C++ tools onPython3.9; venv pip upgraded.
9. Portable relative runtime paths. The post-build request now authorizes commit/push to the user-specified origin; see section 24.

## 21. Files to Read First
HANDOFF.md; README.md; backend/app/main.py; backend/app/database/__init__.py; backend/app/models/__init__.py; backend/app/schemas/__init__.py; backend/app/parsers/__init__.py; backend/app/analyzers/__init__.py; backend/app/services/pipeline.py; backend/app/routers/api.py; backend/tests/test_api.py; backend/tests/test_analysis.py; frontend/src/App.tsx; frontend/src/api.ts; docker-compose.yml.

## 22. Git Status
Git initialized; inherited configured identity used. Original implementation checkpoint: dea7ee3 — Build AnnoPilot local annotation review MVP; documentation checkpoint a77ae2c. Origin is now https://github.com/TMTower18/AnnoPilot.git, supplied by the user. Active branch master; origin was empty when fetched. Runtime stores/media/logs/dependencies/builds remain ignored. Git metadata requires sandbox escalation because .git is read-only in the sandbox. Run git status, git log -5 --oneline and git ls-remote origin refs/heads/master for authoritative commit/push state. Original npm audit reported 0 vulnerabilities. Compose config and frontend/Swagger/proxy health checks passed; production datasets still [].

## 23. Resume Instruction
New Codex session:
1.Read this entire HANDOFF.md.
2.Inspect Files to Read First.
3.Run git status and git log -5 --oneline.
4.Verify actual source before assuming handoff is correct.
5.Source and current test results are source of truth.
6.Start first unfinished item under Exact Next Steps.
7.Do not rebuild completed functionality unless verification shows it is broken.
8.Update HANDOFF.md after meaningful progress.

## 24. Post-build dataset guard and appearance (2026-10-03)

Implemented reusable DatasetRequiredRoute and accessible DatasetRequiredDialog, with dynamic feature names, Cancel/Escape/focus handling and direct Upload Dataset CTA. Sidebar stays clickable. URL navigation covers /, /dashboard, /dataset, /difficulty, /smart-sampling, /workload, /review-queue and /settings with browser history. Protected pages wait for dataset status; backend failures show Unable to check dataset status and Retry instead of a missing-data prompt. Home/upload and Settings remain usable without a dataset. Successful upload resumes the requested feature. Existing scoring, import, sampling, balancing, review and export code is preserved.

AppearanceProvider centralizes Light/Dark/System (default System) and Small/Medium/Large (default Medium), validates and persists localStorage preferences, listens to live prefers-color-scheme changes, handles storage failures and synchronizes storage events. Initialization precedes React rendering. Semantic CSS palette tokens replace hardcoded UI colors; rem typography scales globally (14/16/18px root). Settings uses labelled native radio groups with visible checked states and keyboard focus. Charts/tooltips use palette tokens; layouts wrap and tables scroll at larger sizes.

Changed frontend/src/App.tsx, main.tsx, style.css; added appearance.css, appearance/AppearanceProvider.tsx, components/AppearanceSettings.tsx, components/DatasetRequiredRoute.tsx, hooks/usePageNavigation.ts and scripts/verify-appearance.cjs. README documents navigation and preferences. No backend, dependency or scoring changes.

Verification: frontend npm run build succeeded; backend pytest 39 passed. Focused Node appearance lifecycle check passed defaults, persistence, live System changes in both directions, explicit override, invalid preference fallback and listener cleanup. Browser verified all six sidebar prompts and direct URLs, Cancel/Escape, upload CTA, Settings without data, Light/Dark/System and all three font sizes, immediate updates and reload persistence. Light/Large screenshot saved in the chat visualization folder outside repository. Browser observed separate loading states; an isolated proxy targeting an unavailable API verified the error/Retry state and independent Appearance settings. Existing isolated ui_fixture dataset opened Dashboard, Dataset, Difficulty, Smart Sampling, Workload and Review Queue without guards, preserving existing plans/reviews. Production database remained empty; no demo data inserted.

Only temporary test servers on 3001/3002/8001 are used for regression checks and stopped afterward; production ports 3000/8000 remain running. Browser preferences restored to System/Medium. Docker verification remains outstanding because Desktop is manually paused, as recorded above. No additional product blocker identified. Commit/push authorized to origin master, without force push; verify actual Git output for the final hash.

## 25. Permanent Home and larger interface fonts (2026-10-03)

Refactored the existing landing into components/HomePage.tsx, added House/Home as the first sidebar item, and removed automatic `/` → Dashboard navigation when datasets exist. Home is always the product overview, independent of dataset loading; Dashboard remains the protected dataset health view. Home includes real seven-step workflow, five core module cards, task support with explicit RLE/pose/3D limitations, existing upload CTA and a compact active dataset card from the backend. No statistics or demo data are synthesized. Upload from Home still defaults to Dashboard after successful analysis. Dataset-required navigation retains feature-specific messages with Stay Here / Upload Dataset; direct protected URLs show the guard, never silently render Home.

Central root font tokens now use Small 16px, Medium 18px (default) and Large 20px. Existing rem typography scales app-wide, with additional section/table hierarchy adjustments. Sidebar width scales with rem; controls have scalable minimum heights, rows have larger padding, table headers wrap with minimum widths, cards wrap, and modals scroll within the viewport. Home uses existing semantic palettes in Light/Dark/System and responsive workflow/module grids. localStorage persistence and live System listeners are unchanged.

Changed App.tsx, style.css, appearance.css, DatasetRequiredRoute.tsx; added HomePage.tsx; updated README and this HANDOFF. Backend/scoring/planning/review logic and dependencies unchanged.

Verification: npm run build PASS; backend pytest 39 passed in 1.70s; focused appearance lifecycle check PASS. Browser computed roots exactly 16/18/20px and verified System/Medium persistence after reload. Home without data, Settings, sidebar Home highlight, Difficulty → Stay Here (still `/`), Smart Sampling → Upload, and all six protected direct URLs passed. Existing isolated fixture verified `/` stays Home with real 2 samples/4 annotations, Open Dashboard and return Home; all workflow pages opened at Large with no document horizontal overflow. Light and Dark Home screenshots inspected; System resolved to device dark and lifecycle test covered live changes. Responsive Home at observed CSS widths 600/1280 and table/upload modal at 960 inspected, with no document horizontal overflow; modal fit/scroll was bounded. No new production or demo datasets created.

Local production servers were stopped before this session; restarted with Start-AnnoPilot.ps1 (wrapper IDs backend 23276/frontend 24368). Test-only fixture servers on 3001/8001 stopped after verification. Production remains on 3000/8000; browser left on Home with System/Medium. Docker verification remains unperformed as previously recorded. Commit/push this verified source to existing origin/master without force; Git output is authoritative for final hash.
