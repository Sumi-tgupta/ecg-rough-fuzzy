# Phase 1 — Foundation and interface system

## Objective

Create a stable, understandable project foundation before implementing missing research components. Refresh the existing browser workspace into a polished, responsive research product with a coherent Apple-inspired visual language and working light/dark appearance modes.

## Included in this phase

1. Audited the archive and distinguished the browser demonstration from the incomplete Python research scaffold.
2. Replaced the minimal browser page shell with a branded workspace layout, sidebar navigation, page-introduction area, data-mode indicator, mobile navigation drawer and footer disclaimer.
3. Added a central semantic design system in `web/styles.css`.
4. Added persistent light/dark theme switching with browser storage and theme-aware chart foregrounds.
5. Improved responsive layout, hover/focus states, reduced-motion handling and table/card styling.
6. Rewrote the top-level README and architecture documentation to reflect the actual contents of the supplied archive.
7. Documented the current state, design conventions and phased target architecture.

## Deliberately not included

- No replacement of the rough-set algorithm or fuzzy classifier.
- No new Python classifier or experiment pipeline.
- No claims that synthetic browser training is equivalent to MIT-BIH training.
- No API/backend integration or cloud persistence.
- No medical or clinical functionality claims.

These items require separate algorithmic implementation and validation. Mixing them into the visual-foundation phase would make it harder to isolate regressions.

## Verification performed

- `node --check web/core.js`
- `node --check web/app.js`
- `node tests/browser_core_smoke.test.js`
- Executed the existing browser training function in Node to verify the deterministic synthetic benchmark still completes.
- Added a dependency-free Node smoke test to preserve key browser-model contracts.
- Verified the current benchmark output: 1,228 synthetic training beats, 430 synthetic test beats, 25 browser features, 9 selected attributes, 60 rules in the fuzzy-only model and 60 rules in the rough-set/fuzzy model.
- Checked Python dependency availability in the working environment. WFDB, PyWavelets and Streamlit are not installed here; this is separate from the missing source modules noted in the repository audit.

## Current benchmark baseline

| Quantity | Current deterministic browser run |
|---|---:|
| Training beats | 1,228 |
| Test beats | 430 |
| Browser feature count | 25 |
| Rough-set reduct size | 9 |
| Fuzzy-only accuracy | 93.49% |
| Rough-set + fuzzy accuracy | 85.35% |

These values describe the synthetic generator and browser algorithms only. They are not MIT-BIH results and must not be used as clinical performance evidence.

## Acceptance criteria

- A modern browser can open `web/index.html` directly or through a static server.
- All six existing workspace sections remain available.
- Appearance can be switched between light and dark and persists after refresh.
- The workspace remains usable at desktop, tablet and mobile widths.
- Synthetic-data limitations remain visible and unambiguous.
- JavaScript syntax checks pass and the demo training function continues to execute.

## Exit criteria for Phase 2

Before beginning real-data integration, agree on one canonical feature schema and implement the missing Python fuzzy classifier and experiment orchestration. The current browser demo should remain labelled as a demonstration until equivalence with the Python research engine has been demonstrated by tests.
