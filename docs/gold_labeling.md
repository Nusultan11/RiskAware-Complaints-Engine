# Gold Labeling Guide

## Goal

Create a high-quality human-labeled dataset for:
- `gold_legal_threat` (`0/1`)
- `gold_priority` (`P1/P2/P3`)

This gold set is used to evaluate and improve model quality beyond weak labels.

## Input and Output

- Input script: `scripts/create_gold_set.py`
- Generated file: `reports/labeling/gold_candidates.csv`
- Manifest: `reports/labeling/gold_manifest.json`

## Labeling Rules

### 1) `gold_legal_threat`
- `1`: the complaint explicitly contains legal escalation intent or action
  - examples: lawsuit intent, attorney involvement, formal legal action mention
- `0`: no explicit legal escalation signal

### 2) `gold_priority`
- `P1`: critical, high-risk, immediate operational/escalation need
- `P2`: medium urgency, important but not immediate crisis
- `P3`: low urgency, routine handling

## Annotation Columns

In `gold_candidates.csv`, annotators fill:
- `gold_legal_threat`
- `gold_priority`
- `annotator_id`
- `confidence` (`1..5`)
- `review_notes`

## Quality Process

1. Double-label at least 20 percent of rows (two annotators independently).
2. Compute agreement on the overlap (Cohen's kappa or simple agreement).
3. Resolve disagreements by adjudication and update final labels.
4. Freeze a versioned output file (for example `gold_v1.csv`).

## Recommended First Batch

- Start with `n_samples=1500`.
- Keep class representation balanced by weak `priority` distribution.
- Expand to `3000+` once initial model comparison is ready.

