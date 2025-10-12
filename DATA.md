# Data Card

## Source
Custom reviews dataset (binary sentiment). Ensure you have permission to use and share the data.

## Schema
- `text` — raw user review (UTF-8 string)
- `label` — sentiment class: `positive` or `negative`

## Notes
- No PII stored.
- Train/test split is stratified.
- If using a public dataset, cite the source in this file.

## Usage
Place CSV at `data/reviews_mapped.csv`.  
Run: `make train && make evaluate`.
