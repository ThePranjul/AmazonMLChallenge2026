# Amazon ML Challenge 2026 — Business Entity Resolution

## V2 final submission package

This package corresponds to the Iteration-2 pipeline whose generated `matching_results.tsv`
received a **0.800 public leaderboard score** during the challenge.

### Output files

Place the two final Kaggle-generated files in `output/`:

- `matching_results.tsv` — leaderboard-scored predictions.
- `candidate_pairs.tsv` — the exact candidate set scored during test inference.

The completed V2 Kaggle run produced and officially validated both files.

### V2 configuration

- Candidate cap: **40** per Source-1 entity
- Sparse signal top-k: **80**
- Name posting cap: **15,000**
- Address posting cap: **30,000**
- Core-name posting cap: **15,000**
- Name max document frequency: **0.02**
- Address max document frequency: **0.05**
- Core-name max document frequency: **0.03**
- Hash space: **2^22**
- Minimum token length: **3**
- Name/address/core blocking weights: **2 / 1 / 3**
- Postal blocking weight: **30**
- Postal top-k: **25**
- Final feature count: **25**
- Random seed: **42**
- Holdout split: **10% of Source-1 entities**
- Final source thresholds: **S2 = 0.775, S3 = 0.810**
- Target-uniqueness post-processing: **enabled**

### Validation and run statistics

- Training S1 entities: **2,206,821**
- Training candidate pairs: **65,446,203**
- Blocking recall: **79.8865%**
- Candidate reduction ratio: **0.00055277%**
- Training rows after sampling: **27,467,420**
- Positive training rows: **5,493,484**
- Negative training rows: **21,973,936**
- Final all-data training rows: **30,510,110**
- Global-threshold holdout macro F0.5: **0.830418**
- Source-threshold holdout macro F0.5: **0.830481**
- Source-threshold + target-uniqueness holdout macro F0.5: **0.831102**

### Test inference statistics

- Test Source-1 entities: **1,732,544**
- Test candidate pairs: **53,049,572**
- Test S1 entities with at least one candidate: **1,721,382**
- Maximum candidates per entity: **40**
- `matching_results.tsv`: **1,732,544** rows
- `matching_results.tsv` empty rows: **254,958**
- `matching_results.tsv` non-empty rows: **1,477,586**
- `candidate_pairs.tsv`: **1,732,544** rows
- `candidate_pairs.tsv` empty rows: **11,162**
- `candidate_pairs.tsv` non-empty rows: **1,721,382**

### Validation

The completed Kaggle run executed the organizer's official validator and reported:

`PASS — no blocking issues found. Safe to submit.`

and

`[PASS] OFFICIAL VALIDATOR PASSED`

### Reproduction

The exact executed V2 notebook is included at:

`code/business_entity_resolution/src/amazonmlchallenge_v2_final.ipynb`

The notebook expects the organizer-provided `student_resource` dataset mounted in the
Kaggle environment and writes the outputs to `/kaggle/working/output/`.

A portable Python reproduction script is also included at:

`code/business_entity_resolution/src/business_entity_resolution.py`

Run it from an environment containing the challenge `student_resource` directory:

```bash
python src/business_entity_resolution.py \
  --resource-dir /path/to/student_resource \
  --output-dir ./output
```

### Data-use statement

No external business/entity database, web lookup, geocoding service, entity-resolution API,
or other external business data was used. The pipeline uses only the challenge-provided data.
