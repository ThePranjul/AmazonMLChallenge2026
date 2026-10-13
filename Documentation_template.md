# Amazon ML Challenge 2026 — Business Entity Resolution
## Methodology Documentation — Iteration 2 / V2

### 1. Problem understanding

The task is to identify all Source 2 and Source 3 records corresponding to each Source 1
entity. Each Source 1 entity may map to zero, one, or multiple records. The evaluation metric
is macro F0.5, making precision especially important.

The submission contains two output artifacts:

- `matching_results.tsv`: final predicted matches.
- `candidate_pairs.tsv`: the final candidate set actually scored by the matching model.

### 2. Data preparation

Names and addresses are normalized using lowercase conversion, transliteration for non-ASCII
text where needed, selected business/address abbreviation expansion, punctuation normalization,
and whitespace normalization. Postal codes, house numbers, compact names, and a stopword-removed
core business name are derived for candidate generation and matching.

### 3. Open-set country handling

Blocking is performed dynamically within country partitions. Country labels are treated as
open-set strings rather than hard-coding a fixed country list. During test inference, France was
present as an unseen country relative to the training data and was handled as a separate
partition.

### 4. Candidate generation / blocking

The V2 blocking stage combines multiple sparse and exact-key signals:

- normalized-name sparse overlap;
- core-name sparse overlap;
- normalized-address sparse overlap;
- exact normalized-name keys;
- exact core-name keys;
- house-number/core-name combinations;
- core-name + postal-code combinations;
- postal-code candidates.

Sparse matrices use `HashingVectorizer` with 2^22 buckets and `sparse_dot_topn`. Common tokens
are restricted using document-frequency and absolute posting caps. The candidate cap is 40 per
Source-1 entity, with up to 80 candidates retained from each major sparse signal before final
combination.

The completed V2 training run generated 65,446,203 candidate pairs and recovered 79.8865% of
ground-truth positive pairs at the blocking stage.

### 5. Training split and labels

Ground-truth positive pairs are expanded into pair-level labels. Validation is performed using a
Source-1-level holdout so that the same Source-1 entity does not occur in both the training and
holdout partitions through different candidate rows.

The V2 run used a 90% / 10% Source-1 entity split:

- training S1 entities: 1,986,139
- holdout S1 entities: 220,682

### 6. Feature engineering

The V2 matcher uses 25 pairwise features covering:

- blocking score and log-transformed blocking score;
- exact normalized-name, core-name, compact-name, and address matches;
- postal and core-name/postal agreement;
- house-number agreement and conflict;
- Source-3 indicator;
- several RapidFuzz name and address similarities;
- name/address length differences and ratios;
- missing-field indicators;
- name/address interaction features.

### 7. Matching model

The matcher is a LightGBM binary classifier trained on positive candidate pairs and sampled
negative candidates. The final model is refit on all labelled Source-1 training entities after
threshold selection.

### 8. Threshold selection and post-processing

The global holdout result was:

`F0.5 = 0.830418` at threshold `0.800`.

Source-specific threshold tuning selected:

- S2 threshold: `0.775`
- S3 threshold: `0.810`

This gave holdout macro F0.5 of `0.830481`. Applying target-uniqueness post-processing on the
holdout increased the measured value to `0.831102`, so target-uniqueness was enabled for test
inference.

### 9. Final V2 test run

The completed test inference produced:

```text
Test Source 1 entities          1,732,544
Test candidate pairs            53,049,572
S1 entities with candidates     1,721,382
Maximum candidates/entity       40

matching_results.tsv rows       1,732,544
  empty                         254,958
  non-empty                   1,477,586

candidate_pairs.tsv rows       1,732,544
  empty                          11,162
  non-empty                   1,721,382
```

The organizer's official validator reported:

`PASS — no blocking issues found. Safe to submit.`

### 10. Public leaderboard result

The generated V2 `matching_results.tsv` was uploaded to the challenge Portal and received a
**0.800 public leaderboard F0.5 score**.

### 11. Data-use / fair-play statement

No external business/entity database, web lookup, geocoding service, entity-resolution API, or
other external business data was used. All matching decisions were derived from the challenge-
provided data.
