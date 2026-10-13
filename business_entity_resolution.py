"""
Amazon ML Challenge 2026 — Business Entity Resolution (V2)
Portable source extracted from the final V2 Kaggle notebook.
"""
import argparse
# =====================================================================
# CELL 1 - IMPORTS
# =====================================================================

import os
import sys
import re
import gc
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from rapidfuzz import fuzz
from unidecode import unidecode
import lightgbm as lgb

print("All imports loaded successfully.")

# =====================================================================
# CELL 2 - PATHS + CONFIGURATION (ITERATION 2)
# =====================================================================

parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 V2 entity resolution pipeline")
parser.add_argument("--resource-dir", required=True, help="Path to organizer student_resource directory")
parser.add_argument("--output-dir", default="./output", help="Directory for generated output files")
args = parser.parse_args()
RESOURCE_DIR = str(Path(args.resource_dir).resolve())
INPUT_DIR = f"{RESOURCE_DIR}/dataset"
TRAIN_DIR = f"{INPUT_DIR}/train"
TEST_DIR = f"{INPUT_DIR}/test"
VALIDATOR_PATH = f"{RESOURCE_DIR}/utils/validate_submission.py"
OUTPUT_DIR = str(Path(args.output_dir).resolve())
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# HIGH-RECALL BLOCKING CONFIGURATION
# ---------------------------------------------------------------------
NAME_MAX_DOC_FREQ = 0.02
ADDR_MAX_DOC_FREQ = 0.05
CORE_MAX_DOC_FREQ = 0.03

NAME_MAX_POSTINGS = 15_000
ADDR_MAX_POSTINGS = 30_000
CORE_MAX_POSTINGS = 15_000

MIN_TOKEN_LEN = 3

# More candidates are allowed because candidate generation is the recall ceiling.
CANDIDATES_PER_ENTITY = 40
BLOCK_SIGNAL_TOPK = 80

BLOCK_NAME_WEIGHT = 2
BLOCK_CORE_WEIGHT = 3
BLOCK_ADDR_WEIGHT = 1
POSTAL_WEIGHT = 30

# Exact-key passes are deliberately capped. If a key is too popular, it is skipped
# and the fuzzy/sparse signals handle it instead of injecting arbitrary candidates.
EXACT_NAME_TOPK = 50
EXACT_CORE_TOPK = 50
EXACT_HOUSE_CORE_TOPK = 30
EXACT_CORE_POSTAL_TOPK = 30
EXACT_GROUP_CAP = 100
POSTAL_TOPK = 25

# ---------------------------------------------------------------------
# SPARSE BLOCKING CONFIGURATION
# ---------------------------------------------------------------------
HASH_BITS = 22
HASH_FEATURES = 1 << HASH_BITS
SPARSE_BUILD_CHUNK = 100_000
BLOCK_QUERY_CHUNK = 5_000
MAX_PRODUCT_NNZ = 1_000_000

# ---------------------------------------------------------------------
# FEATURE / TRAINING CONFIGURATION
# ---------------------------------------------------------------------
FEATURE_CHUNK = 100_000
NEG_PER_POS = 4
MIN_NEG_PER_CHUNK = 2_000
RANDOM_SEED = 42
MIN_BLOCKING_RECALL = 0.95

ABBR_NAME = {
    r"\bcorp\b": "corporation",
    r"\bcorpn\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bco\b": "company",
    r"&": " and ",
}

ABBR_ADDR = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bapt\b": "apartment",
    r"\bhwy\b": "highway",
    r"\bpl\b": "place",
    r"\bctr\b": "center",
    r"\bste\b": "suite",
}

STOP_TOKENS_NAME = {
    "corporation", "incorporated", "limited", "private", "company", "llc", "llp",
    "and", "the", "of", "inc", "corp", "ltd", "pvt", "co", "group",
    "enterprises", "enterprise", "industries", "industry", "solutions",
    "services", "service",
}

NAME_STOP_PATTERN = (
    r"\b(?:" + "|".join(
        re.escape(x) for x in sorted(STOP_TOKENS_NAME, key=len, reverse=True)
    ) + r")\b"
)

FEATURE_COLS = [
    "block_score",
    "block_score_log",
    "name_exact",
    "core_name_exact",
    "compact_name_exact",
    "addr_exact",
    "postal_match",
    "core_postal_match",
    "house_match",
    "house_conflict",
    "source3",
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "name_len_diff",
    "addr_len_diff",
    "name_len_ratio",
    "addr_len_ratio",
    "name_missing",
    "addr_missing",
    "name_addr_interaction",
]

print("Iteration-2 configuration loaded.")
print(f"Candidate cap: {CANDIDATES_PER_ENTITY}")
print(f"Sparse signal top-k: {BLOCK_SIGNAL_TOPK}")
print(f"Feature count: {len(FEATURE_COLS)}")


# =====================================================================
# CELL 3 - PRE-FLIGHT CHECK
# =====================================================================

print("=" * 75)
print("PRE-FLIGHT CHECK")
print("=" * 75)

fails = []
warns = []


def check(condition, ok_msg, fail_msg):
    if condition:
        print(f"  [PASS] {ok_msg}")
    else:
        print(f"  [FAIL] {fail_msg}")
        fails.append(fail_msg)


# -------------------------------------------------------------
# Paths
# -------------------------------------------------------------

print("\n[1] Checking paths...")

check(
    os.path.isdir(INPUT_DIR),
    f"INPUT_DIR exists: {INPUT_DIR}",
    f"INPUT_DIR missing: {INPUT_DIR}",
)

check(
    os.path.isdir(TRAIN_DIR),
    f"TRAIN_DIR exists: {TRAIN_DIR}",
    f"TRAIN_DIR missing: {TRAIN_DIR}",
)

check(
    os.path.isdir(TEST_DIR),
    f"TEST_DIR exists: {TEST_DIR}",
    f"TEST_DIR missing: {TEST_DIR}",
)

check(
    os.path.isfile(VALIDATOR_PATH),
    f"Validator exists: {VALIDATOR_PATH}",
    f"Validator missing: {VALIDATOR_PATH}",
)


# -------------------------------------------------------------
# Required files
# -------------------------------------------------------------

print("\n[2] Checking data files...")

required_files = [
    f"{TRAIN_DIR}/train_source1.tsv",
    f"{TRAIN_DIR}/train_source2.tsv",
    f"{TRAIN_DIR}/train_source3.tsv",
    f"{TRAIN_DIR}/train_ground_truth.tsv",
    f"{TEST_DIR}/test_source1.tsv",
    f"{TEST_DIR}/test_source2.tsv",
    f"{TEST_DIR}/test_source3.tsv",
]

for path in required_files:
    check(
        os.path.isfile(path),
        f"Found {path}",
        f"Missing {path}",
    )


# -------------------------------------------------------------
# Packages
# -------------------------------------------------------------

print("\n[3] Checking packages...")

package_checks = {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": sparse.__version__ if hasattr(sparse, "__version__") else "available",
    "rapidfuzz": "available",
    "lightgbm": lgb.__version__,
    "sklearn HashingVectorizer": "available",
}

for name, version in package_checks.items():
    print(f"  [PASS] {name}: {version}")


# -------------------------------------------------------------
# Header checks
# -------------------------------------------------------------

print("\n[4] Checking schemas...")

expected_source_cols = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}

for path in [
    f"{TRAIN_DIR}/train_source1.tsv",
    f"{TRAIN_DIR}/train_source2.tsv",
    f"{TRAIN_DIR}/train_source3.tsv",
    f"{TEST_DIR}/test_source1.tsv",
    f"{TEST_DIR}/test_source2.tsv",
    f"{TEST_DIR}/test_source3.tsv",
]:

    cols = set(
        pd.read_csv(
            path,
            sep="\t",
            nrows=0
        ).columns
    )

    check(
        expected_source_cols.issubset(cols),
        f"Schema OK: {Path(path).name}",
        f"Bad schema: {Path(path).name}",
    )

gt_cols = set(
    pd.read_csv(
        f"{TRAIN_DIR}/train_ground_truth.tsv",
        sep="\t",
        nrows=0
    ).columns
)

check(
    {
        "source1_entity_id",
        "matched_entity_ids",
    }.issubset(gt_cols),
    "Ground-truth schema OK",
    "Ground-truth schema is incorrect",
)


# -------------------------------------------------------------
# Configuration sanity
# -------------------------------------------------------------

print("\n[5] Checking configuration...")

check(
    CANDIDATES_PER_ENTITY >= 1,
    "Candidate cap is valid",
    "Candidate cap must be >= 1",
)

check(
    BLOCK_SIGNAL_TOPK >= CANDIDATES_PER_ENTITY,
    "Signal top-k is >= final candidate cap",
    "BLOCK_SIGNAL_TOPK is too small",
)

check(
    HASH_FEATURES >= 2**20,
    "Hash space is sufficiently large",
    "Hash space is too small",
)

check(
    0 < NAME_MAX_DOC_FREQ <= 1,
    "NAME_MAX_DOC_FREQ is valid",
    "Invalid NAME_MAX_DOC_FREQ",
)

check(
    0 < ADDR_MAX_DOC_FREQ <= 1,
    "ADDR_MAX_DOC_FREQ is valid",
    "Invalid ADDR_MAX_DOC_FREQ",
)


# -------------------------------------------------------------
# Synthetic sparse check
# -------------------------------------------------------------

print("\n[6] Testing sparse hashing...")

try:

    smoke_vec = HashingVectorizer(
        analyzer="word",
        token_pattern=rf"(?u)\b\w{{{MIN_TOKEN_LEN},}}\b",
        lowercase=False,
        ngram_range=(1, 1),
        binary=True,
        norm=None,
        alternate_sign=False,
        n_features=2**14,
        dtype=np.float32,
    )

    smoke_other = smoke_vec.transform(
        [
            "abc corporation road",
            "abc limited road",
            "completely different business",
        ]
    )

    smoke_query = smoke_vec.transform(
        [
            "abc corporation",
            "different business",
        ]
    )

    smoke_prod = (
        smoke_query @ smoke_other.T
    ).tocsr()

    check(
        smoke_prod.nnz > 0,
        "Sparse matrix multiplication works",
        "Sparse matrix multiplication failed",
    )

except Exception as e:

    fails.append(
        f"Sparse smoke test failed: {repr(e)}"
    )

    print(
        f"  [FAIL] Sparse smoke test: {e}"
    )


# -------------------------------------------------------------
# Output directory
# -------------------------------------------------------------

print("\n[7] Checking output directory...")

try:

    test_file = f"{OUTPUT_DIR}/.__write_test__"

    with open(test_file, "w") as f:
        f.write("ok")

    os.remove(test_file)

    print(
        "  [PASS] OUTPUT_DIR is writable."
    )

except Exception as e:

    fails.append(
        f"OUTPUT_DIR is not writable: {e}"
    )

    print(
        f"  [FAIL] OUTPUT_DIR: {e}"
    )


# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------

print("\n" + "=" * 75)
print("PRE-FLIGHT SUMMARY")
print("=" * 75)

print(f"FAIL: {len(fails)}")
print(f"WARN: {len(warns)}")

if fails:

    print("\nDO NOT PROCEED.")
    for x in fails:
        print(" -", x)

    raise RuntimeError(
        "Pre-flight check failed."
    )

print("\n[PASS] Pre-flight checks completed successfully.")

# =====================================================================
# CELL 3A - INSTALL / CHECK FAST SPARSE TOP-K LIBRARY
# =====================================================================

import subprocess
import sys

print("=" * 75)
print("CHECKING sparse_dot_topn")
print("=" * 75)

try:
    from sparse_dot_topn import sp_matmul_topn
    print("[PASS] sparse_dot_topn is already installed.")

except ImportError:

    print("[INFO] Installing sparse-dot-topn...")

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "sparse-dot-topn",
        ]
    )

    from sparse_dot_topn import sp_matmul_topn

    print("[PASS] sparse_dot_topn installed successfully.")


# Tiny functionality test.
A = sparse.csr_matrix(
    np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.float32
    )
)

B = sparse.csr_matrix(
    np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float32
    )
)

C = sp_matmul_topn(
    A,
    B.T,
    top_n=2,
    threshold=1,
    sort=True,
    n_threads=max(
        1,
        os.cpu_count() - 1
    ),
)

assert C.shape == (3, 4)
assert C.nnz > 0

print(
    f"[PASS] Sparse top-k test succeeded. "
    f"shape={C.shape}, nnz={C.nnz}"
)

del A, B, C
gc.collect()

# =====================================================================
# CELL 4 - FAST NORMALIZATION + ER BLOCKING KEYS
# =====================================================================

def normalize_name_series(s):
    s = s.fillna("").astype("string")
    mask = s.str.contains(r"[^\x00-\x7F]", regex=True, na=False)
    if mask.any():
        s.loc[mask] = s.loc[mask].map(unidecode)
    s = s.str.lower()
    for pat, repl in ABBR_NAME.items():
        s = s.str.replace(pat, repl, regex=True)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def normalize_address_series(s):
    s = s.fillna("").astype("string")
    mask = s.str.contains(r"[^\x00-\x7F]", regex=True, na=False)
    if mask.any():
        s.loc[mask] = s.loc[mask].map(unidecode)
    s = s.str.lower()
    s = s.str.replace(r"near\s+[^,]*", " ", regex=True)
    for pat, repl in ABBR_ADDR.items():
        s = s.str.replace(pat, repl, regex=True)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def prepare_dataframe(df):
    names = df["business_name"].fillna("").astype("string")
    addresses = df["business_address"].fillna("").astype("string")

    pin = addresses.str.extract(r"\b(\d{6})\b", expand=False)
    zip5 = addresses.str.extract(r"\b(\d{5})(?:-\d{4})?\b", expand=False)
    postal = pin.fillna(zip5).fillna("").astype("string")

    name_norm = normalize_name_series(names)
    addr_norm = normalize_address_series(addresses)

    # Core name removes generic/legal terms and is a major recall channel.
    core_name = name_norm.str.replace(
        NAME_STOP_PATTERN, " ", regex=True
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Space-stripped name catches variants such as "moon boards" vs "moonboards".
    compact_name = name_norm.str.replace(" ", "", regex=False)

    # First number in normalized address; useful for cross-field blocking/features.
    house_number = addr_norm.str.extract(
        r"\b(\d+[a-z]?)\b", expand=False
    ).fillna("").astype("string")

    name_missing = (name_norm == "").astype(np.int8)
    addr_missing = (addr_norm == "").astype(np.int8)
    name_len = name_norm.str.len().fillna(0).clip(0, 32767).astype(np.int16)
    addr_len = addr_norm.str.len().fillna(0).clip(0, 32767).astype(np.int16)

    out = pd.DataFrame({
        "entity_id": df["entity_id"].astype("string"),
        "country": df["country"].fillna("").astype("string"),
        "_row": df["_row"].astype(np.int32),
        "name_norm": name_norm,
        "core_name": core_name,
        "compact_name": compact_name,
        "addr_norm": addr_norm,
        "postal": postal,
        "house_number": house_number,
        "name_len": name_len,
        "addr_len": addr_len,
        "name_missing": name_missing,
        "addr_missing": addr_missing,
    })
    return out


# =====================================================================
# CELL 5 - NORMALIZATION SMOKE TEST
# =====================================================================

print("=" * 75)
print("NORMALIZATION SMOKE TEST")
print("=" * 75)

smoke = pd.DataFrame(
    {
        "entity_id": [
            "S1-TEST-1",
            "S1-TEST-2",
            "S1-TEST-3",
        ],

        "business_name": [
            "ABC Corp & Sons Pvt Ltd",
            "राज इनवेस्टमेंट्स एलएलपी",
            "",
        ],

        "business_address": [
            "123 Main Rd, Delhi 110001",
            "Near SBI ATM, MG Road 110002",
            "",
        ],

        "country": [
            "India",
            "India",
            "US",
        ],
    }
)

smoke["_row"] = np.arange(
    len(smoke),
    dtype=np.int32
)

smoke = prepare_dataframe(
    smoke
)

print(smoke)


# -------------------------------------------------------------
# Basic normalization checks
# -------------------------------------------------------------

assert smoke.loc[0, "name_norm"] != ""
assert smoke.loc[0, "addr_norm"] != ""

assert smoke.loc[0, "postal"] == "110001"


# -------------------------------------------------------------
# Transliteration check
# -------------------------------------------------------------

original_native = "राज इनवेस्टमेंट्स एलएलपी"

transliterated = smoke.loc[
    1,
    "name_norm"
]

assert transliterated != ""

# Transliteration should remove the original non-ASCII script.
assert not bool(
    re.search(
        r"[^\x00-\x7F]",
        transliterated
    )
)

# The normalized value should contain useful alphabetic content.
assert any(
    ch.isalpha()
    for ch in transliterated
)

print(
    f"\n[PASS] Native-script transliteration worked:"
)
print(
    f"  Original    : {original_native}"
)
print(
    f"  Normalized  : {transliterated}"
)


# -------------------------------------------------------------
# Missing-field checks
# -------------------------------------------------------------

assert smoke.loc[2, "name_missing"] == 1
assert smoke.loc[2, "addr_missing"] == 1

print(
    "[PASS] Missing-name/address handling works."
)


# -------------------------------------------------------------
# Address normalization
# -------------------------------------------------------------

assert "road" in smoke.loc[
    0,
    "addr_norm"
]

assert "near" not in smoke.loc[
    1,
    "addr_norm"
]

print(
    "[PASS] Address normalization works."
)


print(
    "\n" + "=" * 75
)

print(
    "[PASS] NORMALIZATION SMOKE TEST PASSED"
)

print(
    "=" * 75
)



# =====================================================================
# CELL 6 - HIGH-RECALL MULTI-PASS SPARSE + EXACT-KEY BLOCKING
# =====================================================================

WORD_HASH_VECTOR = HashingVectorizer(
    analyzer="word",
    token_pattern=rf"(?u)\b\w{{{MIN_TOKEN_LEN},}}\b",
    lowercase=False,
    ngram_range=(1, 1),
    binary=True,
    norm=None,
    alternate_sign=False,
    n_features=HASH_FEATURES,
    dtype=np.float32,
)


def build_filtered_sparse(texts, max_doc_freq, max_postings, vectorizer=WORD_HASH_VECTOR):
    texts = np.asarray(texts, dtype=object)
    n = len(texts)
    n_features = vectorizer.n_features
    if n == 0:
        return sparse.csr_matrix((0, n_features), dtype=np.float32), np.zeros(n_features, dtype=bool)

    blocks = []
    for start in range(0, n, SPARSE_BUILD_CHUNK):
        end = min(start + SPARSE_BUILD_CHUNK, n)
        blocks.append(vectorizer.transform(texts[start:end].tolist()))
    X = sparse.vstack(blocks, format="csr")
    del blocks
    gc.collect()

    df_counts = np.asarray(X.sum(axis=0)).ravel()
    max_allowed = max(1, min(int(max_doc_freq * n), max_postings))
    keep = (df_counts > 0) & (df_counts <= max_allowed)
    X = X[:, keep].tocsr()
    X.eliminate_zeros()
    del df_counts
    gc.collect()
    return X, keep


def transform_filtered(texts, keep, vectorizer=WORD_HASH_VECTOR):
    texts = np.asarray(texts, dtype=object)
    if len(keep) == 0:
        return sparse.csr_matrix((len(texts), 0), dtype=np.float32)
    X = vectorizer.transform(texts.tolist())
    X = X[:, keep].tocsr()
    X.eliminate_zeros()
    return X


def topk_sparse_matches(query_matrix, other_matrix, topk):
    if query_matrix.shape[0] == 0 or other_matrix.shape[0] == 0 or query_matrix.shape[1] == 0:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))

    n_threads = max(1, (os.cpu_count() or 2) - 1)
    result = sp_matmul_topn(
        query_matrix.tocsr(), other_matrix.T,
        top_n=topk, threshold=1, sort=True, n_threads=n_threads
    ).tocsr()
    if result.nnz == 0:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))

    rows = np.repeat(np.arange(result.shape[0], dtype=np.int32), np.diff(result.indptr))
    cols = result.indices.astype(np.int32, copy=False)
    scores = result.data.astype(np.float32, copy=False)
    del result
    gc.collect()
    return rows, cols, scores


def build_exact_index(keys):
    """Hash/factorize one exact blocking key, then sort integer codes for fast group access."""
    arr = np.asarray(keys, dtype=object)
    valid_pos = np.flatnonzero(arr != "")
    if len(valid_pos) == 0:
        return None

    codes, uniques = pd.factorize(arr[valid_pos], sort=False)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_codes)) + 1]
    ends = np.r_[starts[1:], len(sorted_codes)]

    return {
        "index": pd.Index(np.asarray(uniques, dtype=object)),
        "starts": starts.astype(np.int32, copy=False),
        "ends": ends.astype(np.int32, copy=False),
        "positions": valid_pos[order].astype(np.int32, copy=False),
    }


def exact_key_candidates(query_keys, key_index, topk, weight, max_group=EXACT_GROUP_CAP):
    if key_index is None:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))

    q = np.asarray(query_keys, dtype=object)
    valid_q = np.flatnonzero(q != "")
    if len(valid_q) == 0:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))

    codes = key_index["index"].get_indexer(q[valid_q])
    good = codes >= 0
    if not good.any():
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))

    rows_good = valid_q[good]
    codes_good = codes[good]
    rows_out, cols_out, scores_out = [], [], []

    for code in np.unique(codes_good):
        qrows = rows_good[codes_good == code]
        left = int(key_index["starts"][code])
        right = int(key_index["ends"][code])
        group_size = right - left
        if group_size == 0 or group_size > max_group:
            continue
        opos = key_index["positions"][left:min(right, left + topk)]
        if len(opos) == 0:
            continue
        rows_out.append(np.repeat(qrows, len(opos)).astype(np.int32, copy=False))
        cols_out.append(np.tile(opos, len(qrows)).astype(np.int32, copy=False))
        scores_out.append(np.full(len(qrows) * len(opos), weight, dtype=np.float32))

    if not rows_out:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))
    return np.concatenate(rows_out), np.concatenate(cols_out), np.concatenate(scores_out)


def build_postal_index(postals):
    postals = np.asarray(postals, dtype=object)
    valid = np.flatnonzero(postals != "")
    if len(valid) == 0:
        return np.empty(0, object), np.empty(0, np.int32)
    order = np.argsort(postals[valid], kind="stable")
    return postals[valid][order], valid[order].astype(np.int32, copy=False)


def postal_candidates(query_postal, sorted_postal, sorted_other_idx, topk):
    query_postal = np.asarray(query_postal, dtype=object)
    nonempty = np.flatnonzero(query_postal != "")
    if len(nonempty) == 0 or len(sorted_postal) == 0:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))

    rows, cols, scores = [], [], []
    for postal in np.unique(query_postal[nonempty]):
        q_rows = np.flatnonzero(query_postal == postal)
        left = np.searchsorted(sorted_postal, postal, side="left")
        right = np.searchsorted(sorted_postal, postal, side="right")
        other_rows = sorted_other_idx[left:min(left + topk, right)]
        if len(other_rows) == 0:
            continue
        rows.append(np.repeat(q_rows, len(other_rows)).astype(np.int32, copy=False))
        cols.append(np.tile(other_rows, len(q_rows)).astype(np.int32, copy=False))
        scores.append(np.full(len(q_rows) * len(other_rows), POSTAL_WEIGHT, dtype=np.float32))
    if not rows:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(scores)


def block_country_partition(s1_g, s2_g, s3_g):
    if len(s1_g) == 0 or (len(s2_g) == 0 and len(s3_g) == 0):
        return (np.empty(0, np.int32), np.empty(0, np.int8), np.empty(0, np.int32), np.empty(0, np.float32))

    other_name = np.concatenate([s2_g["name_norm"].to_numpy(dtype=object), s3_g["name_norm"].to_numpy(dtype=object)])
    other_core = np.concatenate([s2_g["core_name"].to_numpy(dtype=object), s3_g["core_name"].to_numpy(dtype=object)])
    other_compact = np.concatenate([s2_g["compact_name"].to_numpy(dtype=object), s3_g["compact_name"].to_numpy(dtype=object)])
    other_addr = np.concatenate([s2_g["addr_norm"].to_numpy(dtype=object), s3_g["addr_norm"].to_numpy(dtype=object)])
    other_postal = np.concatenate([s2_g["postal"].to_numpy(dtype=object), s3_g["postal"].to_numpy(dtype=object)])
    other_house = np.concatenate([s2_g["house_number"].to_numpy(dtype=object), s3_g["house_number"].to_numpy(dtype=object)])

    # Strong combined keys.
    other_house_core = np.where(
        (other_house != "") & (other_core != ""),
        np.char.add(np.char.add(other_house.astype(str), "|"), other_core.astype(str)),
        "",
    )
    other_core_postal = np.where(
        (other_core != "") & (other_postal != ""),
        np.char.add(np.char.add(other_core.astype(str), "|"), other_postal.astype(str)),
        "",
    )

    other_src = np.concatenate([np.zeros(len(s2_g), dtype=np.int8), np.ones(len(s3_g), dtype=np.int8)])
    other_pos = np.concatenate([s2_g["_row"].to_numpy(dtype=np.int32), s3_g["_row"].to_numpy(dtype=np.int32)])

    print(f"    S1={len(s1_g):,} | S2={len(s2_g):,} | S3={len(s3_g):,}", flush=True)

    print("    Building sparse name/core/address matrices...", flush=True)
    name_X, name_keep = build_filtered_sparse(other_name, NAME_MAX_DOC_FREQ, NAME_MAX_POSTINGS)
    core_X, core_keep = build_filtered_sparse(other_core, CORE_MAX_DOC_FREQ, CORE_MAX_POSTINGS)
    addr_X, addr_keep = build_filtered_sparse(other_addr, ADDR_MAX_DOC_FREQ, ADDR_MAX_POSTINGS)
    print(f"    Name={name_X.shape}, nnz={name_X.nnz:,}; Core={core_X.shape}, nnz={core_X.nnz:,}; Addr={addr_X.shape}, nnz={addr_X.nnz:,}", flush=True)

    print("    Building exact-key indexes...", flush=True)
    idx_compact = build_exact_index(other_compact)
    idx_core = build_exact_index(other_core)
    idx_house_core = build_exact_index(other_house_core)
    idx_core_postal = build_exact_index(other_core_postal)
    sorted_postal, sorted_other_idx = build_postal_index(other_postal)

    s1_name = s1_g["name_norm"].to_numpy(dtype=object)
    s1_core = s1_g["core_name"].to_numpy(dtype=object)
    s1_compact = s1_g["compact_name"].to_numpy(dtype=object)
    s1_addr = s1_g["addr_norm"].to_numpy(dtype=object)
    s1_postal = s1_g["postal"].to_numpy(dtype=object)
    s1_house = s1_g["house_number"].to_numpy(dtype=object)
    s1_house_core = np.where(
        (s1_house != "") & (s1_core != ""),
        np.char.add(np.char.add(s1_house.astype(str), "|"), s1_core.astype(str)),
        "",
    )
    s1_core_postal = np.where(
        (s1_core != "") & (s1_postal != ""),
        np.char.add(np.char.add(s1_core.astype(str), "|"), s1_postal.astype(str)),
        "",
    )
    s1_pos_all = s1_g["_row"].to_numpy(dtype=np.int32)

    all_s1, all_src, all_o, all_score = [], [], [], []
    n_s1 = len(s1_g)

    for start in range(0, n_s1, BLOCK_QUERY_CHUNK):
        end = min(start + BLOCK_QUERY_CHUNK, n_s1)
        if start == 0 or start % (BLOCK_QUERY_CHUNK * 10) == 0:
            print(f"    Blocking S1 {start:,}-{end:,}/{n_s1:,}", flush=True)

        pieces = []

        nq = transform_filtered(s1_name[start:end], name_keep)
        nr, nc, ns = topk_sparse_matches(nq, name_X, BLOCK_SIGNAL_TOPK)
        if len(nr):
            pieces.append(pd.DataFrame({"q": nr, "o": nc, "score": ns * BLOCK_NAME_WEIGHT}))
        del nq

        cq = transform_filtered(s1_core[start:end], core_keep)
        cr, cc, cs = topk_sparse_matches(cq, core_X, BLOCK_SIGNAL_TOPK)
        if len(cr):
            pieces.append(pd.DataFrame({"q": cr, "o": cc, "score": cs * BLOCK_CORE_WEIGHT}))
        del cq

        aq = transform_filtered(s1_addr[start:end], addr_keep)
        ar, ac, ass = topk_sparse_matches(aq, addr_X, BLOCK_SIGNAL_TOPK)
        if len(ar):
            pieces.append(pd.DataFrame({"q": ar, "o": ac, "score": ass * BLOCK_ADDR_WEIGHT}))
        del aq

        q, o, sc = exact_key_candidates(s1_compact[start:end], idx_compact, EXACT_NAME_TOPK, 90.0)
        if len(q): pieces.append(pd.DataFrame({"q": q, "o": o, "score": sc}))
        q, o, sc = exact_key_candidates(s1_core[start:end], idx_core, EXACT_CORE_TOPK, 110.0)
        if len(q): pieces.append(pd.DataFrame({"q": q, "o": o, "score": sc}))
        q, o, sc = exact_key_candidates(s1_house_core[start:end], idx_house_core, EXACT_HOUSE_CORE_TOPK, 85.0)
        if len(q): pieces.append(pd.DataFrame({"q": q, "o": o, "score": sc}))
        q, o, sc = exact_key_candidates(s1_core_postal[start:end], idx_core_postal, EXACT_CORE_POSTAL_TOPK, 120.0)
        if len(q): pieces.append(pd.DataFrame({"q": q, "o": o, "score": sc}))

        pr, pc, ps = postal_candidates(s1_postal[start:end], sorted_postal, sorted_other_idx, POSTAL_TOPK)
        if len(pr): pieces.append(pd.DataFrame({"q": pr, "o": pc, "score": ps}))

        if pieces:
            c = pd.concat(pieces, ignore_index=True)
            c = c.groupby(["q", "o"], sort=False, as_index=False)["score"].sum()
            c = c.sort_values(["q", "score"], ascending=[True, False], kind="mergesort")
            c = c.groupby("q", sort=False, group_keys=False).head(CANDIDATES_PER_ENTITY)
            q_idx = c["q"].to_numpy(dtype=np.int32)
            o_idx = c["o"].to_numpy(dtype=np.int32)
            all_s1.append(s1_pos_all[start:end][q_idx])
            all_src.append(other_src[o_idx])
            all_o.append(other_pos[o_idx])
            all_score.append(c["score"].to_numpy(dtype=np.float32))

        del nr, nc, ns, cr, cc, cs, ar, ac, ass, pr, pc, ps, pieces
        gc.collect()

    del name_X, core_X, addr_X, name_keep, core_keep, addr_keep
    del other_name, other_core, other_compact, other_addr, other_postal, other_house
    del other_house_core, other_core_postal, other_src, other_pos
    del idx_compact, idx_core, idx_house_core, idx_core_postal
    del sorted_postal, sorted_other_idx
    gc.collect()

    if not all_s1:
        return (np.empty(0, np.int32), np.empty(0, np.int8), np.empty(0, np.int32), np.empty(0, np.float32))

    return (
        np.concatenate(all_s1), np.concatenate(all_src),
        np.concatenate(all_o), np.concatenate(all_score)
    )


def block_all(s1_df, s2_df, s3_df):
    s2_groups = {country: group for country, group in s2_df.groupby("country", sort=False, observed=True)}
    s3_groups = {country: group for country, group in s3_df.groupby("country", sort=False, observed=True)}
    outputs = []
    for country, s1_g in s1_df.groupby("country", sort=False, observed=True):
        print("\n" + "=" * 70, flush=True)
        print(f"COUNTRY: {country}", flush=True)
        print("=" * 70, flush=True)
        s2_g = s2_groups.get(country, s2_df.iloc[0:0])
        s3_g = s3_groups.get(country, s3_df.iloc[0:0])
        if len(s2_g) == 0 and len(s3_g) == 0:
            print("  No S2/S3 records for this country.")
            continue
        r = block_country_partition(s1_g, s2_g, s3_g)
        if len(r[0]):
            outputs.append(r)
    if not outputs:
        return {"s1_pos": np.empty(0, np.int32), "src": np.empty(0, np.int8), "o_pos": np.empty(0, np.int32), "score": np.empty(0, np.float32)}
    return {
        "s1_pos": np.concatenate([x[0] for x in outputs]),
        "src": np.concatenate([x[1] for x in outputs]),
        "o_pos": np.concatenate([x[2] for x in outputs]),
        "score": np.concatenate([x[3] for x in outputs]),
    }


# =====================================================================
# CELL 7 - BLOCKING SMOKE TEST
# =====================================================================

print("=" * 75)
print("BLOCKING SMOKE TEST")
print("=" * 75)

# Build smoke data in the SAME raw schema as the real sources, then use
# prepare_dataframe() so the blocking code is tested against the exact
# derived-column schema it will see in production.

def make_smoke_source(rows):
    df = pd.DataFrame(rows)
    df["_row"] = np.arange(len(df), dtype=np.int32)
    return prepare_dataframe(df)

smoke_s1 = make_smoke_source({
    "entity_id": ["S1-0", "S1-1", "S1-2"],
    "country": ["US", "India", "France"],
    "business_name": [
        "ABC Corporation",
        "Raj Investments",
        "Foo Cafe",
    ],
    "business_address": [
        "123 Main Road, New York 10001",
        "5 MG Road, Delhi 110001",
        "10 Paris Street, Paris 75001",
    ],
})

smoke_s2 = make_smoke_source({
    "entity_id": ["S2-0", "S2-1"],
    "country": ["US", "India"],
    "business_name": [
        "ABC Corp",
        "Raj Investments LLP",
    ],
    "business_address": [
        "123 Main Rd, New York 10001",
        "5 MG Road, Delhi 110001",
    ],
})

smoke_s3 = make_smoke_source({
    "entity_id": ["S3-0"],
    "country": ["France"],
    "business_name": ["Foo Cafe"],
    "business_address": [
        "10 Paris Street, Paris 75001",
    ],
})

smoke_pairs = block_all(
    smoke_s1,
    smoke_s2,
    smoke_s3,
)

assert len(smoke_pairs["s1_pos"]) > 0

# S1 row 0 should retrieve S2 row 0.
found_us = (
    (smoke_pairs["s1_pos"] == 0)
    & (smoke_pairs["src"] == 0)
    & (smoke_pairs["o_pos"] == 0)
)

# S1 row 1 should retrieve S2 row 1.
found_india = (
    (smoke_pairs["s1_pos"] == 1)
    & (smoke_pairs["src"] == 0)
    & (smoke_pairs["o_pos"] == 1)
)

# S1 row 2 should retrieve S3 row 0 in an unseen country (France).
found_france = (
    (smoke_pairs["s1_pos"] == 2)
    & (smoke_pairs["src"] == 1)
    & (smoke_pairs["o_pos"] == 0)
)

assert found_us.any()
assert found_india.any()
assert found_france.any()

assert (
    np.bincount(
        smoke_pairs["s1_pos"],
        minlength=len(smoke_s1),
    ).max()
    <= CANDIDATES_PER_ENTITY
)

print(
    f"\n[PASS] Generated "
    f"{len(smoke_pairs['s1_pos'])} smoke-test candidates."
)
print("[PASS] US, India and unseen France country partitions work.")
print("[PASS] Final candidate cap is respected.")


# =====================================================================
# CELL 8 - LOAD + PREPROCESS TRAINING DATA
# =====================================================================

t0 = time.time()

print("=" * 75)
print("LOADING + PREPROCESSING TRAINING DATA")
print("=" * 75)


def load_source(
    path,
    source_prefix
):

    raw = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    print(
        f"  Loaded {Path(path).name}: "
        f"{len(raw):,} rows"
    )

    # Source-ID integrity.
    bad_prefix = ~raw["entity_id"].str.startswith(
        source_prefix
    )

    if bad_prefix.any():

        raise RuntimeError(
            f"{source_prefix} file contains "
            f"{bad_prefix.sum():,} invalid entity IDs."
        )

    # Entity IDs must be unique within each source.
    if raw["entity_id"].duplicated().any():

        dup = int(
            raw["entity_id"].duplicated().sum()
        )

        raise RuntimeError(
            f"{source_prefix} contains "
            f"{dup:,} duplicate entity IDs."
        )

    raw["_row"] = np.arange(
        len(raw),
        dtype=np.int32
    )

    prepared = prepare_dataframe(
        raw
    )

    del raw
    gc.collect()

    return prepared


print("\n[1] Source 1...")
s1 = load_source(
    f"{TRAIN_DIR}/train_source1.tsv",
    "S1-"
)

print(
    f"    elapsed: "
    f"{time.time() - t0:.1f}s"
)

print("\n[2] Source 2...")
s2 = load_source(
    f"{TRAIN_DIR}/train_source2.tsv",
    "S2-"
)

print(
    f"    elapsed: "
    f"{time.time() - t0:.1f}s"
)

print("\n[3] Source 3...")
s3 = load_source(
    f"{TRAIN_DIR}/train_source3.tsv",
    "S3-"
)

print(
    f"    elapsed: "
    f"{time.time() - t0:.1f}s"
)


# Country sanity checks.
print("\n[4] Country distributions...")

for name, df in [
    ("S1", s1),
    ("S2", s2),
    ("S3", s3),
]:

    print(
        f"\n{name}:"
    )

    print(
        df["country"]
        .value_counts(dropna=False)
        .head(20)
    )


print(
    "\n[PASS] Training data loaded and normalized."
)

print(
    f"Total preprocessing time: "
    f"{(time.time() - t0) / 60:.2f} minutes"
)

# =====================================================================
# CELL 9 - GROUND TRUTH INTEGRITY CHECK
# =====================================================================

print("=" * 75)
print("GROUND TRUTH INTEGRITY CHECK")
print("=" * 75)


def make_pair_keys(
    s1_pos,
    src,
    o_pos
):
    return (
        (s1_pos.astype(np.uint64) << np.uint64(33))
        |
        (src.astype(np.uint64) << np.uint64(32))
        |
        o_pos.astype(np.uint64)
    )


# -------------------------------------------------------------
# Load ground truth
# -------------------------------------------------------------

gt = pd.read_csv(
    f"{TRAIN_DIR}/train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    keep_default_na=False,
    na_filter=False,
)

print(
    f"Ground-truth rows: {len(gt):,}"
)


# -------------------------------------------------------------
# Expand comma-separated match lists
# -------------------------------------------------------------

gt_expanded = gt[
    [
        "source1_entity_id",
        "matched_entity_ids",
    ]
].copy()

gt_expanded["matched_entity_id"] = (
    gt_expanded["matched_entity_ids"]
    .str.split(",")
)

gt_expanded = (
    gt_expanded
    .explode(
        "matched_entity_id",
        ignore_index=True
    )
)

gt_expanded["matched_entity_id"] = (
    gt_expanded["matched_entity_id"]
    .str.strip()
)

gt_expanded = gt_expanded[
    (gt_expanded["source1_entity_id"] != "")
    &
    (gt_expanded["matched_entity_id"] != "")
]


# -------------------------------------------------------------
# Build entity-ID indices
# -------------------------------------------------------------

s1_index = pd.Index(
    s1["entity_id"]
)

s2_index = pd.Index(
    s2["entity_id"]
)

s3_index = pd.Index(
    s3["entity_id"]
)


# -------------------------------------------------------------
# Map IDs to row positions
# -------------------------------------------------------------

gt_s1_pos = s1_index.get_indexer(
    gt_expanded["source1_entity_id"]
)

gt_s2_pos = s2_index.get_indexer(
    gt_expanded["matched_entity_id"]
)

gt_s3_pos = s3_index.get_indexer(
    gt_expanded["matched_entity_id"]
)


# -------------------------------------------------------------
# Verify mapping
# -------------------------------------------------------------

bad_s1 = (
    gt_s1_pos < 0
)

bad_other = (
    (gt_s2_pos < 0)
    &
    (gt_s3_pos < 0)
)

print(
    f"Unmapped S1 ground-truth IDs: "
    f"{bad_s1.sum():,}"
)

print(
    f"Unmapped S2/S3 ground-truth IDs: "
    f"{bad_other.sum():,}"
)

if bad_s1.any() or bad_other.any():

    raise RuntimeError(
        "Ground truth contains IDs that cannot be mapped "
        "to the training source files."
    )


# -------------------------------------------------------------
# Determine source of each matched entity
# 0 = S2
# 1 = S3
# -------------------------------------------------------------

gt_src = np.where(
    gt_s2_pos >= 0,
    0,
    1
).astype(
    np.int8
)


# -------------------------------------------------------------
# Select the correct source-local row position
# -------------------------------------------------------------

gt_o_pos = np.empty(
    len(gt_src),
    dtype=np.int32
)

s2_mask = (
    gt_src == 0
)

s3_mask = ~s2_mask

# IMPORTANT:
# We index S2 only with S2 positions and S3 only with S3 positions.
if s2_mask.any():

    gt_o_pos[s2_mask] = (
        gt_s2_pos[s2_mask]
    ).astype(
        np.int32
    )

if s3_mask.any():

    gt_o_pos[s3_mask] = (
        gt_s3_pos[s3_mask]
    ).astype(
        np.int32
    )


gt_s1_pos = (
    gt_s1_pos.astype(
        np.int32
    )
)


# -------------------------------------------------------------
# Compact ground-truth pair keys
# -------------------------------------------------------------

gt_keys = make_pair_keys(
    gt_s1_pos,
    gt_src,
    gt_o_pos
)

gt_keys = np.unique(
    gt_keys
)


# -------------------------------------------------------------
# Country consistency check
# -------------------------------------------------------------

s1_country_arr = (
    s1["country"]
    .to_numpy(dtype=object)
)

s2_country_arr = (
    s2["country"]
    .to_numpy(dtype=object)
)

s3_country_arr = (
    s3["country"]
    .to_numpy(dtype=object)
)


# Allocate output first.
gt_other_country = np.empty(
    len(gt_src),
    dtype=object
)

# S2 rows -> S2 country array only
if s2_mask.any():

    gt_other_country[s2_mask] = (
        s2_country_arr[
            gt_o_pos[s2_mask]
        ]
    )

# S3 rows -> S3 country array only
if s3_mask.any():

    gt_other_country[s3_mask] = (
        s3_country_arr[
            gt_o_pos[s3_mask]
        ]
    )


country_mismatch = (
    s1_country_arr[gt_s1_pos]
    != gt_other_country
)


# -------------------------------------------------------------
# Results
# -------------------------------------------------------------

print(
    f"\nPositive match pairs: "
    f"{len(gt_keys):,}"
)

print(
    f"S2 positive pairs: "
    f"{s2_mask.sum():,}"
)

print(
    f"S3 positive pairs: "
    f"{s3_mask.sum():,}"
)

print(
    f"Cross-country GT pairs: "
    f"{country_mismatch.sum():,}"
)


if country_mismatch.any():

    # Show a few examples before stopping.
    bad_idx = np.flatnonzero(
        country_mismatch
    )[:10]

    print("\nExamples of country mismatches:")

    for i in bad_idx:

        print(
            "S1:",
            s1_country_arr[gt_s1_pos[i]],
            "| Other:",
            gt_other_country[i],
            "| Source:",
            "S2" if gt_src[i] == 0 else "S3"
        )

    raise RuntimeError(
        "Ground truth contains cross-country matches. "
        "Country hard partition would be unsafe."
    )


print(
    "\n[PASS] Ground truth contains no cross-country matches."
)

print(
    "[PASS] Country-partitioned blocking is safe."
)

print(
    "[PASS] Ground-truth integrity check completed."
)

# =====================================================================
# CELL 10 - FULL BLOCKING + RECALL GATE
# =====================================================================

print("=" * 75)
print("FULL TRAINING BLOCKING")
print("=" * 75)

t0 = time.time()

# -------------------------------------------------------------
# Run blocking
# -------------------------------------------------------------

pairs = block_all(
    s1,
    s2,
    s3
)

n_pairs = len(
    pairs["s1_pos"]
)

print("\n" + "=" * 75)
print("BLOCKING RESULTS")
print("=" * 75)

print(
    f"Candidate pairs: "
    f"{n_pairs:,}"
)

if n_pairs == 0:

    raise RuntimeError(
        "Blocking produced zero candidate pairs."
    )


# -------------------------------------------------------------
# Candidate statistics
# -------------------------------------------------------------

candidate_counts = np.bincount(
    pairs["s1_pos"],
    minlength=len(s1)
)

nonzero_counts = (
    candidate_counts[
        candidate_counts > 0
    ]
)

print(
    f"S1 entities with >=1 candidate: "
    f"{len(nonzero_counts):,}/{len(s1):,}"
)

if len(nonzero_counts) > 0:

    print(
        f"Mean candidates/entity: "
        f"{nonzero_counts.mean():.2f}"
    )

    print(
        f"Median candidates/entity: "
        f"{np.median(nonzero_counts):.2f}"
    )

    print(
        f"Maximum candidates/entity: "
        f"{nonzero_counts.max()}"
    )

else:

    raise RuntimeError(
        "No S1 entity received a candidate."
    )


# -------------------------------------------------------------
# Final candidate-cap check
# -------------------------------------------------------------

if (
    candidate_counts.max()
    > CANDIDATES_PER_ENTITY
):

    raise RuntimeError(
        "Candidate cap was violated."
    )

print(
    f"[PASS] Candidate cap <= "
    f"{CANDIDATES_PER_ENTITY}"
)


# -------------------------------------------------------------
# Blocking recall
# -------------------------------------------------------------

print(
    "\nChecking blocking recall..."
)

candidate_keys = make_pair_keys(
    pairs["s1_pos"],
    pairs["src"],
    pairs["o_pos"]
)

# Sort IN PLACE instead of creating np.unique(candidate_keys).
# This substantially reduces temporary memory usage.
candidate_keys.sort()

# gt_keys was created with np.unique() in Cell 9, so it is sorted.
gt_positions = np.searchsorted(
    candidate_keys,
    gt_keys
)

valid_gt = (
    gt_positions
    < len(candidate_keys)
)

gt_recovered = np.zeros(
    len(gt_keys),
    dtype=bool
)

if valid_gt.any():

    gt_recovered[
        valid_gt
    ] = (
        candidate_keys[
            gt_positions[
                valid_gt
            ]
        ]
        ==
        gt_keys[
            valid_gt
        ]
    )

blocking_recall = (
    gt_recovered.mean()
)

print(
    f"Ground-truth positive pairs: "
    f"{len(gt_keys):,}"
)

print(
    f"Recovered positive pairs: "
    f"{gt_recovered.sum():,}"
)

print(
    f"Blocking recall: "
    f"{blocking_recall:.4%}"
)


# -------------------------------------------------------------
# Candidate reduction ratio
# -------------------------------------------------------------

print(
    "\nCalculating candidate reduction..."
)

s1_country_counts = (
    s1["country"]
    .value_counts()
)

s2_country_counts = (
    s2["country"]
    .value_counts()
)

s3_country_counts = (
    s3["country"]
    .value_counts()
)

possible_same_country = 0

for country, n1 in s1_country_counts.items():

    n2 = int(
        s2_country_counts.get(
            country,
            0
        )
    )

    n3 = int(
        s3_country_counts.get(
            country,
            0
        )
    )

    possible_same_country += (
        int(n1)
        * (n2 + n3)
    )

reduction_ratio = (
    n_pairs
    / max(
        possible_same_country,
        1
    )
)

print(
    f"Candidate reduction ratio: "
    f"{reduction_ratio:.8%}"
)


# -------------------------------------------------------------
# Release temporary recall arrays
# -------------------------------------------------------------

del candidate_keys
del gt_positions
del valid_gt
del gt_recovered

gc.collect()


# -------------------------------------------------------------
# Blocking quality gate
# -------------------------------------------------------------

print(
    "\n" + "=" * 75
)
print(
    "BLOCKING QUALITY GATE"
)
print(
    "=" * 75
)

if blocking_recall >= 0.95:

    print(
        f"[PASS] Blocking recall = "
        f"{blocking_recall:.4%}"
    )

elif blocking_recall >= 0.85:

    print(
        f"[WARNING] Blocking recall = "
        f"{blocking_recall:.4%}"
    )

    print(
        "[WARNING] Below 95%, but continuing so this "
        "version can produce a baseline submission."
    )

else:

    print(
        f"[WARNING] Blocking recall = "
        f"{blocking_recall:.4%}"
    )

    print(
        "[WARNING] Recall is low; this is an exploratory "
        "baseline and should be improved in the next iteration."
    )


print(
    f"\nTotal blocking time: "
    f"{(time.time() - t0) / 60:.2f} minutes"
)

print(
    "[INFO] Continuing to feature generation and training."
)

# =====================================================================
# CELL 11 - RICH PAIRWISE FEATURES
# =====================================================================

FEATURE_ARRAYS = {
    "s1_name": s1["name_norm"].to_numpy(dtype=object),
    "s1_core": s1["core_name"].to_numpy(dtype=object),
    "s1_compact": s1["compact_name"].to_numpy(dtype=object),
    "s1_addr": s1["addr_norm"].to_numpy(dtype=object),
    "s1_postal": s1["postal"].to_numpy(dtype=object),
    "s1_house": s1["house_number"].to_numpy(dtype=object),
    "s1_name_len": s1["name_len"].to_numpy(),
    "s1_addr_len": s1["addr_len"].to_numpy(),

    "s2_name": s2["name_norm"].to_numpy(dtype=object),
    "s2_core": s2["core_name"].to_numpy(dtype=object),
    "s2_compact": s2["compact_name"].to_numpy(dtype=object),
    "s2_addr": s2["addr_norm"].to_numpy(dtype=object),
    "s2_postal": s2["postal"].to_numpy(dtype=object),
    "s2_house": s2["house_number"].to_numpy(dtype=object),
    "s2_name_len": s2["name_len"].to_numpy(),
    "s2_addr_len": s2["addr_len"].to_numpy(),

    "s3_name": s3["name_norm"].to_numpy(dtype=object),
    "s3_core": s3["core_name"].to_numpy(dtype=object),
    "s3_compact": s3["compact_name"].to_numpy(dtype=object),
    "s3_addr": s3["addr_norm"].to_numpy(dtype=object),
    "s3_postal": s3["postal"].to_numpy(dtype=object),
    "s3_house": s3["house_number"].to_numpy(dtype=object),
    "s3_name_len": s3["name_len"].to_numpy(),
    "s3_addr_len": s3["addr_len"].to_numpy(),
}


def compute_features_chunk(arrays, s1_pos, src, o_pos, block_score):
    n = len(s1_pos)
    n1_name = arrays["s1_name"][s1_pos]
    n1_core = arrays["s1_core"][s1_pos]
    n1_compact = arrays["s1_compact"][s1_pos]
    n1_addr = arrays["s1_addr"][s1_pos]
    n1_postal = arrays["s1_postal"][s1_pos]
    n1_house = arrays["s1_house"][s1_pos]
    n1_name_len = arrays["s1_name_len"][s1_pos].astype(np.int32, copy=False)
    n1_addr_len = arrays["s1_addr_len"][s1_pos].astype(np.int32, copy=False)

    n2_name = np.empty(n, dtype=object)
    n2_core = np.empty(n, dtype=object)
    n2_compact = np.empty(n, dtype=object)
    n2_addr = np.empty(n, dtype=object)
    n2_postal = np.empty(n, dtype=object)
    n2_house = np.empty(n, dtype=object)
    n2_name_len = np.empty(n, dtype=np.int32)
    n2_addr_len = np.empty(n, dtype=np.int32)

    m2 = src == 0
    m3 = ~m2
    if m2.any():
        idx = o_pos[m2]
        n2_name[m2] = arrays["s2_name"][idx]
        n2_core[m2] = arrays["s2_core"][idx]
        n2_compact[m2] = arrays["s2_compact"][idx]
        n2_addr[m2] = arrays["s2_addr"][idx]
        n2_postal[m2] = arrays["s2_postal"][idx]
        n2_house[m2] = arrays["s2_house"][idx]
        n2_name_len[m2] = arrays["s2_name_len"][idx]
        n2_addr_len[m2] = arrays["s2_addr_len"][idx]
    if m3.any():
        idx = o_pos[m3]
        n2_name[m3] = arrays["s3_name"][idx]
        n2_core[m3] = arrays["s3_core"][idx]
        n2_compact[m3] = arrays["s3_compact"][idx]
        n2_addr[m3] = arrays["s3_addr"][idx]
        n2_postal[m3] = arrays["s3_postal"][idx]
        n2_house[m3] = arrays["s3_house"][idx]
        n2_name_len[m3] = arrays["s3_name_len"][idx]
        n2_addr_len[m3] = arrays["s3_addr_len"][idx]

    feats = np.empty((n, len(FEATURE_COLS)), dtype=np.float32)
    feats[:, 0] = block_score
    feats[:, 1] = np.log1p(np.maximum(block_score, 0))

    name_nonempty = (n1_name != "") & (n2_name != "")
    addr_nonempty = (n1_addr != "") & (n2_addr != "")
    feats[:, 2] = (name_nonempty & (n1_name == n2_name)).astype(np.float32)
    feats[:, 3] = ((n1_core != "") & (n2_core != "") & (n1_core == n2_core)).astype(np.float32)
    feats[:, 4] = ((n1_compact != "") & (n2_compact != "") & (n1_compact == n2_compact)).astype(np.float32)
    feats[:, 5] = (addr_nonempty & (n1_addr == n2_addr)).astype(np.float32)
    feats[:, 6] = ((n1_postal == n2_postal) & (n1_postal != "")).astype(np.float32)
    feats[:, 7] = ((n1_core != "") & (n2_core != "") & (n1_postal != "") & (n1_postal == n2_postal) & (n1_core == n2_core)).astype(np.float32)
    feats[:, 8] = ((n1_house != "") & (n2_house != "") & (n1_house == n2_house)).astype(np.float32)
    feats[:, 9] = ((n1_house != "") & (n2_house != "") & (n1_house != n2_house)).astype(np.float32)
    feats[:, 10] = (src == 1).astype(np.float32)

    feats[:, 11] = np.fromiter((fuzz.ratio(a, b) if a and b else 0.0 for a, b in zip(n1_name, n2_name)), dtype=np.float32, count=n)
    feats[:, 12] = np.fromiter((fuzz.token_sort_ratio(a, b) if a and b else 0.0 for a, b in zip(n1_name, n2_name)), dtype=np.float32, count=n)
    feats[:, 13] = np.fromiter((fuzz.token_set_ratio(a, b) if a and b else 0.0 for a, b in zip(n1_name, n2_name)), dtype=np.float32, count=n)
    feats[:, 14] = np.fromiter((fuzz.partial_ratio(a, b) if a and b else 0.0 for a, b in zip(n1_name, n2_name)), dtype=np.float32, count=n)

    feats[:, 15] = np.fromiter((fuzz.ratio(a, b) if a and b else 0.0 for a, b in zip(n1_addr, n2_addr)), dtype=np.float32, count=n)
    feats[:, 16] = np.fromiter((fuzz.token_sort_ratio(a, b) if a and b else 0.0 for a, b in zip(n1_addr, n2_addr)), dtype=np.float32, count=n)
    feats[:, 17] = np.fromiter((fuzz.token_set_ratio(a, b) if a and b else 0.0 for a, b in zip(n1_addr, n2_addr)), dtype=np.float32, count=n)

    feats[:, 18] = np.abs(n1_name_len - n2_name_len).astype(np.float32)
    feats[:, 19] = np.abs(n1_addr_len - n2_addr_len).astype(np.float32)
    feats[:, 20] = np.minimum(n1_name_len, n2_name_len) / np.maximum(np.maximum(n1_name_len, n2_name_len), 1)
    feats[:, 21] = np.minimum(n1_addr_len, n2_addr_len) / np.maximum(np.maximum(n1_addr_len, n2_addr_len), 1)
    feats[:, 22] = ((n1_name == "") | (n2_name == "")).astype(np.float32)
    feats[:, 23] = ((n1_addr == "") | (n2_addr == "")).astype(np.float32)
    feats[:, 24] = (feats[:, 11] * feats[:, 15] / 100.0).astype(np.float32)
    return feats

print("[PASS] Iteration-2 feature functions defined successfully.")
print("[PASS] FEATURE_ARRAYS created successfully.")


# =====================================================================
# CELL 12 - FEATURE / MODEL SMOKE TEST
# =====================================================================

print("=" * 75)
print("FEATURE + MODEL SMOKE TEST")
print("=" * 75)

rng = np.random.default_rng(
    RANDOM_SEED
)

n_pairs = len(
    pairs["s1_pos"]
)

sample_n = min(
    20_000,
    n_pairs
)

sample_idx = rng.choice(
    n_pairs,
    size=sample_n,
    replace=False
)

sample_s1 = pairs[
    "s1_pos"
][sample_idx]

sample_src = pairs[
    "src"
][sample_idx]

sample_o = pairs[
    "o_pos"
][sample_idx]

sample_score = pairs[
    "score"
][sample_idx]


X_smoke = compute_features_chunk(
    FEATURE_ARRAYS,
    sample_s1,
    sample_src,
    sample_o,
    sample_score
)

assert X_smoke.shape == (
    sample_n,
    len(FEATURE_COLS)
)

assert np.isfinite(
    X_smoke
).all()


sample_keys = make_pair_keys(
    sample_s1,
    sample_src,
    sample_o
)

sample_labels = np.isin(
    sample_keys,
    gt_keys
)

print(
    f"Feature matrix: "
    f"{X_smoke.shape}"
)

print(
    f"Positive examples: "
    f"{sample_labels.sum():,}"
)

print(
    f"Negative examples: "
    f"{(~sample_labels).sum():,}"
)


# Small LightGBM test only if both classes appear.
if (
    sample_labels.any()
    and (~sample_labels).any()
):

    smoke_model = lgb.LGBMClassifier(
        n_estimators=5,
        num_leaves=15,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    smoke_model.fit(
        X_smoke,
        sample_labels.astype(
            np.int8
        )
    )

    smoke_probs = (
        smoke_model
        .predict_proba(
            X_smoke
        )[:, 1]
    )

    assert np.isfinite(
        smoke_probs
    ).all()

    print(
        "[PASS] LightGBM smoke training succeeded."
    )

else:

    print(
        "[PASS] Feature smoke test succeeded."
    )

    print(
        "[WARN] Smoke sample did not contain both classes; "
        "skipping tiny LightGBM fit."
    )


del X_smoke
del sample_keys
del sample_labels

gc.collect()

print(
    "\n[PASS] Feature pipeline is ready for full training."
)

# =====================================================================
# CELL 13 - TRAIN, SOURCE-AWARE THRESHOLD TUNING, FINAL REFIT
# =====================================================================

print("=" * 75)
print("ITERATION 2 MODEL TRAINING")
print("=" * 75)
t0 = time.time()
rng = np.random.default_rng(RANDOM_SEED)
n_s1 = len(s1)
n_pairs = len(pairs["s1_pos"])

# 10% grouped holdout: more labelled S1 entities remain in the training pool.
perm = rng.permutation(n_s1)
holdout_n = max(1, int(0.10 * n_s1))
holdout_ids = perm[:holdout_n]
is_train_s1 = np.ones(n_s1, dtype=bool)
is_train_s1[holdout_ids] = False

print(f"Training S1 entities: {is_train_s1.sum():,}")
print(f"Holdout S1 entities: {holdout_ids.size:,}")


def build_sample_matrix(use_train_mask):
    X_parts, y_parts = [], []
    for start in range(0, n_pairs, FEATURE_CHUNK):
        end = min(start + FEATURE_CHUNK, n_pairs)
        p_s1 = pairs["s1_pos"][start:end]
        p_src = pairs["src"][start:end]
        p_o = pairs["o_pos"][start:end]
        p_score = pairs["score"][start:end]
        mask = use_train_mask[p_s1]
        if not mask.any():
            continue
        local = np.flatnonzero(mask)
        keys = make_pair_keys(p_s1[local], p_src[local], p_o[local])
        labels = np.isin(keys, gt_keys)
        pos_idx = local[labels]
        neg_idx = local[~labels]
        desired_neg = max(MIN_NEG_PER_CHUNK, NEG_PER_POS * max(len(pos_idx), 1))
        if len(neg_idx) > desired_neg:
            neg_idx = rng.choice(neg_idx, size=desired_neg, replace=False)
        selected = np.concatenate([pos_idx, neg_idx])
        if len(selected) == 0:
            continue
        X_parts.append(compute_features_chunk(FEATURE_ARRAYS, p_s1[selected], p_src[selected], p_o[selected], p_score[selected]))
        y_parts.append(np.concatenate([np.ones(len(pos_idx), dtype=np.int8), np.zeros(len(neg_idx), dtype=np.int8)]))
        if start == 0 or start % (FEATURE_CHUNK * 10) == 0:
            print(f"  processed {end:,}/{n_pairs:,}")
    X = np.vstack(X_parts).astype(np.float32, copy=False)
    y = np.concatenate(y_parts)
    del X_parts, y_parts
    gc.collect()
    return X, y

print("\n[1] Building 90% training sample...")
X_train, y_train = build_sample_matrix(is_train_s1)
print(f"Training rows: {len(y_train):,}")
print(f"Positive rows: {int(y_train.sum()):,}")
print(f"Negative rows: {int((y_train == 0).sum()):,}")

model = lgb.LGBMClassifier(
    n_estimators=350,
    num_leaves=63,
    learning_rate=0.05,
    min_child_samples=50,
    reg_lambda=1.0,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=-1,
)
model.fit(X_train, y_train)
del X_train, y_train
gc.collect()
print(f"Initial model trained in {(time.time() - t0) / 60:.2f} minutes")

# ---------------------------------------------------------------------
# Holdout probabilities
# ---------------------------------------------------------------------
holdout_mask = ~is_train_s1[pairs["s1_pos"]]
hold_s1 = pairs["s1_pos"][holdout_mask]
hold_src = pairs["src"][holdout_mask]
hold_o = pairs["o_pos"][holdout_mask]
hold_score = pairs["score"][holdout_mask]
hold_probs_parts = []
for start in range(0, len(hold_s1), FEATURE_CHUNK):
    end = min(start + FEATURE_CHUNK, len(hold_s1))
    Xh = compute_features_chunk(FEATURE_ARRAYS, hold_s1[start:end], hold_src[start:end], hold_o[start:end], hold_score[start:end])
    hold_probs_parts.append(model.predict_proba(Xh)[:, 1].astype(np.float32))
    del Xh
hold_probs = np.concatenate(hold_probs_parts)
del hold_probs_parts
gc.collect()


def macro_f05_thresholds(s1_pos, src, o_pos, probs, t2, t3, gt_keys, eval_s1_ids, total_s1, uniqueness=False):
    pred_mask = np.where(src == 0, probs >= t2, probs >= t3)
    pred_s1 = s1_pos[pred_mask]
    pred_src = src[pred_mask]
    pred_o = o_pos[pred_mask]
    pred_probs = probs[pred_mask]

    if uniqueness and len(pred_s1):
        # A target record should correspond to one Source-1 entity. Keep the
        # highest-confidence prediction for repeated S2/S3 target IDs.
        order = np.argsort(-pred_probs, kind="stable")
        seen2, seen3 = set(), set()
        keep = np.zeros(len(order), dtype=bool)
        for j in order:
            key = (int(pred_src[j]), int(pred_o[j]))
            seen = seen2 if key[0] == 0 else seen3
            if key not in seen:
                seen.add(key)
                keep[j] = True
        pred_s1 = pred_s1[keep]
        pred_src = pred_src[keep]
        pred_o = pred_o[keep]

    pred_n = np.bincount(pred_s1, minlength=total_s1).astype(np.float64) if len(pred_s1) else np.zeros(total_s1, np.float64)
    pred_keys = make_pair_keys(pred_s1, pred_src, pred_o) if len(pred_s1) else np.empty(0, np.uint64)
    tp_n = np.zeros(total_s1, np.float64)
    if len(pred_keys):
        gt_idx = np.searchsorted(gt_keys, pred_keys)
        valid = gt_idx < len(gt_keys)
        is_tp = np.zeros(len(pred_keys), dtype=bool)
        if valid.any():
            is_tp[valid] = gt_keys[gt_idx[valid]] == pred_keys[valid]
        tp_n = np.bincount(pred_s1, weights=is_tp.astype(np.float64), minlength=total_s1)

    gt_s1 = (gt_keys >> np.uint64(33)).astype(np.int32)
    true_n = np.bincount(gt_s1, minlength=total_s1).astype(np.float64)
    pn, tp, tn = pred_n[eval_s1_ids], tp_n[eval_s1_ids], true_n[eval_s1_ids]
    precision = np.divide(tp, pn, out=np.zeros_like(tp), where=pn > 0)
    recall = np.divide(tp, tn, out=np.zeros_like(tp), where=tn > 0)
    denom = 0.25 * precision + recall
    f = np.divide(1.25 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    f[(pn == 0) & (tn == 0)] = 1.0
    return float(f.mean())


def tune_global_threshold():
    best_t, best_f = 0.70, -1.0
    for t in np.arange(0.55, 0.91, 0.025):
        f = macro_f05_thresholds(hold_s1, hold_src, hold_o, hold_probs, float(t), float(t), gt_keys, holdout_ids, n_s1, False)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f

best_threshold, global_f = tune_global_threshold()
print(f"\nGlobal threshold={best_threshold:.3f}; holdout macro F0.5={global_f:.6f}")

# Coordinate search for source-specific thresholds.
t2, t3 = best_threshold, best_threshold
best_source_f = global_f
for _ in range(2):
    for cand in np.arange(max(0.50, t2 - 0.10), min(0.95, t2 + 0.1001), 0.025):
        f = macro_f05_thresholds(hold_s1, hold_src, hold_o, hold_probs, float(cand), t3, gt_keys, holdout_ids, n_s1, False)
        if f > best_source_f:
            t2, best_source_f = float(cand), f
    for cand in np.arange(max(0.50, t3 - 0.10), min(0.95, t3 + 0.1001), 0.025):
        f = macro_f05_thresholds(hold_s1, hold_src, hold_o, hold_probs, t2, float(cand), gt_keys, holdout_ids, n_s1, False)
        if f > best_source_f:
            t3, best_source_f = float(cand), f

# Fine local coordinate search.
for cand in np.arange(max(0.50, t2 - 0.03), min(0.95, t2 + 0.0301), 0.01):
    f = macro_f05_thresholds(hold_s1, hold_src, hold_o, hold_probs, float(cand), t3, gt_keys, holdout_ids, n_s1, False)
    if f > best_source_f:
        t2, best_source_f = float(cand), f
for cand in np.arange(max(0.50, t3 - 0.03), min(0.95, t3 + 0.0301), 0.01):
    f = macro_f05_thresholds(hold_s1, hold_src, hold_o, hold_probs, t2, float(cand), gt_keys, holdout_ids, n_s1, False)
    if f > best_source_f:
        t3, best_source_f = float(cand), f

print("=" * 75)
print(f"SOURCE-SPECIFIC THRESHOLDS: S2={t2:.3f}, S3={t3:.3f}")
print(f"HOLDOUT MACRO F0.5 (source thresholds): {best_source_f:.6f}")
print("=" * 75)

# Test whether one-to-one target consistency improves the holdout metric.
consistent_f = macro_f05_thresholds(hold_s1, hold_src, hold_o, hold_probs, t2, t3, gt_keys, holdout_ids, n_s1, True)
use_uniqueness = consistent_f > best_source_f
print(f"Holdout with target-uniqueness post-processing: {consistent_f:.6f}")
print(f"Use target-uniqueness at inference: {use_uniqueness}")

# ---------------------------------------------------------------------
# Final refit on ALL labelled S1 entities.
# ---------------------------------------------------------------------
print("\n[2] Refitting final model on ALL training S1 entities...")
all_train_mask = np.ones(n_s1, dtype=bool)
X_final, y_final = build_sample_matrix(all_train_mask)
print(f"Final training rows: {len(y_final):,}")
final_model = lgb.LGBMClassifier(
    n_estimators=350,
    num_leaves=63,
    learning_rate=0.05,
    min_child_samples=50,
    reg_lambda=1.0,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=-1,
)
final_model.fit(X_final, y_final)
model = final_model
del X_final, y_final, final_model, hold_probs
gc.collect()

best_threshold_s2 = float(t2)
best_threshold_s3 = float(t3)

model.booster_.save_model(f"{OUTPUT_DIR}/model.txt")
with open(f"{OUTPUT_DIR}/thresholds.txt", "w") as f:
    f.write(f"S2={best_threshold_s2:.4f}\nS3={best_threshold_s3:.4f}\n")
with open(f"{OUTPUT_DIR}/threshold.txt", "w") as f:
    f.write(str(float((best_threshold_s2 + best_threshold_s3) / 2.0)))
with open(f"{OUTPUT_DIR}/use_uniqueness.txt", "w") as f:
    f.write("1" if use_uniqueness else "0")

print("\n[PASS] Iteration-2 final model trained on all labelled training entities.")
print(f"Final thresholds: S2={best_threshold_s2:.3f}, S3={best_threshold_s3:.3f}")
print(f"Inference uniqueness post-processing: {use_uniqueness}")
print(f"Total training pipeline time: {(time.time() - t0) / 60:.2f} minutes")


# =====================================================================
# CELL 14 - TEST INFERENCE (ITERATION 2)
# =====================================================================

print("=" * 75)
print("ITERATION 2 TEST INFERENCE")
print("=" * 75)
t0 = time.time()


def load_test_source(path, source_prefix):
    raw = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
        usecols=["entity_id", "business_name", "business_address", "country"],
    )
    print(f"  Loaded {Path(path).name}: {len(raw):,} rows")
    if (~raw["entity_id"].str.startswith(source_prefix)).any():
        raise RuntimeError(f"{Path(path).name} contains invalid {source_prefix} entity IDs.")
    if raw["entity_id"].duplicated().any():
        raise RuntimeError(f"{Path(path).name} contains duplicate entity IDs.")
    raw["_row"] = np.arange(len(raw), dtype=np.int32)
    result = prepare_dataframe(raw)
    del raw
    gc.collect()
    return result

print("\n[1] Loading test sources...")
test_s1 = load_test_source(f"{TEST_DIR}/test_source1.tsv", "S1-")
test_s2 = load_test_source(f"{TEST_DIR}/test_source2.tsv", "S2-")
test_s3 = load_test_source(f"{TEST_DIR}/test_source3.tsv", "S3-")

train_countries = set(s1["country"].unique()) | set(s2["country"].unique()) | set(s3["country"].unique())
test_countries = set(test_s1["country"].unique()) | set(test_s2["country"].unique()) | set(test_s3["country"].unique())
print("Training countries:", sorted(train_countries))
print("Test countries:", sorted(test_countries))
print("Unseen test countries:", sorted(test_countries - train_countries))

print("\n[2] Blocking test data...")
test_pairs = block_all(test_s1, test_s2, test_s3)
n_test_pairs = len(test_pairs["s1_pos"])
if n_test_pairs == 0:
    raise RuntimeError("Test blocking generated zero candidates.")

test_counts = np.bincount(test_pairs["s1_pos"], minlength=len(test_s1))
print(f"Test candidate pairs: {n_test_pairs:,}")
print(f"Test S1 entities with candidates: {(test_counts > 0).sum():,}/{len(test_s1):,}")
print(f"Mean candidates/entity: {test_counts[test_counts > 0].mean():.2f}")
print(f"Maximum candidates/entity: {test_counts.max()}")
if test_counts.max() > CANDIDATES_PER_ENTITY:
    raise RuntimeError("Test candidate cap was violated.")

# Feature arrays.
test_arrays = {
    "s1_name": test_s1["name_norm"].to_numpy(dtype=object),
    "s1_core": test_s1["core_name"].to_numpy(dtype=object),
    "s1_compact": test_s1["compact_name"].to_numpy(dtype=object),
    "s1_addr": test_s1["addr_norm"].to_numpy(dtype=object),
    "s1_postal": test_s1["postal"].to_numpy(dtype=object),
    "s1_house": test_s1["house_number"].to_numpy(dtype=object),
    "s1_name_len": test_s1["name_len"].to_numpy(),
    "s1_addr_len": test_s1["addr_len"].to_numpy(),
    "s2_name": test_s2["name_norm"].to_numpy(dtype=object),
    "s2_core": test_s2["core_name"].to_numpy(dtype=object),
    "s2_compact": test_s2["compact_name"].to_numpy(dtype=object),
    "s2_addr": test_s2["addr_norm"].to_numpy(dtype=object),
    "s2_postal": test_s2["postal"].to_numpy(dtype=object),
    "s2_house": test_s2["house_number"].to_numpy(dtype=object),
    "s2_name_len": test_s2["name_len"].to_numpy(),
    "s2_addr_len": test_s2["addr_len"].to_numpy(),
    "s3_name": test_s3["name_norm"].to_numpy(dtype=object),
    "s3_core": test_s3["core_name"].to_numpy(dtype=object),
    "s3_compact": test_s3["compact_name"].to_numpy(dtype=object),
    "s3_addr": test_s3["addr_norm"].to_numpy(dtype=object),
    "s3_postal": test_s3["postal"].to_numpy(dtype=object),
    "s3_house": test_s3["house_number"].to_numpy(dtype=object),
    "s3_name_len": test_s3["name_len"].to_numpy(),
    "s3_addr_len": test_s3["addr_len"].to_numpy(),
}

print("\n[3] Scoring candidates...")
test_probs_parts = []
for start in range(0, n_test_pairs, FEATURE_CHUNK):
    end = min(start + FEATURE_CHUNK, n_test_pairs)
    X_chunk = compute_features_chunk(
        test_arrays,
        test_pairs["s1_pos"][start:end],
        test_pairs["src"][start:end],
        test_pairs["o_pos"][start:end],
        test_pairs["score"][start:end],
    )
    test_probs_parts.append(model.predict_proba(X_chunk)[:, 1].astype(np.float32))
    del X_chunk
    if start == 0 or start % (FEATURE_CHUNK * 10) == 0:
        print(f"  scored {end:,}/{n_test_pairs:,}")
test_probs = np.concatenate(test_probs_parts)
del test_probs_parts
gc.collect()

# Optional target uniqueness chosen from held-out data.
if use_uniqueness and len(test_probs):
    chosen = (np.where(test_pairs["src"] == 0, test_probs >= best_threshold_s2, test_probs >= best_threshold_s3))
    idx = np.flatnonzero(chosen)
    if len(idx):
        order = idx[np.argsort(-test_probs[idx], kind="stable")]
        seen2, seen3 = set(), set()
        keep = np.zeros(len(order), dtype=bool)
        for j, original_idx in enumerate(order):
            key = (int(test_pairs["src"][original_idx]), int(test_pairs["o_pos"][original_idx]))
            seen = seen2 if key[0] == 0 else seen3
            if key not in seen:
                seen.add(key)
                keep[j] = True
        final_pred_idx = order[keep]
    else:
        final_pred_idx = np.empty(0, dtype=np.int64)
else:
    final_pred_idx = np.flatnonzero(
        np.where(test_pairs["src"] == 0, test_probs >= best_threshold_s2, test_probs >= best_threshold_s3)
    )

final_pred_mask = np.zeros(n_test_pairs, dtype=bool)
final_pred_mask[final_pred_idx] = True

print(f"Predicted match edges: {int(final_pred_mask.sum()):,}")

print("\n[4] Writing submission files...")
s1_ids = test_s1["entity_id"].to_numpy(dtype=object)
s2_ids = test_s2["entity_id"].to_numpy(dtype=object)
s3_ids = test_s3["entity_id"].to_numpy(dtype=object)

order = np.argsort(test_pairs["s1_pos"], kind="stable")
sorted_s1 = test_pairs["s1_pos"][order]
sorted_src = test_pairs["src"][order]
sorted_o = test_pairs["o_pos"][order]
sorted_pred = final_pred_mask[order]

with open(f"{OUTPUT_DIR}/candidate_pairs.tsv", "w", encoding="utf-8") as fcand, open(f"{OUTPUT_DIR}/matching_results.tsv", "w", encoding="utf-8") as fmatch:
    fcand.write("source1_entity_id\tcandidate_entity_ids\n")
    fmatch.write("source1_entity_id\tmatched_entity_ids\n")
    ptr = 0
    total = len(sorted_s1)
    for s1_pos in range(len(test_s1)):
        while ptr < total and sorted_s1[ptr] < s1_pos:
            ptr += 1
        start = ptr
        while ptr < total and sorted_s1[ptr] == s1_pos:
            ptr += 1
        if start == ptr:
            fcand.write(f"{s1_ids[s1_pos]}\t\n")
            fmatch.write(f"{s1_ids[s1_pos]}\t\n")
            continue

        src = sorted_src[start:ptr]
        o = sorted_o[start:ptr]
        pred = sorted_pred[start:ptr]
        ids = np.empty(len(o), dtype=object)
        m2 = src == 0
        if m2.any():
            ids[m2] = s2_ids[o[m2]]
        m3 = ~m2
        if m3.any():
            ids[m3] = s3_ids[o[m3]]
        fcand.write(f"{s1_ids[s1_pos]}\t{','.join(ids.tolist())}\n")
        fmatch.write(f"{s1_ids[s1_pos]}\t{','.join(ids[pred].tolist())}\n")

print("\n[PASS] Output files written.")
print(f"  {OUTPUT_DIR}/candidate_pairs.tsv")
print(f"  {OUTPUT_DIR}/matching_results.tsv")
print(f"Total inference time: {(time.time() - t0) / 60:.2f} minutes")


# =====================================================================
# CELL 15 - OUTPUT INTEGRITY CHECK + OFFICIAL VALIDATOR
# =====================================================================

print("=" * 75)
print("FINAL OUTPUT VALIDATION")
print("=" * 75)


matching_path = (
    f"{OUTPUT_DIR}/matching_results.tsv"
)

candidate_path = (
    f"{OUTPUT_DIR}/candidate_pairs.tsv"
)


# -------------------------------------------------------------
# Basic filesystem checks
# -------------------------------------------------------------

assert os.path.isfile(
    matching_path
)

assert os.path.isfile(
    candidate_path
)

print(
    "[PASS] Both output files exist."
)


# -------------------------------------------------------------
# Header checks
# -------------------------------------------------------------

matching_header = (
    pd.read_csv(
        matching_path,
        sep="\t",
        nrows=0
    ).columns.tolist()
)

candidate_header = (
    pd.read_csv(
        candidate_path,
        sep="\t",
        nrows=0
    ).columns.tolist()
)

assert matching_header == [
    "source1_entity_id",
    "matched_entity_ids",
]

assert candidate_header == [
    "source1_entity_id",
    "candidate_entity_ids",
]

print(
    "[PASS] Output headers are correct."
)


# -------------------------------------------------------------
# Run official validator
# -------------------------------------------------------------

print(
    "\nRunning official validator..."
)

result = subprocess.run(
    [
        sys.executable,
        VALIDATOR_PATH,
        "--matching",
        matching_path,
        "--candidate",
        candidate_path,
        "--test-dir",
        TEST_DIR,
    ],
    capture_output=True,
    text=True,
)

print(
    result.stdout
)

if result.stderr:
    print(
        result.stderr
    )


if result.returncode != 0:

    raise RuntimeError(
        "Official submission validator failed."
    )

print(
    "\n" + "=" * 75
)

print(
    "[PASS] OFFICIAL VALIDATOR PASSED"
)

print(
    "=" * 75
)


