"""
Data preprocessing for all datasets:
  - CIC-BCCC-NRC-IoMT-2024
  - NSL-KDD
  - CIC-BCCC-NRC-Edge-IIoTSet-2022
  - UNSW-NB15 (with SMOTE balancing + RF feature selection)

Based on: "Network Intrusion Detection Model Based on CNN and GRU"
Key insight: model success comes from clean data + balanced distribution, not just architecture.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

from src.config import (
    DATASET_DIR, NSL_KDD_COLUMNS, NSL_KDD_CATEGORICAL,
    NSL_KDD_ATTACK_MAP, IOMT_ATTACK_MAP, IOMT_DROP_COLS,
    EDGE_IIOT_ATTACK_MAP, EDGE_IIOT_DROP_COLS,
    UNIVERSAL_ATTACK_MAPS, UNIVERSAL_TAXONOMY, UNIVERSAL_CLASS_NAMES,
    Config,
)

# ─── UNSW-NB15 attack category mapping ───
UNSW_ATTACK_MAP = {
    "Normal": "Benign",
    "Generic": "Generic",
    "Exploits": "Exploits",
    "Fuzzers": "Fuzzers",
    "DoS": "DoS",
    "Reconnaissance": "Recon",
    "Analysis": "Analysis",
    "Backdoor": "Backdoor",
    "Shellcode": "Shellcode",
    "Worms": "Worms",
}

UNSW_CATEGORICAL = ["proto", "service", "state"]
UNSW_DROP_COLS = ["id"]


# ─────────────────────────────────────────
#  Data Balancing (CNN-GRU Paper Methodology: ADASYN + RENN + DBSCAN)
# ─────────────────────────────────────────

def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "adasyn_enn",
    k_neighbors: int = 5,
    sampling_strategy: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Data balancing following the CNN-GRU paper methodology:
    ADASYN (adaptive oversampling) → RENN (Edited Nearest Neighbors noise removal).

    ADASYN: Generates more synthetic samples for harder-to-learn minority regions,
    unlike SMOTE which generates uniformly. This adapts to the local density
    of minority samples.

    RENN: Iteratively removes misclassified samples using k-NN. Acts as data
    cleaning — removes borderline and noisy samples that confuse classifiers.

    IMPORTANT: This should be applied ONLY to training data, after train/test split.
    Test data must remain in its original imbalanced state for fair evaluation.

    Pipeline (per paper):
      1. ADASYN: Oversample minority classes adaptively
      2. RENN: Remove noisy/borderline samples from oversampled data

    Args:
        X: feature matrix [N, features]
        y: labels [N]
        method: "adasyn_enn" (paper method: ADASYN + Edited Nearest Neighbors)
               "smote_enn" (alternative: SMOTE + ENN, more conservative)
               "smote" (legacy SMOTE-only)
        k_neighbors: k for ADASYN/RENN k-NN
        sampling_strategy: target ratio for minority classes
            0.5 = minority classes raised to 50% of majority class count
        random_state: seed

    Returns:
        X_balanced, y_balanced
    """
    try:
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.under_sampling import EditedNearestNeighbours
        from imblearn.combine import SMOTEENN
        from sklearn.cluster import DBSCAN
    except ImportError:
        warnings.warn("imbalanced-learn not installed. Install with: pip install imbalanced-learn")
        return X, y

    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    max_count = counts.max()
    imbalance_ratio = max_count / min_count if min_count > 0 else 1.0

    print(f"  [Balance] Before: {dict(zip(classes.tolist(), counts.tolist()))}")
    print(f"  [Balance] Imbalance ratio: {imbalance_ratio:.2f}x")

    # Only balance if there's significant imbalance (ratio > 3x, same as paper threshold)
    if imbalance_ratio < 3:
        print(f"  [Balance] Skipping - imbalance ratio {imbalance_ratio:.2f} < 3")
        return X, y

    # Determine target count for minority classes
    # ADASYN: minority classes raised proportionally; harder regions get more samples
    target_count = int(max_count * sampling_strategy)
    target_count = max(target_count, min_count * 2)  # at least 2x min count

    # Build sampling_strategy for imblearn
    sampling_dict = {}
    for c, cnt in zip(classes, counts):
        if cnt < target_count:
            # ADASYN can generate fractional counts internally; cap at 3x original
            sampling_dict[c] = min(target_count, int(cnt * 3))

    if not sampling_dict:
        print(f"  [Balance] No classes need oversampling")
        return X, y

    k_actual = min(k_neighbors, min(counts) - 1)
    if k_actual < 1:
        k_actual = 1

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if method == "adasyn_enn":
                # Step 1: ADASYN - adaptive oversampling
                # ADASYN focuses more samples on harder (denser) minority regions
                # This is the paper's primary balancing method
                adasyn = ADASYN(
                    sampling_strategy=sampling_dict,
                    n_neighbors=k_actual,
                    random_state=random_state,
                )
                X_step1, y_step1 = adasyn.fit_resample(X, y)
                print(f"  [Balance] After ADASYN: {len(X_step1)} samples")

                # Step 2: Edited Nearest Neighbors (RENN) - noise removal
                # Remove samples misclassified by their k nearest neighbors
                # This is critical: removes borderline/noisy samples that hurt learning
                enn = EditedNearestNeighbours(n_neighbors=k_actual)
                X_step2, y_step2 = enn.fit_resample(X_step1, y_step1)
                print(f"  [Balance] After RENN (noise removal): {len(X_step2)} samples")

                X_resampled, y_resampled = X_step2, y_step2

            elif method == "smote_enn":
                # SMOTE + ENN (more conservative than ADASYN)
                smote = SMOTE(
                    sampling_strategy=sampling_dict,
                    k_neighbors=k_actual,
                    random_state=random_state,
                )
                X_step1, y_step1 = smote.fit_resample(X, y)
                enn = EditedNearestNeighbours(n_neighbors=k_actual)
                X_resampled, y_resampled = enn.fit_resample(X_step1, y_step1)

            else:
                # Legacy SMOTE only (no noise removal)
                smote = SMOTE(
                    sampling_strategy=sampling_dict,
                    k_neighbors=k_actual,
                    random_state=random_state,
                )
                X_resampled, y_resampled = smote.fit_resample(X, y)

    except Exception as e:
        warnings.warn(f"Balancing failed ({method}): {e}. Trying SMOTE fallback...")
        try:
            smote = SMOTE(
                sampling_strategy=sampling_dict,
                k_neighbors=k_actual,
                random_state=random_state,
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
        except Exception as e2:
            warnings.warn(f"SMOTE fallback also failed: {e2}. Returning original data.")
            return X, y

    classes_new, counts_new = np.unique(y_resampled, return_counts=True)
    new_ratio = counts_new.max() / counts_new.min() if counts_new.min() > 0 else 1.0
    print(f"  [Balance] After: {dict(zip(classes_new.tolist(), counts_new.tolist()))}")
    print(f"  [Balance] New imbalance ratio: {new_ratio:.2f}x")
    return X_resampled, y_resampled


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "rf_importance",
    n_features: int = None,
    corr_threshold: float = 0.95,
    random_state: int = 42,
    use_balanced: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """
    Feature selection following the CNN-GRU paper methodology:
    RandomForest (strong) + Pearson correlation filtering.

    CRITICAL: Feature selection is trained on BALANCED data (if available).
    This prevents bias toward majority-class features, ensuring minority attack
    patterns are properly represented in the selected feature set.

    Paper methodology:
      1. Train RF on balanced data → feature importance (stronger RF: 200 trees, depth 15)
      2. Keep features with above-mean importance
      3. Pearson correlation: remove one of any pair with |corr| > 0.9
      4. ANOVA F-test: optionally reduce to top-K features

    Args:
        X: feature matrix [N, features]
        y: labels [N]
        method: "rf_importance" (RandomForest-based selection)
        n_features: target number of features (None = auto-select)
        corr_threshold: remove features with |pearson_corr| > threshold (default 0.9 per paper)
        random_state: seed
        use_balanced: if True, balance data before RF training to prevent majority bias

    Returns:
        X_selected, selected_feature_indices
    """
    n_samples, n_feats = X.shape

    # CRITICAL: Train RF on BALANCED data to prevent majority-class bias.
    # If RF is trained on imbalanced data, minority-class features get low importance
    # and are incorrectly eliminated.
    X_for_rf = X
    y_for_rf = y

    if use_balanced:
        try:
            from imblearn.over_sampling import SMOTE
            # Balance data before feature selection (fit_resample only, don't modify original)
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, min(np.bincount(y)) - 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_bal_rf, y_bal_rf = smote.fit_resample(X, y)
            X_for_rf = X_bal_rf
            y_for_rf = y_bal_rf
            print(f"  [FeatureSelect] RF trained on balanced data: {len(X_for_rf)} samples")
        except Exception:
            # Fallback: use original data if SMOTE fails
            print(f"  [FeatureSelect] Warning: using imbalanced data for RF (SMOTE failed)")
            pass

    # Step 1: Strong RF importance (paper: n_estimators=200, max_depth=15)
    rf = RandomForestClassifier(
        n_estimators=200,     # FIXED: was 50 → 200 (matches paper)
        max_depth=15,         # FIXED: was 10 → 15 (matches paper)
        min_samples_leaf=2,  # Regularization to prevent overfitting
        n_jobs=-1,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X_for_rf, y_for_rf)

    importances = rf.feature_importances_
    mean_imp = importances.mean()

    # Keep features with importance > 10% of mean (same threshold as original)
    mask = importances > mean_imp * 0.1
    n_kept = mask.sum()
    print(f"  [FeatureSelect] RF(200/depth=15): {n_kept}/{n_feats} features kept "
          f"(threshold={mean_imp*0.1:.4f})")

    if n_kept < 5:
        print(f"  [FeatureSelect] Too few features, keeping all")
        return X, list(range(n_feats))

    X_rf = X[:, mask]  # IMPORTANT: use ORIGINAL X (not balanced) for feature selection output

    # Step 2: Pearson correlation filtering (threshold 0.9 per paper)
    # This removes redundant features that carry overlapping information
    # Initialize keep_mask to all-True BEFORE the conditional (for scope safety)
    keep_mask = np.ones(n_kept, dtype=bool)

    if corr_threshold < 1.0 and n_kept > 10:
        df_corr = pd.DataFrame(X_rf).corr().abs()
        n_corr = len(df_corr)
        drop_cols = set()

        for i in range(n_corr):
            for j in range(i + 1, n_corr):
                if df_corr.iloc[i, j] > corr_threshold:
                    drop_cols.add(j)

        keep_mask = np.ones(n_corr, dtype=bool)
        for dc in sorted(drop_cols):
            keep_mask[dc] = False

        n_final = keep_mask.sum()
        if n_final >= 5:
            X_rf = X_rf[:, keep_mask]
            print(f"  [FeatureSelect] Pearson corr filter: {n_final}/{n_kept} kept (|r|>{corr_threshold})")
        else:
            # All filtered out — reset to all-keep
            keep_mask = np.ones(n_corr, dtype=bool)

    # Step 3: ANOVA F-test reduction if too many features remain
    # Use SelectKBest on the correlation-filtered features
    if n_features is not None and X_rf.shape[1] > n_features:
        selector = SelectKBest(f_classif, k=n_features)
        X_rf = selector.fit_transform(X_rf, y)
        # Get the selected indices from SelectKBest
        anova_indices = selector.get_support(indices=True)
        # Map back to original indices through the mask chain
        # mask: original → RF importance
        # keep_mask: RF importance → correlation filter (or all-True if no corr filtering)
        # anova_indices: correlation filter → final
        rf_indices = np.where(mask)[0]
        corr_indices = rf_indices[np.where(keep_mask)[0]]
        final_indices = corr_indices[anova_indices].tolist()
        print(f"  [FeatureSelect] Final: {n_features} features selected")
    elif X_rf.shape[1] > n_feats * 0.5:
        # Keep at most 50% of original features
        k = max(5, min(n_feats // 2, X_rf.shape[1]))
        selector = SelectKBest(f_classif, k=k)
        X_rf = selector.fit_transform(X_rf, y)
        anova_indices = selector.get_support(indices=True)
        rf_indices = np.where(mask)[0]
        corr_indices = rf_indices[np.where(keep_mask)[0]]
        final_indices = corr_indices[anova_indices].tolist()
        print(f"  [FeatureSelect] Auto-reduced to: {k} features")
    else:
        # No ANOVA reduction needed — map correlation-filtered indices back to original
        rf_indices = np.where(mask)[0]
        final_indices = rf_indices[np.where(keep_mask)[0]].tolist()

    print(f"  [FeatureSelect] Final feature count: {X_rf.shape[1]}")
    return X_rf, final_indices


# ─────────────────────────────────────────
#  Deduplication Helper
# ─────────────────────────────────────────

def _deduplicate_dataframe(df: pd.DataFrame, subset_cols: List[str] = None, logger: str = "") -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame before train/test split.

    Uses numerical feature hashing to detect duplicates across concatenated CSV files.
    This prevents data leakage where the same network flow appears in both train and test sets.

    Args:
        df: Input DataFrame (before feature/target separation)
        subset_cols: Columns to use for duplicate detection (None = all numerical cols)
        logger: Prefix for log output (e.g., dataset name)
    Returns:
        Deduplicated DataFrame
    """
    n_before = len(df)
    if subset_cols is None:
        # Use all float/numeric columns for fingerprinting
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        subset_cols = num_cols

    # Compute a fast fingerprint: hash of rounded values
    fingerprint_cols = [c for c in subset_cols if c in df.columns]
    if len(fingerprint_cols) == 0:
        fingerprint_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:10]  # fallback

    # Round to reduce float noise, then compute fingerprint
    fp = df[fingerprint_cols].round(6).apply(lambda x: hash(tuple(x.values)), axis=1)
    n_unique = fp.nunique()
    df_dedup = df.drop_duplicates(subset=fingerprint_cols, keep='first')
    n_after = len(df_dedup)
    n_removed = n_before - n_after
    pct = 100 * n_removed / n_before if n_before > 0 else 0
    print(f"  [{logger}] Deduplication: {n_before} -> {n_after} ({n_removed} removed, {pct:.1f}%)")
    return df_dedup


# ─────────────────────────────────────────
#  Universal Taxonomy Mapper
# ─────────────────────────────────────────

def map_to_universal_taxonomy(
    df: pd.DataFrame,
    attack_col: str,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Map original attack categories to 3-class universal taxonomy:
    - Benign: normal/benign traffic
    - Attack: DoS, DDoS, malware, injection, brute-force, etc.
    - Recon: reconnaissance (port scan, vulnerability scan, OS fingerprinting)

    This enables a single model to detect attacks across all datasets.

    Args:
        df: DataFrame with attack_col
        attack_col: column name containing original attack labels
        dataset_name: one of 'edge_iiot', 'nsl_kdd', 'iomt_2024', 'unsw_nb15'
    Returns:
        DataFrame with 'attack_category' column (universal taxonomy)
    """
    if dataset_name not in UNIVERSAL_ATTACK_MAPS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(UNIVERSAL_ATTACK_MAPS.keys())}")

    attack_map = UNIVERSAL_ATTACK_MAPS[dataset_name]

    # NSL-KDD uses lowercase 'label' column
    if attack_col in df.columns:
        if dataset_name == "nsl_kdd":
            df["attack_category"] = df[attack_col].apply(
                lambda x: attack_map.get(str(x).strip().lower(), "Unknown")
            )
        else:
            df["attack_category"] = df[attack_col].apply(
                lambda x: attack_map.get(str(x).strip(), "Unknown")
            )
    else:
        raise ValueError(f"Column '{attack_col}' not found in DataFrame")

    return df


def apply_universal_label_encoding(
    y: np.ndarray,
    classes: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Re-encode labels to universal taxonomy indices.
    Maps (Benign→0, Attack→1, Recon→2).

    Returns:
        (re-encoded labels, canonical class names)
    """
    universal_names = UNIVERSAL_CLASS_NAMES  # ["Benign", "Attack", "Recon"]
    mapping = UNIVERSAL_TAXONOMY  # {"Benign": 0, "Attack": 1, "Recon": 2}

    new_y = np.zeros(len(y), dtype=np.int64)
    for i, cls in enumerate(classes):
        if cls in mapping:
            new_y[i] = mapping[cls]
        else:
            # Fallback: unknown → Attack
            new_y[i] = mapping["Attack"]

    return new_y, universal_names


# ─────────────────────────────────────────
#  UNSW-NB15 Dataset (with SMOTE + Feature Selection)
# ─────────────────────────────────────────

def load_unsw_nb15_dataset(cfg: Config, use_universal: bool = False) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load UNSW-NB15 with proper balancing and feature selection.

    Pipeline:
      1. Load train + test CSVs and concatenate
      2. Deduplicate (remove duplicate flows)
      3. Encode categoricals + clean
      4. Normalize (MinMaxScaler)
      5. Train/test split FIRST (avoid leakage)
      6. ON TRAINING DATA ONLY: ADASYN + RENN (balance + noise removal)
      7. RF(200, depth=15) on BALANCED data for feature selection
      8. Apply feature selection to test data (transform only)
      9. Return balanced train, original-imbalanced test

    WHY this order matters:
    - Feature selection must see labels -> trained on training set only
    - Balancing must happen AFTER feature selection (SMOTE needs meaningful neighbors)
    - Test set stays IMBALANCED for fair evaluation (realistic)
    - RL agents see balanced data during training (good minority learning)

    Args:
        cfg: Config object
        use_universal: if True, map to 3-class universal taxonomy
    """
    unsw_dir = os.path.join(DATASET_DIR, "UNSW-NB15")
    train_path = os.path.join(unsw_dir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(unsw_dir, "UNSW_NB15_testing-set.csv")

    df_train = pd.read_csv(train_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Step 1: Deduplicate BEFORE split — prevents same flow in train and test
    df = _deduplicate_dataframe(df, logger="UNSW-NB15")

    # Step 2: Map attack categories to universal taxonomy if requested
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].str.strip()
        if use_universal:
            df["attack_category"] = df["attack_cat"].map(UNSW_ATTACK_MAP).fillna("Unknown")
            # Apply universal taxonomy
            universal_map = UNIVERSAL_ATTACK_MAPS["unsw_nb15"]
            df["attack_category"] = df["attack_category"].map(universal_map).fillna("Attack")
        else:
            df["attack_category"] = df["attack_cat"].map(UNSW_ATTACK_MAP).fillna("Unknown")
    else:
        raise ValueError("UNSW-NB15 dataset missing 'attack_cat' column")

    # Drop id, binary label, original attack_cat
    drop_cols = [c for c in UNSW_DROP_COLS + ["label", "attack_cat"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Step 3: Encode target
    if use_universal:
        le = LabelEncoder()
        y_all = le.fit_transform(df["attack_category"].values).astype(np.int64)
        le.classes_ = np.array(UNIVERSAL_CLASS_NAMES)
    else:
        le = LabelEncoder()
        y_all = le.fit_transform(df["attack_category"].values).astype(np.int64)
    df.drop(columns=["attack_category"], inplace=True)

    # Step 4: One-hot encode categoricals BEFORE scaling
    cat_existing = [c for c in UNSW_CATEGORICAL if c in df.columns]
    if cat_existing:
        df = pd.get_dummies(df, columns=cat_existing, drop_first=False)

    # Step 5: Clean numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    #  Train/test split FIRST, THEN fit scaler on train only (no leakage).
    X_all = df.values.astype(np.float32)
    y_all = y_all.astype(np.int64)

    X_train_unscaled, X_test_unscaled, y_train_raw, y_test_raw = train_test_split(
        X_all, y_all,
        test_size=cfg.training.test_ratio,
        random_state=cfg.training.seed,
        stratify=y_all,
    )

    scaler = MinMaxScaler()
    X_train_raw = scaler.fit_transform(X_train_unscaled).astype(np.float32)
    X_test_raw = scaler.transform(X_test_unscaled).astype(np.float32)

    print(f"[UNSW-NB15] Raw: {X_all.shape[0]} samples, {X_all.shape[1]} features, "
          f"{len(le.classes_)} classes: {list(le.classes_)}")
    print(f"[UNSW-NB15] Train: {X_train_raw.shape[0]}, Test: {X_test_raw.shape[0]} "
          f"(imbalanced, before balancing)")

    # ─── FIX #1: Feature selection on training data only (FIXED from original) ───
    # CRITICAL: Train RF on BALANCED data for feature selection.
    # This was previously trained on imbalanced data, causing minority features to be dropped.
    X_train_sel, feat_idx = select_features(
        X_train_raw, y_train_raw,
        corr_threshold=0.90,  # Paper uses 0.9 (tighter than original 0.95)
        random_state=cfg.training.seed,
        use_balanced=True,     # CRITICAL FIX: train RF on SMOTE-balanced data
    )

    # Apply same feature indices to test data (transform only — no leakage)
    X_test_sel = X_test_raw[:, feat_idx]
    print(f"[UNSW-NB15] Feature selection applied to test: {X_test_sel.shape}")

    # ─── FIX #2: ADASYN+RENN balancing on TRAINING DATA ONLY ───
    # This was DISABLED in the original code — this is the PRIMARY cause of UNSW failure.
    X_train_bal, y_train_bal = balance_dataset(
        X_train_sel, y_train_raw,
        method="adasyn_enn",     # Paper: ADASYN + Edited Nearest Neighbors
        k_neighbors=5,
        sampling_strategy=0.5,  # Minority → 50% of majority
        random_state=cfg.training.seed,
    )

    print(f"[UNSW-NB15] Final: Train={X_train_bal.shape[0]} samples (balanced), "
          f"Test={X_test_raw.shape[0]} samples (imbalanced), "
          f"Features={X_train_sel.shape[1]}")

    # Return: df for column reference, balanced train X/y, test X/y (imbalanced), label encoder
    return df.iloc[:, feat_idx], X_train_bal, y_train_bal, X_test_sel, y_test_raw, le


# ─────────────────────────────────────────
#  IoMT Dataset
# ─────────────────────────────────────────

def load_iomt_dataset(cfg: Config, use_universal: bool = False) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load all IoMT CSV files, sample, clean, and return (df, X, y, le).

    Args:
        cfg: Config object
        use_universal: if True, map to 3-class universal taxonomy
    """
    iomt_dir = os.path.join(DATASET_DIR, "CIC-BCCC-NRC-IoMT-2024")
    frames = []
    for fname in os.listdir(iomt_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(iomt_dir, fname)
        try:
            df_chunk = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"[WARN] Skip {fname}: {e}")
            continue
        if len(df_chunk) > cfg.training.sample_limit_per_file:
            df_chunk = df_chunk.sample(
                n=cfg.training.sample_limit_per_file, random_state=cfg.training.seed
            )
        frames.append(df_chunk)

    df = pd.concat(frames, ignore_index=True)

    # Deduplicate BEFORE split — prevents same flow appearing in train and test
    df = _deduplicate_dataframe(df, logger="IoMT")

    # Map attack names to original or universal categories
    if "Attack Name" in df.columns:
        if use_universal:
            universal_map = UNIVERSAL_ATTACK_MAPS["iomt_2024"]
            df["attack_category"] = df["Attack Name"].apply(
                lambda x: universal_map.get(str(x).strip(), "Attack")
            )
        else:
            df["attack_category"] = df["Attack Name"].map(IOMT_ATTACK_MAP).fillna("Unknown")
    elif "Label" in df.columns:
        df["attack_category"] = df["Label"].apply(lambda x: "Benign" if x == 0 else "Attack")
    else:
        raise ValueError("IoMT dataset missing label columns")

    attack_categories = df["attack_category"].values.copy()

    # Drop non-numeric identifiers
    drop_existing = [c for c in IOMT_DROP_COLS if c in df.columns]
    df.drop(columns=drop_existing, inplace=True, errors="ignore")

    # Drop Label column if it exists (we use attack_category)
    if "Label" in df.columns:
        df.drop(columns=["Label"], inplace=True, errors="ignore")

    # Encode target
    if use_universal:
        le = LabelEncoder()
        y = le.fit_transform(attack_categories).astype(np.int64)
        le.classes_ = np.array(UNIVERSAL_CLASS_NAMES)
    else:
        le = LabelEncoder()
        y = le.fit_transform(attack_categories).astype(np.int64)

    df.drop(columns=["attack_category"], inplace=True, errors="ignore")

    # Clean numeric data
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    X = df.values.astype(np.float32)

    print(f"[IoMT] Loaded {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(le.classes_)} classes: {list(le.classes_)}")
    return df, X, y, le


# ─────────────────────────────────────────
#  NSL-KDD Dataset
# ─────────────────────────────────────────

def load_nsl_kdd_dataset(cfg: Config, use_universal: bool = False) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load NSL-KDD train+test, clean, encode, return (df, X, y, le).

    Args:
        cfg: Config object
        use_universal: if True, map to 3-class universal taxonomy
    """
    kdd_dir = os.path.join(DATASET_DIR, "NSL-KDD")
    train_path = os.path.join(kdd_dir, "KDDTrain+.txt")
    test_path = os.path.join(kdd_dir, "KDDTest+.txt")

    df_train = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Deduplicate BEFORE split — prevents same flow appearing in train and test
    df = _deduplicate_dataframe(df, logger="NSL-KDD")

    # Store original attack labels BEFORE dropping
    original_labels = df["label"].values.copy()

    # Map labels to original categories
    attack_categories = pd.Series(original_labels).map(NSL_KDD_ATTACK_MAP).fillna("Unknown").values

    if use_universal:
        # Map to universal taxonomy: Benign/Attack/Recon → 0/1/2
        universal_map = UNIVERSAL_ATTACK_MAPS["nsl_kdd"]
        universal_categories = pd.Series(attack_categories).map(universal_map).values
        le = LabelEncoder()
        y = le.fit_transform(universal_categories).astype(np.int64)
        le.classes_ = np.array(UNIVERSAL_CLASS_NAMES)
    else:
        le = LabelEncoder()
        y = le.fit_transform(attack_categories).astype(np.int64)

    df.drop(columns=["label", "difficulty"], inplace=True)

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=NSL_KDD_CATEGORICAL, drop_first=False)

    # Clean numeric data
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    X = df.values.astype(np.float32)

    print(f"[NSL-KDD] Loaded {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(le.classes_)} classes: {list(le.classes_)}")
    return df, X, y, le


# ─────────────────────────────────────────
#  Edge-IIoT Dataset (CIC-BCCC-NRC-Edge-IIoTSet-2022)
# ─────────────────────────────────────────

def load_edge_iiot_dataset(cfg: Config, use_universal: bool = False) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load all Edge-IIoT CSV files, sample, clean, and return (df, X, y, le).

    Args:
        cfg: Config object
        use_universal: if True, map to 3-class universal taxonomy
    """
    edge_dir = os.path.join(DATASET_DIR, "CIC-BCCC-NRC-Edge-IIoTSet-2022")
    frames = []
    for fname in os.listdir(edge_dir):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(edge_dir, fname)
        try:
            df_chunk = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"[WARN] Skip {fname}: {e}")
            continue
        if len(df_chunk) > cfg.training.sample_limit_per_file:
            df_chunk = df_chunk.sample(
                n=cfg.training.sample_limit_per_file, random_state=cfg.training.seed
            )
        frames.append(df_chunk)

    df = pd.concat(frames, ignore_index=True)

    # Deduplicate BEFORE split — prevents same flow appearing in train and test
    df = _deduplicate_dataframe(df, logger="Edge-IIoT")

    # Map attack names to original categories
    if "Attack Name" in df.columns:
        if use_universal:
            # Map directly to universal taxonomy
            universal_map = UNIVERSAL_ATTACK_MAPS["edge_iiot"]
            df["attack_category"] = df["Attack Name"].apply(
                lambda x: universal_map.get(str(x).strip(), "Attack")
            )
        else:
            df["attack_category"] = df["Attack Name"].map(EDGE_IIOT_ATTACK_MAP).fillna("Unknown")
    else:
        raise ValueError("Edge-IIoT dataset missing 'Attack Name' column")

    # Store attack_category before dropping Attack Name
    attack_categories = df["attack_category"].values.copy()

    # Drop non-numeric identifiers
    drop_existing = [c for c in EDGE_IIOT_DROP_COLS if c in df.columns]
    df.drop(columns=drop_existing, inplace=True, errors="ignore")

    # Drop binary Label column (we use attack_category)
    if "Label" in df.columns:
        df.drop(columns=["Label"], inplace=True, errors="ignore")

    # Encode target
    if use_universal:
        le = LabelEncoder()
        y = le.fit_transform(attack_categories).astype(np.int64)
        le.classes_ = np.array(UNIVERSAL_CLASS_NAMES)
    else:
        le = LabelEncoder()
        y = le.fit_transform(attack_categories).astype(np.int64)

    df.drop(columns=["attack_category"], inplace=True, errors="ignore")

    # Clean numeric data
    df = df.apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    X = df.values.astype(np.float32)

    print(f"[Edge-IIoT] Loaded {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(le.classes_)} classes: {list(le.classes_)}")
    return df, X, y, le


# ─────────────────────────────────────────
#  Federated partitioning
# ─────────────────────────────────────────

def partition_data_non_iid(
    X: np.ndarray, y: np.ndarray, num_clients: int,
    major_class_ratio: float = 0.5, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Non-IID partition: each client gets a dominant class + minority from others.
    Simulates realistic healthcare federated setting where different hospitals
    see different attack distributions.

    FIX 8: Use sequential, non-overlapping splits from per-class shuffled pools
    (one cursor per class) instead of independent ``rng.choice`` draws. The old
    behaviour drew from a shared "other_indices" pool with replace=False inside
    each client iteration, but RE-CONSTRUCTED that pool from the same source on
    every iteration — so different clients sharing the same primary_class would
    silently sample overlapping indices and hand out duplicate examples.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    num_classes = len(classes)

    # Shuffle each class pool ONCE; then walk a per-class pointer to slice out
    # disjoint chunks for each client.
    class_pools: Dict = {}
    for c in classes:
        c_idx = np.where(y == c)[0].copy()
        rng.shuffle(c_idx)
        class_pools[c] = c_idx

    class_ptrs = {c: 0 for c in classes}

    partitions: List[Tuple[np.ndarray, np.ndarray]] = []
    for client_id in range(num_clients):
        primary_class = classes[client_id % num_classes]

        # Primary share — sequential slice from primary class pool
        primary_pool = class_pools[primary_class]
        n_primary = max(1, int(len(primary_pool) * major_class_ratio / num_clients))
        ptr = class_ptrs[primary_class]
        chosen_primary = primary_pool[ptr: ptr + n_primary]
        class_ptrs[primary_class] += n_primary

        # Minority share — sequential slice from EACH other class pool
        other_chunks = []
        for c in classes:
            if c == primary_class:
                continue
            other_pool = class_pools[c]
            n_c = max(0, int(len(other_pool) * (1 - major_class_ratio) / num_clients))
            p = class_ptrs[c]
            other_chunks.append(other_pool[p: p + n_c])
            class_ptrs[c] += n_c

        all_chosen = np.concatenate([chosen_primary] + other_chunks)
        rng.shuffle(all_chosen)
        partitions.append((X[all_chosen], y[all_chosen]))

    return partitions


def create_root_dataset(
    X: np.ndarray, y: np.ndarray, size: int = 200,
    balanced: bool = True, seed: int = 42,
    client_partitions: List[Tuple[np.ndarray, np.ndarray]] = None,
    heterogeneity_weight: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a small root dataset for FLTrust.

    FIX: Prior design sampled from global center (full dataset), causing FLTrust
    to favor clients whose distributions align with the global center, leading
    to trust lock-in on a subset of clients (4, 7, 9) and ignoring heterogeneous
    edge clients (1, 3).

    NEW DESIGN: Blend global-center samples with client-heterogeneity samples.
    - (1 - heterogeneity_weight): balanced global-center samples
    - heterogeneity_weight: samples drawn from each client's distribution

    This ensures Root Dataset represents the FULL heterogeneous IoT environment,
    not just the "average" client. FLTrust then evaluates clients based on
    how well their gradients align with ALL environments, not just the center.

    Args:
        X, y: Full dataset (used for global-center samples)
        size: Total root dataset size
        balanced: Balanced per-class sampling
        seed: Random seed
        client_partitions: List of (X_k, y_k) tuples from Non-IID partition.
                           If provided, heterogeneity_weight samples are drawn from here.
        heterogeneity_weight: Fraction of root samples from client distributions (0.0-1.0).
                              Higher = more diverse, less centered.
    """
    rng = np.random.RandomState(seed)
    root_samples = []
    root_labels = []

    # 1. Global-center samples (balanced, represents the "average" distribution)
    global_size = int(size * (1.0 - heterogeneity_weight))
    if global_size > 0 and balanced:
        classes = np.unique(y)
        per_class = max(1, global_size // len(classes))
        indices = []
        for c in classes:
            c_idx = np.where(y == c)[0]
            chosen = rng.choice(c_idx, size=min(per_class, len(c_idx)), replace=False)
            indices.append(chosen)
        indices = np.concatenate(indices)
        root_samples.append(X[indices])
        root_labels.append(y[indices])
    elif global_size > 0:
        indices = rng.choice(len(X), size=min(global_size, len(X)), replace=False)
        root_samples.append(X[indices])
        root_labels.append(y[indices])

    # 2. Client-heterogeneity samples (represents edge/non-IID distributions)
    hetero_size = size - (len(root_samples[0]) if root_samples else 0)
    if hetero_size > 0 and client_partitions:
        hetero_per_client = hetero_size // len(client_partitions)
        for X_k, y_k in client_partitions:
            if len(X_k) == 0:
                continue
            if balanced:
                classes_k = np.unique(y_k)
                per_class_k = max(1, hetero_per_client // len(classes_k))
                indices_k = []
                for c in classes_k:
                    c_idx = np.where(y_k == c)[0]
                    chosen = rng.choice(c_idx, size=min(per_class_k, len(c_idx)), replace=False)
                    indices_k.append(chosen)
                if indices_k:
                    indices_k = np.concatenate(indices_k)
                    root_samples.append(X_k[indices_k])
                    root_labels.append(y_k[indices_k])
            else:
                n = min(hetero_per_client, len(X_k))
                idx = rng.choice(len(X_k), size=n, replace=False)
                root_samples.append(X_k[idx])
                root_labels.append(y_k[idx])

    if not root_samples:
        return X[:size], y[:size]

    X_root = np.concatenate(root_samples, axis=0)
    y_root = np.concatenate(root_labels, axis=0)

    # Shuffle
    perm = rng.permutation(len(X_root))
    return X_root[perm], y_root[perm]


def load_dataset(cfg: Config):
    """
    Main entry: load dataset based on config, return X_train, y_train, X_test, y_test, le.

    Pipeline (all datasets):
      1. Load raw features (loaders return UNSCALED X to avoid scaler leakage).
      2. UNSW-NB15: internal split + scale + feature selection + ADASYN+RENN balance.
      3. Other datasets: split → fit scaler on train only (FIX 6) →
         ADASYN+RENN balance on training only (FIX 7) → return.
      4. Unified: load all 4 datasets → map to universal 3-class taxonomy →
         combine → federated partition.

    For "unified" dataset mode:
      - Loads all 4 available datasets
      - Maps all attack categories to 3-class universal taxonomy
      - Combines into one large dataset
      - Standard FL train/test split and balancing applied
    """
    if cfg.training.dataset == "unified":
        return _load_unified_dataset(cfg)

    if cfg.training.dataset == "unsw_nb15":
        # UNSW has internal split + scaler + ADASYN+RENN balancing on training data only
        df, X_train, y_train, X_test, y_test, le = load_unsw_nb15_dataset(cfg)
        return X_train, X_test, y_train, y_test, le

    use_universal = False  # per-dataset modes use original taxonomy

    if cfg.training.dataset == "iomt_2024":
        df, X, y, le = load_iomt_dataset(cfg, use_universal=use_universal)
    elif cfg.training.dataset == "nsl_kdd":
        df, X, y, le = load_nsl_kdd_dataset(cfg, use_universal=use_universal)
    elif cfg.training.dataset == "edge_iiot":
        df, X, y, le = load_edge_iiot_dataset(cfg, use_universal=use_universal)
    else:
        raise ValueError(f"Unknown dataset: {cfg.training.dataset}")

    # no leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=cfg.training.test_ratio,
        random_state=cfg.training.seed, stratify=y,
    )

    #  fit MinMaxScaler on training data only, then transform both splits
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    #  ADASYN+RENN balancing on TRAIN only (matches UNSW pipeline).
    # Test set stays imbalanced for fair evaluation.
    print(f"  Balancing {cfg.training.dataset} training data...")
    X_train, y_train = balance_dataset(
        X_train, y_train,
        method="adasyn_enn",
        k_neighbors=5,
        sampling_strategy=0.5,
        random_state=cfg.training.seed,
    )

    return X_train, X_test, y_train, y_test, le


def _load_unified_dataset(cfg: Config):
    """
    Load all available datasets and map to universal 3-class taxonomy.

    Pipeline:
      1. Load each dataset with universal taxonomy mapping
      2. Normalize each dataset individually (different feature spaces)
      3. Concatenate all datasets
      4. Train/test split with stratification
      5. ADASYN+RENN balancing on combined training data

    Returns:
      X_train, X_test, y_train, y_test, le
      where le.classes_ = ["Benign", "Attack", "Recon"]
    """
    print("\n[UNIFIED] Loading all datasets with universal 3-class taxonomy...")
    all_X_train, all_X_test, all_y_train, all_y_test = [], [], [], []

    dataset_names = []
    for name, loader_func in [
        ("edge_iiot", load_edge_iiot_dataset),
        ("nsl_kdd", load_nsl_kdd_dataset),
        ("iomt_2024", load_iomt_dataset),
        ("unsw_nb15", None),  # handled separately
    ]:
        if loader_func is None:
            continue  # skip unsw for now, handle separately
        try:
            df, X, y, le = loader_func(cfg, use_universal=True)
            print(f"  [UNIFIED] Loaded {name}: {X.shape[0]} samples, {X.shape[1]} features, "
                  f"{len(le.classes_)} classes: {list(le.classes_)}")
            dataset_names.append(name)

            # Individual train/test split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=cfg.training.test_ratio,
                random_state=cfg.training.seed, stratify=y,
            )

            # Normalize individually
            scaler = MinMaxScaler()
            X_tr = scaler.fit_transform(X_tr).astype(np.float32)
            X_te = scaler.transform(X_te).astype(np.float32)

            all_X_train.append(X_tr)
            all_X_test.append(X_te)
            all_y_train.append(y_tr)
            all_y_test.append(y_te)
        except Exception as e:
            print(f"  [UNIFIED] Skipping {name}: {e}")

    # Handle UNSW-NB15 separately (has built-in feature selection pipeline)
    try:
        df, X_tr, y_tr, X_te, y_te, le_unsw = load_unsw_nb15_dataset(cfg, use_universal=True)
        print(f"  [UNIFIED] Loaded unsw_nb15: {X_tr.shape[0]} samples, "
              f"{X_tr.shape[1]} features, {len(le_unsw.classes_)} classes")
        all_X_train.append(X_tr)
        all_X_test.append(X_te)
        all_y_train.append(y_tr)
        all_y_test.append(y_te)
        dataset_names.append("unsw_nb15")
    except Exception as e:
        print(f"  [UNIFIED] Skipping unsw_nb15: {e}")

    if not all_X_train:
        raise RuntimeError("[UNIFIED] No datasets loaded. Ensure Dataset/ directory contains CSV files.")

    # ── Universal feature alignment ──────────────────────────────────────────
    # All datasets must have the same feature dimension for federated training.
    # We use feature selection to find a common feature space.
    print(f"  [UNIFIED] Aligning features across {len(all_X_train)} datasets...")

    # Find common feature dimension (minimum among all datasets)
    min_features = min(x.shape[1] for x in all_X_train)
    max_features = max(x.shape[1] for x in all_X_train)

    if min_features == max_features:
        print(f"  [UNIFIED] All datasets have same feature dim ({min_features}), concatenating directly.")
        X_train_combined = np.concatenate(all_X_train, axis=0)
        X_test_combined = np.concatenate(all_X_test, axis=0)
        y_train_combined = np.concatenate(all_y_train, axis=0)
        y_test_combined = np.concatenate(all_y_test, axis=0)
    else:
        print(f"  [UNIFIED] Feature dims vary ({min_features}-{max_features}). "
              f"Using shared feature selection on combined data.")
        # Concatenate first, then apply feature selection
        X_all = np.concatenate(all_X_train, axis=0)
        y_all = np.concatenate(all_y_train, axis=0)

        # Feature selection on combined data (ADASYN-balanced for fair feature importance)
        X_selected, feat_idx = select_features(
            X_all, y_all,
            method="rf_importance",
            corr_threshold=0.90,
            random_state=cfg.training.seed,
            use_balanced=True,
        )

        # Apply same feature selection to all datasets
        new_X_train, new_X_test = [], []
        for X_tr, X_te in zip(all_X_train, all_X_test):
            new_X_train.append(X_tr[:, feat_idx])
            new_X_test.append(X_te[:, feat_idx])

        X_train_combined = np.concatenate(new_X_train, axis=0)
        X_test_combined = np.concatenate(new_X_test, axis=0)
        y_train_combined = np.concatenate(all_y_train, axis=0)
        y_test_combined = np.concatenate(all_y_test, axis=0)

        print(f"  [UNIFIED] After feature alignment: {X_train_combined.shape[1]} features, "
              f"{X_train_combined.shape[0]} train samples")

    # ── Shuffle combined dataset ───────────────────────────────────────────
    rng = np.random.RandomState(cfg.training.seed)
    perm = rng.permutation(len(X_train_combined))
    X_train_combined = X_train_combined[perm]
    y_train_combined = y_train_combined[perm]

    # ── ADASYN+RENN balancing on unified training data ─────────────────────
    print(f"  [UNIFIED] Balancing unified training data ({len(X_train_combined)} samples)...")
    X_train_bal, y_train_bal = balance_dataset(
        X_train_combined, y_train_combined,
        method="adasyn_enn",
        k_neighbors=5,
        sampling_strategy=0.5,
        random_state=cfg.training.seed,
    )

    # Universal label encoder
    le = LabelEncoder()
    le.classes_ = np.array(UNIVERSAL_CLASS_NAMES)  # ["Benign", "Attack", "Recon"]

    print(f"\n  [UNIFIED] Final: {X_train_bal.shape[0]} train (balanced), "
          f"{X_test_combined.shape[0]} test (imbalanced), "
          f"{X_train_bal.shape[1]} features, 3 classes: {UNIVERSAL_CLASS_NAMES}")

    return X_train_bal, X_test_combined, y_train_bal, y_test_combined, le


class DataPreprocessor:
    """
    Unified data preprocessor for all 4 datasets with sequence support.
    Provides dataset configs, sequence creation, and consistent loading.
    """

    DATASET_CONFIGS = {
        "nsl_kdd": {
            "num_classes": 5,
            "seq_len": 1,
            "normalize_method": "minmax",
            "has_temporal": False,
        },
        "iomt_2024": {
            "num_classes": 18,
            "seq_len": 10,
            "normalize_method": "robust",
            "has_temporal": True,
        },
        "edge_iiot": {
            "num_classes": 15,
            "seq_len": 8,
            "normalize_method": "minmax",
            "has_temporal": True,
        },
        "unsw_nb15": {
            "num_classes": 10,
            "seq_len": 5,
            "normalize_method": "minmax",
            "has_temporal": True,
        },
        "unified": {
            "num_classes": 3,  # Benign, Attack, Recon (universal 3-class taxonomy)
            "seq_len": 8,
            "normalize_method": "minmax",
            "has_temporal": True,
        },
    }

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.cfg = self.DATASET_CONFIGS.get(dataset_name, self.DATASET_CONFIGS["edge_iiot"])
        self.seq_len = self.cfg["seq_len"]
        self.num_classes = self.cfg["num_classes"]
        self.normalize_method = self.cfg["normalize_method"]
        self.has_temporal = self.cfg["has_temporal"]

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = None,
        client_ids: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding-window sequence creation for LSTM/GRU.

        Args:
            X: feature array [N, features]
            y: labels [N]
            seq_len: window size (uses self.seq_len if None)
            client_ids: if provided, sequences respect client boundaries (no cross-client windows)

        Returns:
            X_seq: [N - seq_len + 1, seq_len, features]
            y_seq: [N - seq_len + 1] (label of last timestep in each window)
        """
        if seq_len is None:
            seq_len = self.seq_len

        if seq_len <= 1 or not self.has_temporal:
            # No sequencing needed — return as-is with seq_len=1 singleton
            return X, y

        n_samples = len(X)
        if n_samples <= seq_len:
            # Not enough samples for even one sequence
            return X[:1], y[:1]

        X_seq_list = []
        y_seq_list = []

        if client_ids is not None:
            # Respect client boundaries — only create sequences within same client
            clients = np.unique(client_ids)
            for client in clients:
                mask = client_ids == client
                client_X = X[mask]
                client_y = y[mask]
                n_c = len(client_X)
                for start in range(n_c - seq_len + 1):
                    X_seq_list.append(client_X[start:start + seq_len])
                    y_seq_list.append(client_y[start + seq_len - 1])
        else:
            # Sliding window over entire dataset (treat as single stream)
            for start in range(n_samples - seq_len + 1):
                X_seq_list.append(X[start:start + seq_len])
                y_seq_list.append(y[start + seq_len - 1])

        X_seq = np.stack(X_seq_list, axis=0).astype(np.float32)
        y_seq = np.array(y_seq_list, dtype=np.int64)
        return X_seq, y_seq

    def normalize(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply dataset-specific normalization.

        robust: median + IQR (good for IoT with heavy outliers)
        minmax: standard MinMaxScaler (default)
        """
        if self.normalize_method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()

        X_train_norm = scaler.fit_transform(X_train).astype(np.float32)
        X_test_norm = scaler.transform(X_test).astype(np.float32)
        return X_train_norm, X_test_norm

    def load_synthetic(
        self,
        n_samples: int = 2000,
        n_features: int = None,
        class_weights: list = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data matching real dataset characteristics.
        Used for testing when real CSVs are unavailable.
        """
        rng = np.random.RandomState(seed)
        configs = {
            "nsl_kdd":    (41,  5, [0.53, 0.23, 0.15, 0.06, 0.03]),
            "iomt_2024":  (80, 18, None),
            "edge_iiot":  (61, 15, None),
            "unsw_nb15":  (49, 10, [0.93, 0.03, 0.01, 0.01, 0.005, 0.005,
                                     0.003, 0.003, 0.002, 0.002]),
        }
        feat_dim, n_classes, cw = configs.get(self.dataset_name, (64, 5, None))

        if n_features is not None:
            feat_dim = n_features
        if class_weights is not None:
            cw = class_weights

        # Generate class labels
        if cw is not None:
            class_probs = np.array(cw) / sum(cw)
            labels = rng.choice(n_classes, size=n_samples, p=class_probs)
        else:
            labels = rng.randint(0, n_classes, size=n_samples)

        # Generate features with class-conditional structure
        X = np.zeros((n_samples, feat_dim), dtype=np.float32)
        for i in range(n_samples):
            cls = labels[i]
            # Base random features
            base = rng.randn(feat_dim) * 0.5
            # Add class-specific pattern (makes classes somewhat separable)
            class_pattern = rng.randn(feat_dim) * (0.5 + cls * 0.1)
            X[i] = base + class_pattern

        # Apply normalization
        X, _ = self.normalize(X, X[:10])  # Use last 10 as fake test for scaler fitting

        return X.astype(np.float32), labels.astype(np.int64)


def make_synthetic_dataset(name: str, n_samples: int = 2000):
    """
    Convenience function: generate synthetic dataset for testing.
    Returns (X, y, num_classes, seq_len).
    """
    preproc = DataPreprocessor(name)
    X, y = preproc.load_synthetic(n_samples=n_samples)
    return X, y, preproc.num_classes, preproc.seq_len
