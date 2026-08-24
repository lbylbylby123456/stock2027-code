import os

# =========================================================
# Thread settings for numerical backend stability.
# Must be set before importing sklearn.
# =========================================================
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import copy
import math
import random
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F


warnings.filterwarnings("ignore")


# =========================================================
# 0. Basic Configuration
# =========================================================

SEED = 42

# Options: "csi300", "sp500"
DATASET_NAME = "csi300"

if DATASET_NAME == "csi300":
    TRAIN_PATH = "processed/csi300_alpha360_like_train.pkl"
    VALID_PATH = "processed/csi300_alpha360_like_valid.pkl"
    TEST_PATH = "processed/csi300_alpha360_like_test.pkl"
    GROUP_SIZE = 300
elif DATASET_NAME == "sp500":
    TRAIN_PATH = "processed/sp500_alpha360_like_train.pkl"
    VALID_PATH = "processed/sp500_alpha360_like_valid.pkl"
    TEST_PATH = "processed/sp500_alpha360_like_test.pkl"
    GROUP_SIZE = 500
else:
    raise ValueError("DATASET_NAME must be one of: csi300, sp500")

LABEL_COL = "label"
SEQ_LEN = 60

# =========================================================
# ReCaT hyperparameters
# =========================================================

D_MODEL = 64
N_HEADS = 4
TRANS_LAYERS = 2
D_FF = 128
DROPOUT = 0.20

RELATION_HIDDEN = 64
CANDIDATE_LAGS = 5
NUM_REGIMES = 3
REGIME_EMB_DIM = 32

LR = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA = DEVICE.type == "cuda"

# Use a smaller setting when CUDA is unavailable.
MAX_EPOCHS = 40 if IS_CUDA else 18
PATIENCE = 8 if IS_CUDA else 5

POINT_WEIGHT = 1.00
RANK_WEIGHT = 0.20
EDGE_UTIL_WEIGHT = 0.05
ENTROPY_WEIGHT = 0.001
SMOOTH_WEIGHT = 0.001

RANK_PAIR_BATCH = 2048 if IS_CUDA else 1024
RANK_TAU = 0.10

UTIL_TOPK = 3
UTIL_MARGIN = 1e-4
UTIL_EVERY = 3 if IS_CUDA else 8          # CPU computes the edge-utility term less frequently.
UTIL_TARGET_SAMPLE = 64 if IS_CUDA else 16

# Computational controls
DAY_SAMPLE_RATIO = 1.00 if IS_CUDA else 0.50
VALID_EVERY = 2
MAX_VALID_DAYS = 240 if IS_CUDA else 120
MAX_TRAIN_STOCKS_PER_DAY = 500 if IS_CUDA else 300
MAX_VALID_STOCKS_PER_DAY = 500 if IS_CUDA else 300
MAX_TEST_STOCKS_PER_DAY = None          # Use all stocks for test evaluation.

RELATION_CHUNK_SIZE = 128 if IS_CUDA else 64
USE_AMP = IS_CUDA
KEEP_FULL_TENSOR_ON_GPU = False         # Keep tensors on CPU by default to reduce peak GPU memory.

# Portfolio evaluation
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0

TOP_PCT = 0.10
BOTTOM_PCT = 0.10
TOP_N = None
BOTTOM_N = None

COST_RATE = 0.001
LABEL_CLIP = 0.20
TRAIN_LABEL_CLIP = 0.20
BACKTEST_MODE = "long_short"

# Optional day-wise target normalization. Disabled by default to match the paper objective.
DAYWISE_LABEL_NORM = False

# Early stopping is based on validation RankIC rather than pointwise loss.
EARLY_STOP_METRIC = "rankic"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# =========================================================
# 1. Data Utilities
# =========================================================

DATE_CANDIDATES = [
    "date", "datetime", "time", "trade_date",
    "timestamp", "day", "dt", "level_0"
]

ID_CANDIDATES = [
    "instrument", "stock", "stock_id", "ticker",
    "code", "symbol", "order_book_id", "level_1"
]

ALWAYS_EXCLUDE = {"index", "__index_level_0__", "level_0", "level_1"}


def load_dataframe(path):
    df = pd.read_pickle(path)

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{path} must contain a pandas DataFrame.")

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name is not None and df.index.name not in df.columns:
        df = df.reset_index()

    if LABEL_COL not in df.columns:
        raise ValueError(f"Column `{LABEL_COL}` is not found in {path}.")

    return df.copy()


def find_first_existing_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def sort_by_date_and_id(df, date_col=None, id_col=None):
    sort_cols = []
    if date_col is not None and date_col in df.columns:
        sort_cols.append(date_col)
    if id_col is not None and id_col in df.columns:
        sort_cols.append(id_col)

    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)
    return df.reset_index(drop=True)


def sort_alpha360_features(feature_cols):
    field_order = {
        "open": 0,
        "high": 1,
        "low": 2,
        "close": 3,
        "adjclose": 4,
        "adj_close": 4,
        "volume": 5,
        "amount": 6,
        "turnover": 7,
    }

    def parse_col(c):
        c_low = str(c).lower()
        if "_lag" not in c_low:
            return (10**9, 10**9, c_low)

        field, lag_part = c_low.split("_lag", 1)
        try:
            lag = int(lag_part)
        except Exception:
            lag = 10**9

        field_rank = field_order.get(field, 999)
        return (lag, field_rank, c_low)

    if any("_lag" in str(c).lower() for c in feature_cols):
        return sorted(feature_cols, key=parse_col)

    return feature_cols


def infer_feature_cols(train_df):
    exclude_cols = {LABEL_COL}
    exclude_cols.update([c for c in DATE_CANDIDATES if c in train_df.columns])
    exclude_cols.update([c for c in ID_CANDIDATES if c in train_df.columns])
    exclude_cols.update([c for c in ALWAYS_EXCLUDE if c in train_df.columns])

    feature_cols = [
        col for col in train_df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col])
    ]
    feature_cols = sort_alpha360_features(feature_cols)

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found.")

    return feature_cols


def extract_xy_groups(df, feature_cols, date_col=None, group_size=300):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    y = pd.to_numeric(df[LABEL_COL], errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if date_col is not None and date_col in df.columns:
        group_ids, _ = pd.factorize(df[date_col].values, sort=True)
    else:
        group_ids = np.arange(len(df)) // group_size

    return X, y, group_ids


def build_day_indices(group_ids, min_size=3):
    day_indices = []
    for g in np.unique(group_ids):
        idx = np.where(group_ids == g)[0]
        if len(idx) >= min_size:
            day_indices.append(idx)
    return day_indices


def reshape_alpha_sequence_np(X, seq_len=60):
    n, f = X.shape
    if f % seq_len == 0:
        input_dim = f // seq_len
        X_seq = X.reshape(n, seq_len, input_dim)
    else:
        input_dim = f
        X_seq = X.reshape(n, 1, input_dim)
    return X_seq.astype(np.float32), input_dim


def flat_tensor_to_seq_tensor(X_flat_t, seq_len=60):
    n, f = X_flat_t.shape
    if f % seq_len == 0:
        input_dim = f // seq_len
        X_seq_t = X_flat_t.view(n, seq_len, input_dim)
    else:
        input_dim = f
        X_seq_t = X_flat_t.view(n, 1, input_dim)
    return X_seq_t, input_dim


def load_and_preprocess():
    train_df = load_dataframe(TRAIN_PATH)
    valid_df = load_dataframe(VALID_PATH)
    test_df = load_dataframe(TEST_PATH)

    train_date_col = find_first_existing_col(train_df, DATE_CANDIDATES)
    valid_date_col = find_first_existing_col(valid_df, DATE_CANDIDATES)
    test_date_col = find_first_existing_col(test_df, DATE_CANDIDATES)

    train_id_col = find_first_existing_col(train_df, ID_CANDIDATES)
    valid_id_col = find_first_existing_col(valid_df, ID_CANDIDATES)
    test_id_col = find_first_existing_col(test_df, ID_CANDIDATES)

    train_df = sort_by_date_and_id(train_df, train_date_col, train_id_col)
    valid_df = sort_by_date_and_id(valid_df, valid_date_col, valid_id_col)
    test_df = sort_by_date_and_id(test_df, test_date_col, test_id_col)

    feature_cols = infer_feature_cols(train_df)

    X_train_raw, y_train_raw, train_groups = extract_xy_groups(
        train_df, feature_cols, train_date_col, GROUP_SIZE
    )
    X_valid_raw, y_valid_raw, valid_groups = extract_xy_groups(
        valid_df, feature_cols, valid_date_col, GROUP_SIZE
    )
    X_test_raw, y_test_raw, test_groups = extract_xy_groups(
        test_df, feature_cols, test_date_col, GROUP_SIZE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_valid = scaler.transform(X_valid_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    if TRAIN_LABEL_CLIP is not None:
        y_train_proc = np.clip(y_train_raw, -TRAIN_LABEL_CLIP, TRAIN_LABEL_CLIP)
        y_valid_proc = np.clip(y_valid_raw, -TRAIN_LABEL_CLIP, TRAIN_LABEL_CLIP)
    else:
        y_train_proc = y_train_raw.copy()
        y_valid_proc = y_valid_raw.copy()

    y_mean = float(np.mean(y_train_proc))
    y_std = float(np.std(y_train_proc) + 1e-8)

    y_train = ((y_train_proc - y_mean) / y_std).astype(np.float32)
    y_valid = ((y_valid_proc - y_mean) / y_std).astype(np.float32)

    X_train_seq, input_dim = reshape_alpha_sequence_np(X_train, SEQ_LEN)
    X_valid_seq, _ = reshape_alpha_sequence_np(X_valid, SEQ_LEN)
    X_test_seq, _ = reshape_alpha_sequence_np(X_test, SEQ_LEN)

    actual_seq_len = X_train_seq.shape[1]

    train_day_indices = build_day_indices(train_groups)
    valid_day_indices = build_day_indices(valid_groups)
    test_day_indices = build_day_indices(test_groups)

    print("========== Dataset Info ==========")
    print(f"Dataset           : {DATASET_NAME}")
    print(f"Device            : {DEVICE}")
    print(f"Train shape       : {train_df.shape}")
    print(f"Valid shape       : {valid_df.shape}")
    print(f"Test shape        : {test_df.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Input shape       : seq_len={actual_seq_len}, input_dim={input_dim}, candidate_lags={min(CANDIDATE_LAGS, actual_seq_len)}")
    print(f"Train days        : {len(train_day_indices)}")
    print(f"Valid days        : {len(valid_day_indices)}")
    print(f"Test days         : {len(test_day_indices)}")
    print(f"Test date column  : {test_date_col}")
    print(f"Test id column    : {test_id_col}")

    return {
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "X_train_seq": X_train_seq,
        "X_valid_seq": X_valid_seq,
        "X_test_seq": X_test_seq,
        "input_dim": input_dim,
        "actual_seq_len": actual_seq_len,
        "y_train_raw": y_train_raw,
        "y_valid_raw": y_valid_raw,
        "y_test_raw": y_test_raw,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_mean": y_mean,
        "y_std": y_std,
        "train_groups": train_groups,
        "valid_groups": valid_groups,
        "test_groups": test_groups,
        "train_day_indices": train_day_indices,
        "valid_day_indices": valid_day_indices,
        "test_day_indices": test_day_indices,
        "test_date_col": test_date_col,
        "test_id_col": test_id_col,
    }


# =========================================================
# 2. Market Regime Estimation without sklearn KMeans
# =========================================================

def build_market_indicators(X, group_ids):
    indicators = []

    for g in np.unique(group_ids):
        idx = np.where(group_ids == g)[0]
        X_day = X[idx]

        cs_mean = X_day.mean(axis=0)
        cs_std = X_day.std(axis=0)

        indicator = np.array([
            float(np.mean(cs_mean)),
            float(np.std(cs_mean)),
            float(np.mean(cs_std)),
            float(np.std(cs_std)),
            float(np.mean(np.abs(cs_mean))),
            float(np.mean(np.abs(cs_std))),
        ], dtype=np.float32)

        indicators.append(indicator)

    return np.vstack(indicators).astype(np.float32)


class SimpleGaussianHMM:
    """
    Lightweight diagonal-covariance Gaussian HMM.
    No sklearn KMeans is used; initialization is implemented locally.
    """

    def __init__(self, n_states=3, n_iter=20, min_covar=1e-4, random_state=42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.min_covar = min_covar
        self.random_state = random_state

        self.pi_ = None
        self.trans_ = None
        self.means_ = None
        self.vars_ = None

    def _init_by_sorted_quantiles(self, X):
        T, D = X.shape
        rng = np.random.default_rng(self.random_state)

        score = X.mean(axis=1)
        order = np.argsort(score)
        labels = np.zeros(T, dtype=np.int64)

        splits = np.array_split(order, self.n_states)
        for k, idx_k in enumerate(splits):
            labels[idx_k] = k

        self.pi_ = np.ones(self.n_states, dtype=np.float64) / self.n_states
        self.trans_ = np.ones((self.n_states, self.n_states), dtype=np.float64)

        for t in range(T - 1):
            self.trans_[labels[t], labels[t + 1]] += 1.0

        self.trans_ = self.trans_ / self.trans_.sum(axis=1, keepdims=True)

        self.means_ = np.zeros((self.n_states, D), dtype=np.float64)
        self.vars_ = np.zeros((self.n_states, D), dtype=np.float64)

        for k in range(self.n_states):
            Xk = X[labels == k]
            if len(Xk) == 0:
                sample_size = max(1, T // self.n_states)
                Xk = X[rng.choice(T, size=sample_size, replace=False)]
            self.means_[k] = Xk.mean(axis=0)
            self.vars_[k] = Xk.var(axis=0) + self.min_covar

    def _log_gaussian_prob(self, X):
        T, D = X.shape
        log_prob = np.zeros((T, self.n_states), dtype=np.float64)

        for k in range(self.n_states):
            mean = self.means_[k]
            var = self.vars_[k]
            log_det = np.sum(np.log(var))
            quad = np.sum((X - mean) ** 2 / var, axis=1)
            log_prob[:, k] = -0.5 * (D * np.log(2 * np.pi) + log_det + quad)

        return log_prob

    def _forward_backward(self, log_emit):
        T, K = log_emit.shape
        log_pi = np.log(self.pi_ + 1e-12)
        log_trans = np.log(self.trans_ + 1e-12)

        log_alpha = np.zeros((T, K), dtype=np.float64)
        log_beta = np.zeros((T, K), dtype=np.float64)

        log_alpha[0] = log_pi + log_emit[0]

        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = log_emit[t, k] + logsumexp(log_alpha[t - 1] + log_trans[:, k])

        log_beta[-1] = 0.0

        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = logsumexp(log_trans[k] + log_emit[t + 1] + log_beta[t + 1])

        log_likelihood = logsumexp(log_alpha[-1])

        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)

        xi_sum = np.zeros((K, K), dtype=np.float64)

        for t in range(T - 1):
            log_xi = (
                log_alpha[t, :, None]
                + log_trans
                + log_emit[t + 1][None, :]
                + log_beta[t + 1][None, :]
                - log_likelihood
            )
            xi_sum += np.exp(log_xi)

        return gamma, xi_sum

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        T, D = X.shape

        if T < self.n_states:
            raise ValueError("Number of time steps must be larger than n_states.")

        self._init_by_sorted_quantiles(X)

        for _ in range(self.n_iter):
            log_emit = self._log_gaussian_prob(X)
            gamma, xi_sum = self._forward_backward(log_emit)

            self.pi_ = gamma[0] + 1e-6
            self.pi_ = self.pi_ / self.pi_.sum()

            self.trans_ = xi_sum + 1e-6
            self.trans_ = self.trans_ / self.trans_.sum(axis=1, keepdims=True)

            weights = gamma.sum(axis=0) + 1e-8

            for k in range(self.n_states):
                self.means_[k] = (gamma[:, k][:, None] * X).sum(axis=0) / weights[k]
                diff = X - self.means_[k]
                self.vars_[k] = (gamma[:, k][:, None] * diff * diff).sum(axis=0) / weights[k]
                self.vars_[k] = np.maximum(self.vars_[k], self.min_covar)

        return self

    def filtered_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        log_emit = self._log_gaussian_prob(X)

        T, K = log_emit.shape
        log_trans = np.log(self.trans_ + 1e-12)

        alpha = np.zeros((T, K), dtype=np.float64)
        alpha[0] = np.log(self.pi_ + 1e-12) + log_emit[0]
        alpha[0] -= logsumexp(alpha[0])

        for t in range(1, T):
            for k in range(K):
                alpha[t, k] = log_emit[t, k] + logsumexp(alpha[t - 1] + log_trans[:, k])
            alpha[t] -= logsumexp(alpha[t])

        return np.exp(alpha).astype(np.float32)



# =========================================================
# HMM market-indicator construction
# =========================================================

def build_market_indicators_from_indices(X, day_indices, name="split"):
    """
    Market-indicator construction using precomputed day indices.
    This avoids repeatedly scanning the full dataset.
    """
    indicators = []
    total_days = len(day_indices)

    for day_id, idx in enumerate(day_indices):
        X_day = X[idx]

        cs_mean = X_day.mean(axis=0)
        cs_std = X_day.std(axis=0)

        indicator = np.array([
            float(np.mean(cs_mean)),
            float(np.std(cs_mean)),
            float(np.mean(cs_std)),
            float(np.std(cs_std)),
            float(np.mean(np.abs(cs_mean))),
            float(np.mean(np.abs(cs_std))),
        ], dtype=np.float32)

        indicators.append(indicator)

        if (day_id + 1) % 300 == 0 or (day_id + 1) == total_days:
            print(f"  Built {name} market indicators: {day_id + 1}/{total_days} days")

    return np.vstack(indicators).astype(np.float32)


def compute_regime_posteriors(data):
    """
    HMM is fitted only on train market indicators, and posterior filtering is
    applied through train -> valid -> test in chronological order.
    """
    print("Estimating HMM regime posteriors without sklearn KMeans...")
    print("Building market indicators with precomputed day indices...")

    train_market = build_market_indicators_from_indices(
        data["X_train"],
        data["train_day_indices"],
        name="train"
    )
    valid_market = build_market_indicators_from_indices(
        data["X_valid"],
        data["valid_day_indices"],
        name="valid"
    )
    test_market = build_market_indicators_from_indices(
        data["X_test"],
        data["test_day_indices"],
        name="test"
    )

    print("Scaling market indicators...")
    market_scaler = StandardScaler()
    train_market_s = market_scaler.fit_transform(train_market).astype(np.float32)

    all_market = np.vstack([train_market, valid_market, test_market])
    all_market_s = market_scaler.transform(all_market).astype(np.float32)

    print("Fitting lightweight Gaussian HMM on training market indicators...")
    hmm = SimpleGaussianHMM(
        n_states=NUM_REGIMES,
        n_iter=20,
        random_state=SEED
    )
    hmm.fit(train_market_s)

    print("Filtering regime probabilities through train -> valid -> test...")
    all_probs = hmm.filtered_proba(all_market_s)

    n_train = len(train_market)
    n_valid = len(valid_market)

    train_probs = all_probs[:n_train]
    valid_probs = all_probs[n_train:n_train + n_valid]
    test_probs = all_probs[n_train + n_valid:]

    print("Finished HMM regime posterior estimation.")

    return (
        torch.FloatTensor(train_probs),
        torch.FloatTensor(valid_probs),
        torch.FloatTensor(test_probs),
    )


# =========================================================
# 3. SparseMax
# =========================================================

def sparsemax(logits, dim=-1):
    logits = logits - logits.max(dim=dim, keepdim=True).values
    zs = torch.sort(logits, dim=dim, descending=True).values

    range_vals = torch.arange(
        1, logits.size(dim) + 1,
        device=logits.device,
        dtype=logits.dtype
    )

    view_shape = [1] * logits.dim()
    view_shape[dim] = -1
    range_vals = range_vals.view(view_shape)

    bound = 1 + range_vals * zs
    cumsum_zs = torch.cumsum(zs, dim=dim)

    is_gt = bound > cumsum_zs
    k = is_gt.sum(dim=dim, keepdim=True).clamp(min=1)

    tau = (torch.gather(cumsum_zs, dim, k - 1) - 1) / k.to(logits.dtype)
    output = torch.clamp(logits - tau, min=0.0)

    return output


# =========================================================
# 4. ReCaT Model
# =========================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ReCaT(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        n_heads=4,
        trans_layers=2,
        d_ff=128,
        dropout=0.2,
        relation_hidden=64,
        candidate_lags=5,
        num_regimes=3,
        regime_emb_dim=32,
    ):
        super().__init__()

        self.d_model = d_model
        self.candidate_lags = candidate_lags
        self.num_regimes = num_regimes

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=trans_layers
        )

        self.pool_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        self.relation_scorer = nn.Sequential(
            nn.Linear(4 * d_model, relation_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(relation_hidden, relation_hidden),
            nn.ReLU(),
            nn.Linear(relation_hidden, 1),
        )

        self.lag_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(candidate_lags)
        ])

        self.self_linear = nn.Linear(d_model, d_model)
        self.msg_linear = nn.Linear(d_model, d_model)
        self.relation_norm = nn.LayerNorm(d_model)
        self.relation_dropout = nn.Dropout(dropout)

        self.regime_encoder = nn.Sequential(
            nn.Linear(num_regimes, regime_emb_dim),
            nn.ReLU(),
            nn.Linear(regime_emb_dim, regime_emb_dim),
            nn.ReLU(),
        )

        self.gamma_net = nn.Linear(d_model + regime_emb_dim, d_model)
        self.beta_net = nn.Linear(d_model + regime_emb_dim, d_model)
        self.final_norm = nn.LayerNorm(d_model)

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def temporal_encode(self, x_seq):
        h = self.input_proj(x_seq)
        h = self.pos_enc(h)
        h = self.temporal_encoder(h)

        pool_logits = self.pool_score(h).squeeze(-1)
        pool_weights = torch.softmax(pool_logits, dim=1)
        e = torch.sum(h * pool_weights.unsqueeze(-1), dim=1)

        return e, h

    def discover_relation_graph(self, E, H):
        """
        A[i, j, ell] is the directed contribution from source stock j at lag ell
        to target stock i.
        """
        n, d = E.shape
        lags = min(self.candidate_lags, H.shape[1])

        lag_states = [H[:, -(ell + 1), :] for ell in range(lags)]
        Z = torch.stack(lag_states, dim=0)  # [L, N, D]

        eye = torch.eye(n, device=E.device).bool()
        all_lag_scores = []

        for ell in range(lags):
            source = Z[ell]  # [N, D]
            chunks = []

            for start in range(0, n, RELATION_CHUNK_SIZE):
                end = min(start + RELATION_CHUNK_SIZE, n)

                target_chunk = E[start:end]  # [C, D]
                c = target_chunk.shape[0]

                target_expand = target_chunk.unsqueeze(1).expand(c, n, d)
                source_expand = source.unsqueeze(0).expand(c, n, d)

                pair_feat = torch.cat(
                    [
                        target_expand,
                        source_expand,
                        target_expand - source_expand,
                        target_expand * source_expand,
                    ],
                    dim=-1,
                )

                score_chunk = self.relation_scorer(pair_feat).squeeze(-1)  # [C, N]
                chunks.append(score_chunk)

            score = torch.cat(chunks, dim=0)  # [N, N]
            score = score.masked_fill(eye, -1e9)
            all_lag_scores.append(score)

        scores = torch.stack(all_lag_scores, dim=-1)  # [N, N, L]

        flat_scores = scores.reshape(n, -1)
        A_flat = sparsemax(flat_scores, dim=1)
        A = A_flat.reshape(n, n, lags)

        A = A.masked_fill(eye.unsqueeze(-1), 0.0)
        A = A / (A.sum(dim=(1, 2), keepdim=True) + 1e-8)

        return A, Z

    def directed_message_passing(self, E, A, Z):
        lag_msgs = []

        for ell in range(A.shape[-1]):
            source_lag = Z[ell]
            transformed = self.lag_transforms[ell](source_lag)
            msg_ell = torch.matmul(A[:, :, ell], transformed)
            lag_msgs.append(msg_ell)

        relation_msg = torch.stack(lag_msgs, dim=0).sum(dim=0)

        update = F.relu(self.self_linear(E) + self.msg_linear(relation_msg))
        update = self.relation_dropout(update)
        h_relation = self.relation_norm(E + update)

        return h_relation

    def regime_modulation(self, h_relation, regime_prob):
        n = h_relation.shape[0]

        regime_emb = self.regime_encoder(regime_prob)
        regime_expand = regime_emb.unsqueeze(0).expand(n, -1)

        mod_input = torch.cat([h_relation, regime_expand], dim=-1)
        gamma = torch.tanh(self.gamma_net(mod_input))
        beta = self.beta_net(mod_input)

        h_mod = self.final_norm((1.0 + gamma) * h_relation + beta)
        return h_mod

    def forward_with_A(self, x_seq, regime_prob, A_override=None):
        E, H = self.temporal_encode(x_seq)

        if A_override is None:
            A, Z = self.discover_relation_graph(E, H)
        else:
            A = A_override
            lags = A.shape[-1]
            Z = torch.stack([H[:, -(ell + 1), :] for ell in range(lags)], dim=0)

        h_relation = self.directed_message_passing(E, A, Z)
        h_mod = self.regime_modulation(h_relation, regime_prob)

        pred = self.pred_head(h_mod).squeeze(-1)
        return pred, A

    def forward(self, x_seq, regime_prob, return_aux=False):
        pred, A = self.forward_with_A(x_seq, regime_prob, A_override=None)
        if return_aux:
            return pred, A
        return pred


# =========================================================
# 5. Losses
# =========================================================

def weighted_pairwise_rank_loss(pred, y, pair_batch=1024, tau=0.10):
    n = pred.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=pred.device)

    b = min(pair_batch, max(2, n * 4))
    idx_i = torch.randint(0, n, (b,), device=pred.device)
    idx_j = torch.randint(0, n, (b,), device=pred.device)

    y_diff = y[idx_i] - y[idx_j]
    pred_diff = pred[idx_i] - pred[idx_j]

    mask = torch.abs(y_diff) > 1e-8
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    y_diff = y_diff[mask]
    pred_diff = pred_diff[mask]

    sign = torch.sign(y_diff)
    weights = torch.abs(y_diff)
    weights = weights / (weights.sum() + 1e-8)

    loss = weights * F.softplus(-sign * pred_diff / tau)
    return loss.sum()


def relation_entropy_loss(A):
    eps = 1e-8
    ent = -torch.sum(A * torch.log(A + eps), dim=(1, 2))
    return ent.mean()


def graph_smoothness_loss(A, A_prev, is_consecutive=True):
    if A_prev is None or not is_consecutive:
        return torch.tensor(0.0, device=A.device)

    if A.shape != A_prev.shape:
        return torch.tensor(0.0, device=A.device)

    A_bar = A.sum(dim=-1)
    A_prev_bar = A_prev.sum(dim=-1)

    return torch.mean((A_bar - A_prev_bar) ** 2)


def predictive_edge_utility_loss(
    model,
    x_seq,
    regime_prob,
    y,
    pred_full,
    A,
    topk=3,
    margin=1e-4,
    target_sample=32,
):
    """
    Top-k predictive edge-utility masking.
    For speed, it samples target stocks.
    """
    n = A.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=A.device)

    with torch.no_grad():
        flat = A.reshape(n, -1)
        k = min(topk, flat.shape[1])
        if k <= 0:
            return torch.tensor(0.0, device=A.device)

        if target_sample is not None and target_sample < n:
            target_ids = torch.randperm(n, device=A.device)[:target_sample]
        else:
            target_ids = torch.arange(n, device=A.device)

        top_idx = torch.topk(flat[target_ids], k=k, dim=1).indices
        base_loss_each = F.smooth_l1_loss(pred_full, y, reduction="none").detach()

    losses = []

    n_source = A.shape[1]
    n_lags = A.shape[2]

    for local_pos, i in enumerate(target_ids):
        i_int = int(i.item())

        for flat_edge in top_idx[local_pos]:
            flat_edge_int = int(flat_edge.item())
            j = flat_edge_int // n_lags
            ell = flat_edge_int % n_lags

            edge_weight = A[i_int, j, ell].detach()
            if edge_weight.item() <= 1e-12:
                continue

            A_drop = A.clone()
            A_drop[i_int, j, ell] = 0.0
            A_drop[i_int] = A_drop[i_int] / (A_drop[i_int].sum() + 1e-8)

            pred_drop, _ = model.forward_with_A(
                x_seq=x_seq,
                regime_prob=regime_prob,
                A_override=A_drop
            )

            drop_loss_i = F.smooth_l1_loss(
                pred_drop[i_int],
                y[i_int],
                reduction="none"
            )

            ce_i = drop_loss_i - base_loss_each[i_int]
            losses.append(edge_weight * F.relu(margin - ce_i))

    if len(losses) == 0:
        return torch.tensor(0.0, device=A.device)

    return torch.stack(losses).mean()


# =========================================================
# 6. Tensor Utilities
# =========================================================

def to_tensor_maybe_gpu(arr, keep_on_gpu=True):
    t = torch.from_numpy(arr).float()
    if keep_on_gpu and DEVICE.type == "cuda":
        t = t.to(DEVICE)
    return t


def maybe_subsample_idx(idx, max_stocks, seed_offset=0):
    if max_stocks is None or len(idx) <= max_stocks:
        return idx

    rng = np.random.default_rng(SEED + seed_offset)
    selected = rng.choice(idx, size=max_stocks, replace=False)
    return np.sort(selected)


def get_day_tensors(X_seq_t, y_t, idx):
    if X_seq_t.device.type == "cuda":
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=DEVICE)
        x_day = X_seq_t[idx_t]
        y_day = y_t[idx_t]
    else:
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        x_day = X_seq_t[idx_t].to(DEVICE, non_blocking=True)
        y_day = y_t[idx_t].to(DEVICE, non_blocking=True)
    return x_day, y_day


def get_regime_for_day(regime_t, day_id):
    if regime_t.device.type == DEVICE.type:
        return regime_t[day_id].to(DEVICE)
    return regime_t[day_id].to(DEVICE)


def save_best_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# =========================================================
# 7. Training and Prediction
# =========================================================

def predict_model(model, data, test_regime):
    model.eval()

    X_test_seq_t = to_tensor_maybe_gpu(data["X_test_seq"], KEEP_FULL_TENSOR_ON_GPU)
    test_regime_t = test_regime.to(DEVICE) if KEEP_FULL_TENSOR_ON_GPU and IS_CUDA else test_regime

    preds = np.zeros(len(data["X_test_seq"]), dtype=np.float32)
    use_amp = USE_AMP and DEVICE.type == "cuda"

    with torch.no_grad():
        for day_id, idx in enumerate(data["test_day_indices"]):
            idx_used = maybe_subsample_idx(idx, MAX_TEST_STOCKS_PER_DAY, seed_offset=900000 + day_id)

            if len(idx_used) != len(idx):
                # If test subsampling is enabled, unpredicted stocks keep zero scores.
                pass

            if X_test_seq_t.device.type == "cuda":
                idx_t = torch.as_tensor(idx_used, dtype=torch.long, device=DEVICE)
                x_day = X_test_seq_t[idx_t]
            else:
                idx_t = torch.as_tensor(idx_used, dtype=torch.long)
                x_day = X_test_seq_t[idx_t].to(DEVICE, non_blocking=True)

            p_day = get_regime_for_day(test_regime_t, day_id)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_scaled = model(x_day, p_day)

            pred_raw = pred_scaled.detach().cpu().numpy() * data["y_std"] + data["y_mean"]
            preds[idx_used] = pred_raw

    return preds


# =========================================================
# 8. Metrics and Backtest
# =========================================================

def prepare_eval_df(test_df, y_pred, label_col="label", date_col=None, inst_col=None, group_size=300):
    df = test_df.copy().reset_index(drop=True)

    if date_col is None or date_col not in df.columns:
        df["datetime"] = np.arange(len(df)) // group_size
        date_col = "datetime"

    if inst_col is None or inst_col not in df.columns:
        df["instrument"] = df.groupby(date_col).cumcount().astype(str)
        inst_col = "instrument"

    if len(y_pred) != len(df):
        raise ValueError(f"len(y_pred)={len(y_pred)} but len(test_df)={len(df)}")

    eval_df = pd.DataFrame({
        "datetime": df[date_col].values,
        "instrument": df[inst_col].astype(str).values,
        "pred": np.asarray(y_pred, dtype=np.float64),
        "label": pd.to_numeric(df[label_col], errors="coerce").values,
    })

    try:
        eval_df["datetime"] = pd.to_datetime(eval_df["datetime"])
    except Exception:
        pass

    eval_df = eval_df.replace([np.inf, -np.inf], np.nan)
    eval_df = eval_df.dropna(subset=["datetime", "instrument", "pred", "label"])
    eval_df = eval_df.sort_values(["datetime", "instrument"]).reset_index(drop=True)

    return eval_df


def safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 3:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 3:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(spearmanr(x, y).correlation)


def calculate_ic_rankic(eval_df, min_stocks=10):
    ic_list = []
    rankic_list = []

    for _, g in eval_df.groupby("datetime"):
        if len(g) < min_stocks:
            continue

        ic = safe_pearson(g["pred"].values, g["label"].values)
        rankic = safe_spearman(g["pred"].values, g["label"].values)

        if not np.isnan(ic):
            ic_list.append(ic)
        if not np.isnan(rankic):
            rankic_list.append(rankic)

    mean_ic = float(np.nanmean(ic_list)) if len(ic_list) > 0 else np.nan
    mean_rankic = float(np.nanmean(rankic_list)) if len(rankic_list) > 0 else np.nan

    return mean_rankic, mean_ic


def backtest_daily_portfolio(
    eval_df,
    mode="long_short",
    top_n=None,
    bottom_n=None,
    top_pct=0.10,
    bottom_pct=0.10,
    trading_days=252,
    risk_free_rate=0.0,
    cost_rate=0.001,
    label_clip=0.20,
    min_stocks=30,
):
    df = eval_df.copy().sort_values(["datetime", "instrument"])

    daily_returns = []
    gross_returns = []
    costs = []
    turnovers = []

    prev_weights = pd.Series(dtype=np.float64)

    for _, g in df.groupby("datetime"):
        g = g.copy()

        if len(g) < min_stocks:
            continue

        ret = g["label"].astype(float).copy()
        if label_clip is not None:
            ret = ret.clip(-label_clip, label_clip)

        g["ret_used"] = ret
        g = g.sort_values("pred", ascending=True)
        n = len(g)

        if top_n is None:
            k_long = max(1, int(n * top_pct))
        else:
            k_long = min(top_n, max(1, n // 2))

        if bottom_n is None:
            k_short = max(1, int(n * bottom_pct))
        else:
            k_short = min(bottom_n, max(1, n // 2))

        long_part = g.tail(k_long)
        short_part = g.head(k_short)

        weights = pd.Series(0.0, index=g["instrument"].values, dtype=np.float64)

        if mode == "long_only":
            weights.loc[long_part["instrument"].values] = 1.0 / k_long
        elif mode == "long_short":
            weights.loc[long_part["instrument"].values] = 0.5 / k_long
            weights.loc[short_part["instrument"].values] = -0.5 / k_short
        else:
            raise ValueError("mode must be 'long_short' or 'long_only'.")

        ret_map = pd.Series(g["ret_used"].values, index=g["instrument"].values)
        gross_ret = float((weights * ret_map).sum())

        union_index = weights.index.union(prev_weights.index)
        w_now = weights.reindex(union_index).fillna(0.0)
        w_prev = prev_weights.reindex(union_index).fillna(0.0)

        turnover = float(np.abs(w_now - w_prev).sum())
        cost = cost_rate * turnover
        net_ret = gross_ret - cost

        daily_returns.append(net_ret)
        gross_returns.append(gross_ret)
        costs.append(cost)
        turnovers.append(turnover)

        prev_weights = weights.copy()

    daily_returns = np.asarray(daily_returns, dtype=np.float64)

    if len(daily_returns) == 0:
        return {
            "ann_return": np.nan,
            "sharpe": np.nan,
            "maxdd": np.nan,
            "daily_returns": daily_returns,
            "gross_returns": np.asarray(gross_returns),
            "costs": np.asarray(costs),
            "turnovers": np.asarray(turnovers),
            "num_days": 0,
            "avg_daily_return": np.nan,
            "avg_gross_return": np.nan,
            "avg_cost": np.nan,
            "avg_turnover": np.nan,
        }

    daily_returns = np.clip(daily_returns, -0.99, 10.0)

    wealth = np.cumprod(1.0 + daily_returns)
    total_return = wealth[-1] - 1.0
    ann_return = (1.0 + total_return) ** (trading_days / len(daily_returns)) - 1.0

    daily_excess = daily_returns - risk_free_rate / trading_days
    daily_vol = np.std(daily_excess, ddof=1)

    sharpe = np.nan if daily_vol < 1e-12 else (
        np.sqrt(trading_days) * np.mean(daily_excess) / daily_vol
    )

    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    maxdd = float(np.min(drawdown))

    return {
        "ann_return": float(ann_return),
        "sharpe": float(sharpe),
        "maxdd": maxdd,
        "daily_returns": daily_returns,
        "gross_returns": np.asarray(gross_returns, dtype=np.float64),
        "costs": np.asarray(costs, dtype=np.float64),
        "turnovers": np.asarray(turnovers, dtype=np.float64),
        "num_days": len(daily_returns),
        "avg_daily_return": float(np.mean(daily_returns)),
        "avg_gross_return": float(np.mean(gross_returns)),
        "avg_cost": float(np.mean(costs)),
        "avg_turnover": float(np.mean(turnovers)),
    }

# =========================================================
# Ranking-oriented training utilities
# =========================================================

def normalize_day_target(y_day):
    """
    Cross-sectional label normalization within each trading day.
    This makes the training objective focus on relative ranking rather than
    absolute return scale.
    """
    if not DAYWISE_LABEL_NORM:
        return y_day

    std = torch.std(y_day, unbiased=False)
    if std < 1e-8:
        return y_day - torch.mean(y_day)

    return (y_day - torch.mean(y_day)) / (std + 1e-8)


def evaluate_valid_score(model, X_valid_seq_t, y_valid_t, valid_regime, data, epoch=None):
    """
    Validation scorer for early stopping.

    Returns:
        valid_rankic: mean Spearman RankIC on sampled validation days
        valid_loss  : pointwise SmoothL1 loss on day-wise normalized labels
    """
    model.eval()
    losses = []
    rankics = []

    valid_ids = list(range(len(data["valid_day_indices"])))
    if len(valid_ids) > MAX_VALID_DAYS:
        rng = np.random.default_rng(SEED + (epoch or 0))
        valid_ids = rng.choice(valid_ids, size=MAX_VALID_DAYS, replace=False).tolist()
        valid_ids = sorted(valid_ids)

    with torch.no_grad():
        for day_id in valid_ids:
            idx = data["valid_day_indices"][day_id]
            idx = maybe_subsample_idx(
                idx,
                MAX_VALID_STOCKS_PER_DAY,
                seed_offset=100000 + day_id
            )

            x_day, y_day = get_day_tensors(X_valid_seq_t, y_valid_t, idx)
            p_day = get_regime_for_day(valid_regime, day_id)

            pred = model(x_day, p_day)
            y_loss = normalize_day_target(y_day)

            point_loss = F.smooth_l1_loss(pred, y_loss)
            losses.append(point_loss.item())

            pred_np = pred.detach().cpu().numpy()
            y_np = y_day.detach().cpu().numpy()

            if len(pred_np) >= 10 and np.std(pred_np) > 1e-12 and np.std(y_np) > 1e-12:
                corr = spearmanr(pred_np, y_np).correlation
                if not np.isnan(corr):
                    rankics.append(float(corr))

    valid_loss = float(np.mean(losses)) if len(losses) > 0 else np.inf
    valid_rankic = float(np.mean(rankics)) if len(rankics) > 0 else -np.inf

    return valid_rankic, valid_loss


def train_model(data, train_regime, valid_regime):
    """
    ReCaT training:
      1) day-wise label normalization for ranking-oriented learning;
      2) higher rank-loss weight;
      3) validation RankIC early stopping;
      4) validation-based early stopping.
    """
    actual_candidate_lags = min(CANDIDATE_LAGS, data["actual_seq_len"])

    X_train_seq_t = to_tensor_maybe_gpu(data["X_train_seq"], KEEP_FULL_TENSOR_ON_GPU)
    X_valid_seq_t = to_tensor_maybe_gpu(data["X_valid_seq"], KEEP_FULL_TENSOR_ON_GPU)

    y_train_t = to_tensor_maybe_gpu(data["y_train"], KEEP_FULL_TENSOR_ON_GPU)
    y_valid_t = to_tensor_maybe_gpu(data["y_valid"], KEEP_FULL_TENSOR_ON_GPU)

    train_regime_t = train_regime.to(DEVICE) if KEEP_FULL_TENSOR_ON_GPU and IS_CUDA else train_regime
    valid_regime_t = valid_regime.to(DEVICE) if KEEP_FULL_TENSOR_ON_GPU and IS_CUDA else valid_regime

    model = ReCaT(
        input_dim=data["input_dim"],
        d_model=D_MODEL,
        n_heads=N_HEADS,
        trans_layers=TRANS_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        relation_hidden=RELATION_HIDDEN,
        candidate_lags=actual_candidate_lags,
        num_regimes=NUM_REGIMES,
        regime_emb_dim=REGIME_EMB_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, MAX_EPOCHS),
        eta_min=LR * 0.10
    )

    use_amp = USE_AMP and DEVICE.type == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_valid_rankic = -float("inf")
    best_valid_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        train_rank_losses = []
        train_point_losses = []

        day_ids = list(range(len(data["train_day_indices"])))

        if DAY_SAMPLE_RATIO < 1.0:
            rng = np.random.default_rng(SEED + epoch)
            num_days = max(1, int(len(day_ids) * DAY_SAMPLE_RATIO))
            day_ids = rng.choice(day_ids, size=num_days, replace=False).tolist()
            day_ids = sorted(day_ids)

        A_prev = None
        prev_day_id = None

        for day_id in day_ids:
            idx = data["train_day_indices"][day_id]
            idx = maybe_subsample_idx(
                idx,
                MAX_TRAIN_STOCKS_PER_DAY,
                seed_offset=epoch * 100000 + day_id
            )

            x_day, y_day = get_day_tensors(X_train_seq_t, y_train_t, idx)
            p_day = get_regime_for_day(train_regime_t, day_id)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred, A = model(x_day, p_day, return_aux=True)

                y_loss = normalize_day_target(y_day)

                point_loss = F.smooth_l1_loss(pred, y_loss)
                rank_loss = weighted_pairwise_rank_loss(
                    pred,
                    y_loss,
                    pair_batch=RANK_PAIR_BATCH,
                    tau=RANK_TAU
                )

                ent_loss = relation_entropy_loss(A)

                is_consecutive = (prev_day_id is not None and day_id == prev_day_id + 1)
                smooth_loss = graph_smoothness_loss(A, A_prev, is_consecutive=is_consecutive)

                if EDGE_UTIL_WEIGHT > 0 and UTIL_EVERY > 0 and epoch % UTIL_EVERY == 0:
                    edge_util_loss = predictive_edge_utility_loss(
                        model=model,
                        x_seq=x_day,
                        regime_prob=p_day,
                        y=y_loss,
                        pred_full=pred,
                        A=A,
                        topk=UTIL_TOPK,
                        margin=UTIL_MARGIN,
                        target_sample=UTIL_TARGET_SAMPLE,
                    )
                else:
                    edge_util_loss = torch.tensor(0.0, device=DEVICE)

                loss = (
                    POINT_WEIGHT * point_loss
                    + RANK_WEIGHT * rank_loss
                    + EDGE_UTIL_WEIGHT * edge_util_loss
                    + ENTROPY_WEIGHT * ent_loss
                    + SMOOTH_WEIGHT * smooth_loss
                )

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            train_losses.append(loss.item())
            train_point_losses.append(point_loss.item())
            train_rank_losses.append(rank_loss.item())

            A_prev = A.detach()
            prev_day_id = day_id

        scheduler.step()

        do_valid = (epoch == 1 or epoch % VALID_EVERY == 0 or epoch == MAX_EPOCHS)

        if do_valid:
            valid_rankic, valid_loss = evaluate_valid_score(
                model=model,
                X_valid_seq_t=X_valid_seq_t,
                y_valid_t=y_valid_t,
                valid_regime=valid_regime_t,
                data=data,
                epoch=epoch,
            )

            improved = valid_rankic > best_valid_rankic
            if improved:
                best_valid_rankic = valid_rankic
                best_valid_loss = valid_loss
                best_state = save_best_state(model)
                wait = 0
            else:
                wait += 1
        else:
            valid_rankic = best_valid_rankic
            valid_loss = best_valid_loss

        if epoch == 1 or epoch % 5 == 0 or do_valid:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {np.mean(train_losses):.6f} | "
                f"Point: {np.mean(train_point_losses):.6f} | "
                f"Rank: {np.mean(train_rank_losses):.6f} | "
                f"Valid RankIC: {valid_rankic:.6f} | "
                f"Valid Loss: {valid_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Days: {len(day_ids)} | "
                f"EdgeUtil: {'on' if (epoch % UTIL_EVERY == 0 and UTIL_EVERY < 999) else 'off'}"
            )

        if wait >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best valid RankIC: {best_valid_rankic:.6f}, "
                f"Best valid loss: {best_valid_loss:.6f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model



# =========================================================
# 10. Main
# =========================================================

if __name__ == "__main__":
    data = load_and_preprocess()

    if not IS_CUDA:
        print(
            "\n[Warning] CUDA is not available. The script is running on CPU.\n"
            "          CPU mode uses a smaller default setting.\n"
            "          For full experiments, use CUDA-enabled PyTorch when available.\n"
        )

    train_regime, valid_regime, test_regime = compute_regime_posteriors(data)

    print(
        f"ReCaT setting: d_model={D_MODEL}, layers={TRANS_LAYERS}, "
        f"lags={min(CANDIDATE_LAGS, data['actual_seq_len'])}, "
        f"day_ratio={DAY_SAMPLE_RATIO}, max_train_stocks={MAX_TRAIN_STOCKS_PER_DAY}, "
        f"util_every={UTIL_EVERY}, util_targets={UTIL_TARGET_SAMPLE}"
    )
    print(f"Effective AMP: {USE_AMP and DEVICE.type == 'cuda'}")
    print(f"Effective keep tensor on GPU: {KEEP_FULL_TENSOR_ON_GPU and DEVICE.type == 'cuda'}")

    model = train_model(
        data=data,
        train_regime=train_regime,
        valid_regime=valid_regime,
    )

    y_pred = predict_model(
        model=model,
        data=data,
        test_regime=test_regime,
    )

    mse = mean_squared_error(data["y_test_raw"], y_pred)
    r2 = r2_score(data["y_test_raw"], y_pred)

    print("\n========== Basic Prediction Metrics ==========")
    print(f"MSE : {mse:.6f}")
    print(f"R2  : {r2:.6f}")

    eval_df = prepare_eval_df(
        test_df=data["test_df"],
        y_pred=y_pred,
        label_col=LABEL_COL,
        date_col=data["test_date_col"],
        inst_col=data["test_id_col"],
        group_size=GROUP_SIZE,
    )

    rank_ic, ic = calculate_ic_rankic(eval_df)

    bt = backtest_daily_portfolio(
        eval_df=eval_df,
        mode=BACKTEST_MODE,
        top_n=TOP_N,
        bottom_n=BOTTOM_N,
        top_pct=TOP_PCT,
        bottom_pct=BOTTOM_PCT,
        trading_days=TRADING_DAYS,
        risk_free_rate=RISK_FREE_RATE,
        cost_rate=COST_RATE,
        label_clip=LABEL_CLIP,
        min_stocks=30,
    )

    bt_no_cost = backtest_daily_portfolio(
        eval_df=eval_df,
        mode=BACKTEST_MODE,
        top_n=TOP_N,
        bottom_n=BOTTOM_N,
        top_pct=TOP_PCT,
        bottom_pct=BOTTOM_PCT,
        trading_days=TRADING_DAYS,
        risk_free_rate=RISK_FREE_RATE,
        cost_rate=0.0,
        label_clip=LABEL_CLIP,
        min_stocks=30,
    )

    print("\n========== ReCaT Results with Fixed Long-Short Evaluation ==========")
    print(f"Mode              : {BACKTEST_MODE}")
    print(f"Top Pct           : {TOP_PCT}")
    print(f"Bottom Pct        : {BOTTOM_PCT}")
    print(f"Cost Rate         : {COST_RATE}")
    print(f"Label Clip        : {LABEL_CLIP}")
    print(f"Num Days          : {bt['num_days']}")
    print(f"RankIC            : {rank_ic:.6f}")
    print(f"IC                : {ic:.6f}")
    print(f"Ann. Return       : {bt['ann_return']:.6f}")
    print(f"Sharpe            : {bt['sharpe']:.6f}")
    print(f"MaxDD             : {bt['maxdd']:.6f}")
    print(f"Avg Daily Ret     : {bt['avg_daily_return']:.6f}")
    print(f"Avg Gross Ret     : {bt['avg_gross_return']:.6f}")
    print(f"Avg Cost          : {bt['avg_cost']:.6f}")
    print(f"Avg Turnover      : {bt['avg_turnover']:.6f}")

    print("\n========== No-cost Diagnostic ==========")
    print(f"Ann. Return       : {bt_no_cost['ann_return']:.6f}")
    print(f"Sharpe            : {bt_no_cost['sharpe']:.6f}")
    print(f"MaxDD             : {bt_no_cost['maxdd']:.6f}")
    print(f"Avg Daily Ret     : {bt_no_cost['avg_daily_return']:.6f}")
    print(f"Avg Turnover      : {bt_no_cost['avg_turnover']:.6f}")

    print("\nLaTeX table row:")
    print(
        "ReCaT "
        f"& {rank_ic:.3f} "
        f"& {ic:.3f} "
        f"& {bt['ann_return'] * 100:.2f}\\% "
        f"& {bt['sharpe']:.2f} "
        f"& ${bt['maxdd'] * 100:.2f}\\%$ \\\\"
    )
