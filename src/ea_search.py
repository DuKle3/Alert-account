# src/ea_search.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import math

# ---------- 個體定義 ----------
@dataclass
class Individual:
    mask: np.ndarray          # 0/1 特徵選擇
    hi_idx: int               # 大額占比欄位索引
    k_mult: float             # Top-K 倍數
    mode: str                 # "topk" or "threshold"

    # ---- ES 相關的「步長 / 變異率」參數 ----
    sigma_k: float            # 控制 k_mult 的變異步長 (step-size)
    p_flip: float             # 控制 mask 每一維被翻轉的機率

# ---------- 小工具 ----------
def _prepare_xy(df: pd.DataFrame, feature_names: List[str]):
    X = df[feature_names].values
    y = df["label"].values
    return X, y

def _fit_xgb(X_tr, y_tr, X_va, y_va, spw: float, seed: int = 42):
    model = XGBClassifier(
        n_estimators=900, learning_rate=0.08, max_depth=8,
        min_child_weight=12, subsample=0.8, colsample_bytree=0.8,
        gamma=1.0, reg_lambda=1.0, objective="binary:logistic",
        scale_pos_weight=max(1.0, spw), random_state=seed,
        n_jobs=-1, tree_method="hist", eval_metric="logloss",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p_va = model.predict_proba(X_va)[:, 1]
    return model, p_va

def _best_threshold(y_true, prob):
    ths = np.linspace(0.01, 0.99, 99)
    best = (0.0, 0.5, 0.0, 0.0)  # f1, th, r, p
    for th in ths:
        y = (prob >= th).astype(int)
        f1 = f1_score(y_true, y, zero_division=0)
        if f1 > best[0]:
            best = (f1, th, recall_score(y_true, y, zero_division=0),
                    precision_score(y_true, y, zero_division=0))
    return best  # f1, th, rec, prec

def _by_topk(y_true, prob, k: int):
    order = np.argsort(-prob)
    y = np.zeros_like(prob, dtype=int)
    y[order[:max(1, k)]] = 1
    return (
        f1_score(y_true, y, zero_division=0),
        recall_score(y_true, y, zero_division=0),
        precision_score(y_true, y, zero_division=0)
    )

# ---------- ES 主要流程 ----------
def init_individual(d: int, n_hi: int, pos_val: int) -> Individual:
    mask = (np.random.rand(d) < 0.4).astype(np.int8)
    if mask.sum() == 0:
        mask[np.random.randint(0, d)] = 1

    hi_idx = np.random.randint(0, n_hi) if n_hi > 0 else 0
    k_mult = float(np.clip(np.random.normal(1.2, 0.3), 0.5, 3.0))
    mode = np.random.choice(["topk", "threshold"])

    # ES：初始化步長 / 變異率
    sigma_k = 0.2      # 控制 k_mult 的變異強度，之後會自動調
    p_flip = 0.06      # 初始 bit-flip 機率，之後也會自動調

    return Individual(
        mask=mask,
        hi_idx=hi_idx,
        k_mult=k_mult,
        mode=mode,
        sigma_k=sigma_k,
        p_flip=p_flip,
    )

def mutate(ind: Individual, n_hi: int = 0) -> Individual:
    # ===== 1. 先對「步長 / 變異率」做 self-adaptation (ES 精髓) =====
    # 這裡用經典的 log-normal：sigma' = sigma * exp(τ * N(0,1))
    tau = 1.0 / math.sqrt(2.0)   # 你也可以調成 1/sqrt(2*d)
    sigma_k = float(max(1e-4, ind.sigma_k * math.exp(tau * np.random.normal())))
    p_flip  = float(np.clip(ind.p_flip * math.exp(tau * np.random.normal()), 1e-4, 0.5))

    # ===== 2. 用更新後的 p_flip 來變異 mask =====
    m = ind.mask.copy()
    flip = np.random.rand(len(m)) < p_flip
    m[flip] ^= 1
    if m.sum() == 0:
        m[np.random.randint(0, len(m))] = 1

    # ===== 3. 變異 hi_idx（離散：保留你原本的 ±1 邏輯） =====
    hi_idx = ind.hi_idx
    if n_hi > 0 and np.random.rand() < 0.3:
        hi_idx = int(np.clip(hi_idx + np.random.choice([-1, 1]), 0, n_hi - 1))

    # ===== 4. 用 sigma_k 控制 k_mult 的變異幅度 =====
    k_mult = float(np.clip(ind.k_mult + sigma_k * np.random.normal(), 0.5, 3.0))

    # ===== 5. mode 小機率在 "topk" / "threshold" 間翻轉 =====
    mode = ind.mode if np.random.rand() > 0.15 else (
        "topk" if ind.mode == "threshold" else "threshold"
    )

    return Individual(
        mask=m,
        hi_idx=hi_idx,
        k_mult=k_mult,
        mode=mode,
        sigma_k=sigma_k,
        p_flip=p_flip,
    )

# def evaluate(
#     ind: Individual,
#     df_tr_full: pd.DataFrame,
#     df_va_full: pd.DataFrame,
#     feature_pool: List[str],
#     hi_feat_candidates: List[str],
#     base_spw: float,
#     alpha: float = 0.01,
#     seed: int = 42,
# ) -> Tuple[float, Dict[str, Any], XGBClassifier, List[str]]:
#     # 1) 選欄
#     feats = [f for f, on in zip(feature_pool, ind.mask) if on]
#     if len(feats) == 0:
#         return -1e9, {"f1": 0.0}, None, []

#     # 2) 若有大額占比候選欄位，加入該索引選到的欄位
#     if len(hi_feat_candidates) > 0:
#         hi_name = hi_feat_candidates[ind.hi_idx]
#         if hi_name not in feats and hi_name in df_tr_full.columns:
#             feats.append(hi_name)

#     # 3) 構建 X / y
#     use_cols = feats + ["label"]
#     X_tr, y_tr = _prepare_xy(df_tr_full[use_cols], feats)
#     X_va, y_va = _prepare_xy(df_va_full[use_cols], feats)

#     # 4) 訓練 + 產生驗證機率
#     model, p_va = _fit_xgb(X_tr, y_tr, X_va, y_va, spw=base_spw, seed=seed)

#     # 5) 依個體模式得到 F1
#     if ind.mode == "topk":
#         k = int(max(1, np.round(ind.k_mult * y_va.sum())))
#         f1, r, p = _by_topk(y_va, p_va, k)
#         metrics = {"mode": "topk", "k": k, "f1": float(f1), "recall": float(r), "precision": float(p)}
#     else:
#         f1, th, r, p = _best_threshold(y_va, p_va)
#         metrics = {"mode": "threshold", "th": float(th), "f1": float(f1), "recall": float(r), "precision": float(p)}

#     # 6) 複雜度懲罰
#     fitness = 0.7 * metrics["f1"] + 0.3 * metrics["recall"] - alpha * (len(feats) / len(feature_pool))
#     return fitness, metrics, model, feats
def evaluate(
    ind: Individual,
    df_tr_full: pd.DataFrame,
    df_va_full: pd.DataFrame,
    feature_pool: List[str],
    hi_feat_candidates: List[str],
    base_spw: float,
    alpha: float = 0.01,
    seed: int = 42,
    sample_ratio: float = 0.2,   # ⬅ 新增：抽樣比例（預設 20%）
) -> Tuple[float, Dict[str, Any], XGBClassifier, List[str]]:
    # 1) 選欄
    feats = [f for f, on in zip(feature_pool, ind.mask) if on]
    if len(feats) == 0:
        return -1e9, {"f1": 0.0}, None, []

    # 2) 若有大額占比候選欄位，加入該索引選到的欄位
    if len(hi_feat_candidates) > 0:
        hi_name = hi_feat_candidates[ind.hi_idx]
        if hi_name not in feats and hi_name in df_tr_full.columns:
            feats.append(hi_name)

    # 3) 先挑出有用欄位
    use_cols = feats + ["label"]
    df_tr = df_tr_full[use_cols]
    df_va = df_va_full[use_cols]

    # 3.5) 在 DataFrame 層級做抽樣（只用部份帳戶）
    if 0 < sample_ratio < 1.0:
        n_tr = max(1, int(len(df_tr) * sample_ratio))
        tr_idx = np.random.choice(len(df_tr), n_tr, replace=False)
        df_tr = df_tr.iloc[tr_idx]

        n_va = max(1, int(len(df_va) * sample_ratio))
        va_idx = np.random.choice(len(df_va), n_va, replace=False)
        df_va = df_va.iloc[va_idx]

    # 4) 構建 X / y（用抽樣後的 df_tr / df_va）
    X_tr, y_tr = _prepare_xy(df_tr, feats)
    X_va, y_va = _prepare_xy(df_va, feats)

    # 5) 訓練 + 產生驗證機率
    model, p_va = _fit_xgb(X_tr, y_tr, X_va, y_va, spw=base_spw, seed=seed)

    # 5) 依個體模式得到 F1
    if ind.mode == "topk":
        k = int(max(1, np.round(ind.k_mult * y_va.sum())))
        f1, r, p = _by_topk(y_va, p_va, k)
        metrics = {"mode": "topk", "k": k, "f1": float(f1), "recall": float(r), "precision": float(p)}
    else:
        f1, th, r, p = _best_threshold(y_va, p_va)
        metrics = {"mode": "threshold", "th": float(th), "f1": float(f1), "recall": float(r), "precision": float(p)}

    # 6) 複雜度懲罰
    fitness = 0.7 * metrics["f1"] + 0.3 * metrics["recall"] - alpha * (len(feats) / len(feature_pool))
    return fitness, metrics, model, feats



def es_search(
    df_tr_full: pd.DataFrame,
    df_va_full: pd.DataFrame,
    feature_pool: List[str],
    hi_feat_candidates: List[str],
    mu: int = 20, lamb: int = 40, gens: int = 20, seed: int = 42,
    alpha: float = 0.01,
) -> Tuple[Individual, Dict[str, Any], XGBClassifier, List[str]]:
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # 計算 base scale_pos_weight
    y_tr = df_tr_full["label"].values
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    base_spw = max(1.0, neg / (pos + 1e-9))
    print(f"[ES] base_spw={base_spw:.2f}")

    n_hi = len(hi_feat_candidates)
    pos_val = int(df_va_full["label"].sum())

    # 初始化族群
    pop = [init_individual(len(feature_pool), n_hi, pos_val) for _ in range(mu)]

    best_tuple: Optional[Tuple[float, Individual, Dict[str, Any], Any, List[str]]] = None
    no_improve = 0

    for g in range(1, gens + 1):
        # 產生子代
        offspring = [mutate(pop[i % mu], n_hi=n_hi) for i in range(lamb)]
        cand = pop + offspring

        evaluated = []
        for ind in cand:
            fit, info, model, feats = evaluate(
                ind, df_tr_full, df_va_full, feature_pool, hi_feat_candidates, base_spw, alpha=alpha, seed=seed
            )
            evaluated.append((fit, ind, info, model, feats))

        evaluated.sort(key=lambda x: x[0], reverse=True)
        pop = [e[1] for e in evaluated[:mu]]
        best = evaluated[0]
        print(f"[ES] gen={g:02d}  fitness={best[0]:.4f}  f1={best[2]['f1']:.4f}  "
              f"mode={best[2]['mode']}  nfeat={len(best[4])}")

        # # 早停判定
        # if best_tuple is None or best[0] > best_tuple[0]:
        #     best_tuple = best
        #     no_improve = 0
        # else:
        #     no_improve += 1
        # if no_improve >= 4:
        #     print("[ES] Early stop.")
        #     break

    assert best_tuple is not None
    _, best_ind, best_info, best_model, best_feats = best_tuple
    return best_ind, best_info, best_model, best_feats
