import argparse
import numpy as np
import pandas as pd

from src.utils import load_data
from src.temporal_filter import filter_transactions_by_event_date
from src.aggregate import aggregate_accounts
from src.dataset import build_labeled_accounts, train_val_split
from src.submission import _predict_scores

# Optional imports (available in repo)
from src.ea_search import es_search
from src.model_xgb import train_xgb_spw_scan


def _prepare_xy(df: pd.DataFrame, features):
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["label"].values
    return X, y


def quantiles(arr, qs=(0.95, 0.99, 0.995, 0.999)):
    arr = np.asarray(arr)
    if arr.size == 0:
        return {q: np.nan for q in qs}
    return {q: float(np.quantile(arr, q)) for q in qs}


def evaluate_topk(y_true, scores, ks):
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    n = len(scores)
    results = []
    for k in sorted(set(int(max(1, min(n, k))) for k in ks)):
        y_hat = np.zeros(n, dtype=int)
        y_hat[order[:k]] = 1
        p = precision_score(y_true, y_hat, zero_division=0)
        r = recall_score(y_true, y_hat, zero_division=0)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        results.append({"k": int(k), "precision": float(p), "recall": float(r), "f1": float(f1)})
    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnostics: thresholds, distributions, PR-at-K")
    parser.add_argument("--mode", choices=["baseline", "es"], default="baseline",
                        help="Train mode: baseline (fast, spw-scan) or es (evolution search)")
    parser.add_argument("--gens", type=int, default=10, help="ES generations (if --mode es)")
    parser.add_argument("--mu", type=int, default=10, help="ES mu (if --mode es)")
    parser.add_argument("--lamb", type=int, default=20, help="ES lambda (if --mode es)")
    args = parser.parse_args()

    # 1) Load + preprocess
    df_txn, df_alert, df_test = load_data()
    df_txn_f = filter_transactions_by_event_date(df_txn, df_alert)
    df_agg = aggregate_accounts(df_txn_f, ref_day=None, windows=(7, 30, 90))

    df_label = build_labeled_accounts(df_agg, df_alert)
    train_df, valid_df, _ = train_val_split(df_label, df_test, seed=42)

    feature_pool = [c for c in df_agg.columns if c not in ["acct", "label", "event_date"]]
    hi_feat_candidates = [c for c in [
        "hi_amt_ratio_from_w30", "hi_amt_ratio_to_w30",
        "hi_amt_ratio_from_w30_p90", "hi_amt_ratio_to_w30_p90",
        "hi_amt_ratio_from_w30_p99", "hi_amt_ratio_to_w30_p99"
    ] if c in df_agg.columns]

    # 2) Train model and get features used
    if args.mode == "es":
        best_ind, best_info, model, feats = es_search(
            df_tr_full=train_df,
            df_va_full=valid_df,
            feature_pool=feature_pool,
            hi_feat_candidates=hi_feat_candidates,
            mu=args.mu, lamb=args.lamb, gens=args.gens, seed=42, alpha=0.01,
        )
        info = best_info
    else:
        model, metrics, feats = train_xgb_spw_scan(train_df, valid_df, seed=42)
        info = metrics

    # 3) Scores on validation
    X_va, y_va = _prepare_xy(valid_df, feats)
    p_va = _predict_scores(model, X_va)
    pos_va = int(y_va.sum())

    print("\n=== Validation label count ===")
    print(f"y_va sum (positives) = {pos_va}")

    # Determine decision and count predicted positives on validation
    pred_pos_va = None
    if info.get("mode") == "threshold":
        th = info.get("best_th") or info.get("th")
        if th is None:
            # fallback: search best threshold by ourselves
            qs = np.linspace(0.01, 0.99, 99)
            best = (0.0, 0.5)
            from sklearn.metrics import f1_score
            for t in qs:
                f1 = f1_score(y_va, (p_va >= t).astype(int), zero_division=0)
                if f1 > best[0]:
                    best = (f1, t)
            th = best[1]
        y_hat = (p_va >= float(th)).astype(int)
        pred_pos_va = int(y_hat.sum())
        print("\n=== Best threshold diagnostics ===")
        print(f"best_th = {float(th):.6f}; predicted positives on valid = {pred_pos_va}")
    elif info.get("mode") == "topk":
        k = info.get("best_k") or info.get("k")
        k = int(k) if k is not None else max(1, pos_va)
        pred_pos_va = int(k)
        print("\n=== Best top-k diagnostics ===")
        print(f"best_k = {int(k)}; predicted positives on valid = {pred_pos_va}")
    else:
        print("\n[WARN] Unknown decision mode; skipping decision diagnostics.")

    # 4) Score distributions: validation vs test
    print("\n=== Score distribution percentiles ===")
    q_va = quantiles(p_va)
    print("Validation percentiles (score):", {str(k): round(v, 6) for k, v in q_va.items()})

    test_accts = set(df_test["acct"])
    df_test_aligned = df_agg[df_agg["acct"].isin(test_accts)].copy()
    X_te = df_test_aligned[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p_te = _predict_scores(model, X_te)
    q_te = quantiles(p_te)
    print("Test percentiles (score):     ", {str(k): round(v, 6) for k, v in q_te.items()})

    # If threshold exists, count predicted positives on test to see drift impact
    if info.get("mode") == "threshold":
        th = info.get("best_th") or info.get("th")
        if th is not None:
            pred_pos_test = int((p_te >= float(th)).sum())
            print(f"Predicted positives on test with best_th: {pred_pos_test}")

    # 5) PR-at-K on validation around y_va.sum()
    print("\n=== PR-at-K on validation (coarse multiples) ===")
    ks = [pos_va, int(1.5*pos_va), 2*pos_va, 3*pos_va, 5*pos_va]
    ks = [k for k in ks if k >= 1]
    prk = evaluate_topk(y_va, p_va, ks)
    for row in prk:
        print(f"K={row['k']:>6}  F1={row['f1']:.4f}  R={row['recall']:.4f}  P={row['precision']:.4f}")

    print("\n=== PR-at-K detailed sweep (K=80..220 step 10) ===")
    sweep_ks = list(range(80, 221, 10))
    sweep_results = evaluate_topk(y_va, p_va, sweep_ks)
    best_row = None
    for row in sweep_results:
        print(f"K={row['k']:>6}  F1={row['f1']:.4f}  R={row['recall']:.4f}  P={row['precision']:.4f}")
        if (best_row is None) or (row['f1'] > best_row['f1']) or (
            row['f1'] == best_row['f1'] and row['recall'] > best_row['recall']
        ):
            best_row = row
    if best_row:
        print(f"Best K in sweep: K={best_row['k']} with F1={best_row['f1']:.4f}, R={best_row['recall']:.4f}, P={best_row['precision']:.4f}")

    # 6) Suggested top-k positives on test for reference
    #    (cannot compute PR without labels, but show how many 1s would be predicted)
    print("\n=== Suggested top-k counts on test (no labels) ===")
    for k in [pos_va, 2*pos_va, 3*pos_va]:
        k = int(max(1, min(len(p_te), k)))
        print(f"If top-k with K={k}, predicted positives on test = {k}")


if __name__ == "__main__":
    main()
