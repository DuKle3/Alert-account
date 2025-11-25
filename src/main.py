import argparse
from typing import List

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from src.utils import load_data
from src.aggregate import aggregate_accounts
from src.temporal_filter import filter_transactions_by_event_date
from src.dataset import build_labeled_accounts, train_val_split
# from src.model_xgb import train_xgb_spw_scan  # 若要跑 baseline 再打開
from src.ea_search import es_search
from src.submission import submit_with_rule, _predict_scores


def evaluate_topk(y_true, scores, ks: List[int]):
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
    results.sort(key=lambda row: (-row["f1"], -row["recall"], row["k"]))
    best = results[0] if results else None
    return results, best

def parse_args():
    parser = argparse.ArgumentParser(description="ES search + submission")
    parser.add_argument("--auto-topk", action="store_true",
                        help="同時輸出依驗證掃描所得最佳 K 的 top-k 提交檔")
    parser.add_argument("--bestk-start", type=int, default=80,
                        help="最佳 K 掃描起點 (含)，僅在 --auto-topk 下生效")
    parser.add_argument("--bestk-end", type=int, default=220,
                        help="最佳 K 掃描終點 (含)，僅在 --auto-topk 下生效")
    parser.add_argument("--bestk-step", type=int, default=10,
                        help="最佳 K 掃描步長，僅在 --auto-topk 下生效")
    return parser.parse_args()


def main(args):
    df_txn, df_alert, df_test = load_data()
    df_txn_f = filter_transactions_by_event_date(df_txn, df_alert)
    df_agg = aggregate_accounts(df_txn_f, ref_day=None, windows=(7, 30, 90))

    df_label = build_labeled_accounts(df_agg, df_alert)
    train, valid, _ = train_val_split(df_label, df_test, seed=42)

    # ---- Feature pools ----
    feature_pool = [c for c in df_agg.columns if c not in ["acct", "label", "event_date"]]
    hi_feat_candidates = [c for c in [
        "hi_amt_ratio_from_w30", "hi_amt_ratio_to_w30",
        "hi_amt_ratio_from_w30_p90", "hi_amt_ratio_to_w30_p90",
        "hi_amt_ratio_from_w30_p99", "hi_amt_ratio_to_w30_p99"
    ] if c in df_agg.columns]

    # ---- Sanity checks ----
    missing_in_train = [c for c in feature_pool if c not in train.columns]
    missing_in_valid = [c for c in feature_pool if c not in valid.columns]
    assert not missing_in_train, f"train 缺少特徵: {missing_in_train[:10]}"
    assert not missing_in_valid, f"valid 缺少特徵: {missing_in_valid[:10]}"
    assert "label" in train.columns and "label" in valid.columns, "train/valid 缺少 label 欄位"

    bad_cols = []
    for c in feature_pool:
        if train[c].isna().any() or valid[c].isna().any(): bad_cols.append(c)
        if np.isinf(train[c].to_numpy(dtype=float, copy=False)).any() or np.isinf(valid[c].to_numpy(dtype=float, copy=False)).any(): bad_cols.append(c)
    assert not bad_cols, f"發現 NaN/Inf 特徵: {sorted(set(bad_cols))[:10]}"

    # ---- ES search ----
    best_ind, best_info, best_model, best_feats = es_search(
        df_tr_full=train,
        df_va_full=valid,
        feature_pool=feature_pool,
        hi_feat_candidates=hi_feat_candidates,
        mu=10, lamb=20, gens=10, seed=42, alpha=0.01,
    )
    assert best_info["mode"] in {"threshold", "topk"}, f"未知 mode: {best_info['mode']}"
    print(f"[BEST] mode={best_info['mode']}  f1={best_info['f1']:.4f}  #feat={len(best_feats)}")

    # ---- Submission ----
    X_va = valid[best_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scores_va = _predict_scores(best_model, X_va)
    y_va = valid["label"].values

    if best_info["mode"] == "threshold":
        best_th = float(best_info["th"])

        # 取得驗證集在 best_th 下的預測數量
        val_pos = int((scores_va >= best_th).sum())
        if val_pos <= 0:
            val_pos = 1
        print(f"[SUBMIT] Validation positives @th={best_th:.6f}: {val_pos}")

        # 門檻版提交
        threshold_out = submit_with_rule(
            best_model, df_agg, df_test, best_feats,
            "threshold", best_th, "result_threshold_es.csv"
        )
        print(f"[SUBMIT] Threshold predicted positives (test): {int(threshold_out['label'].sum())}")

        # 同步輸出 top-k（K = 驗證門檻下的預測數，維持排序）
        k_mapped = int(min(len(df_test), max(1, val_pos)))
        topk_out = submit_with_rule(
            best_model, df_agg, df_test, best_feats,
            "topk", k_mapped, "result_topk_mapped_es.csv"
        )
        print(f"[SUBMIT] Mapped top-k positives (test): {int(topk_out['label'].sum())}")
    else:
        submit_with_rule(best_model, df_agg, df_test, best_feats, "topk", best_info["k"], "result_topk_es.csv")
        val_pos = best_info["k"]

    # 依驗證找最佳 K 的 top-k 提交（選用）
    if args.auto_topk:
        start = max(1, args.bestk_start)
        end = max(start, args.bestk_end)
        step = max(1, args.bestk_step)
        ks = list(range(start, end + 1, step))
        ks = [k for k in ks if k <= len(y_va)]
        if not ks:
            ks = [min(len(y_va), max(1, int(val_pos)))]
        _, best_row = evaluate_topk(y_va, scores_va, ks)
        if best_row:
            best_k = int(best_row["k"])
            print(f"[SUBMIT] Auto best-K on valid: K={best_k}, F1={best_row['f1']:.4f}, R={best_row['recall']:.4f}, P={best_row['precision']:.4f}")
            bestk_out = submit_with_rule(
                best_model, df_agg, df_test, best_feats,
                "topk", best_k, "result_topk_bestk_es.csv"
            )
            print(f"[SUBMIT] Auto top-k positives (test): {int(bestk_out['label'].sum())}")
        else:
            print("[SUBMIT] Auto top-k: 未找到合適的 K，略過。")


if __name__ == "__main__":
    args = parse_args()
    main(args)
