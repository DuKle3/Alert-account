# # src/submission.py
# import numpy as np
# import pandas as pd

# def _align_test(df_agg_full: pd.DataFrame, df_test: pd.DataFrame, feature_names):
#     dfj = df_test[["acct"]].merge(df_agg_full, on="acct", how="left").fillna(0)
#     X = dfj[feature_names].values
#     return X, dfj

# def submit_with_rule(model, df_agg_full, df_test, feature_names, mode: str, decision, out_path: str):
#     X, dfj = _align_test(df_agg_full, df_test, feature_names)
#     prob = model.predict_proba(X)[:, 1]

#     if mode == "threshold":
#         th = float(decision)
#         y = (prob >= th).astype(int)
#     else:
#         k = int(decision)
#         order = np.argsort(-prob)
#         y = np.zeros_like(prob, dtype=int)
#         y[order[:max(1, k)]] = 1

#     out = pd.DataFrame({"acct": dfj["acct"].values, "label": y})
#     out.to_csv(out_path, index=False)
#     print(f"[SUBMIT] Saved to {out_path}  (mode={mode}, decision={decision})")

# src/submission.py
import numpy as np
import pandas as pd

def _predict_scores(model, X):
    """回傳二分類正類分數（越大越可疑）。盡量相容 sklearn / xgboost 兩種介面。"""
    # 1) sklearn / xgboost.XGBClassifier 介面
    try:
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    except Exception:
        pass

    # 2) 原生 xgboost Booster 介面
    try:
        import xgboost as xgb
        dtest = xgb.DMatrix(X)
        return np.asarray(model.predict(dtest)).ravel()
    except Exception:
        pass

    # 3) 退而求其次：用 predict 當分數
    return np.asarray(model.predict(X)).ravel()

def submit_with_rule(model, df_agg, df_test, features, mode, value, out_path):
    """
    產生符合比賽規格的提交檔 acct_predict.csv
    需包含 acct, label 兩欄：
      - label = 1 → 警示帳戶
      - label = 0 → 非警示帳戶
    """
    import numpy as np
    import pandas as pd

    # 只保留 test 帳戶
    test_accts = set(df_test["acct"])
    df = df_agg[df_agg["acct"].isin(test_accts)].copy()

    # 預測分數
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scores = _predict_scores(model, X)
    df["score"] = scores

    # 排序由高到低
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # 規則：threshold 或 topk
    if mode == "threshold":
        th = float(value)
        df["label"] = (df["score"] >= th).astype(int)
    elif mode == "topk":
        k = int(value)
        df["label"] = 0
        df.loc[:k-1, "label"] = 1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 只輸出 acct, label
    out = df[["acct", "label"]]
    out.to_csv(out_path, index=False)
    print(f"[SUBMIT] Saved submission file: {out_path} (shape={out.shape})")

    return out

