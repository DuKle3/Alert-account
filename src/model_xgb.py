# # # import numpy as np 
# # # import pandas as pd 
# # # from xgboost import XGBClassifier 
# # # from sklearn.metrics import f1_score, recall_score, precision_score 

# # # def _prepare_X_y(df: pd.DataFrame):
# # #     X = df.drop(columns=["acct", "label", "event_date"])
# # #     y = df["label"].values 
# # #     return X, y


# # # def train_xgb_baseline(df_train: pd.DataFrame, df_val: pd.DataFrame, seed=42):
# # #     X_train, y_train = _prepare_X_y(df_train)
# # #     X_valid, y_valid = _prepare_X_y(df_val)

# # #     neg = (y_train == 0).sum()
# # #     pos = (y_train == 1).sum()
# # #     scale_pos_weight = max(1.0, neg / (pos + 1e-9))
# # #     print(f"[INFO] scale_pos_weight={scale_pos_weight: .2f} (neg={neg}, pos={pos})")

# # #     model = XGBClassifier(
# # #         n_estimators=400,
# # #         learning_rate=0.05,
# # #         max_depth=6,
# # #         subsample=0.8,
# # #         colsample_bytree=0.8,
# # #         reg_lambda=1.0,
# # #         objective="binary:logistic",
# # #         scale_pos_weight=scale_pos_weight,
# # #         random_state=seed,
# # #         n_jobs=-1,
# # #         tree_method="hist",
# # #         eval_metric="logloss"
# # #     )

# # #     model.fit(
# # #         X_train, y_train,
# # #         eval_set=[(X_valid, y_valid)],
# # #         verbose=False
# # #     )

# # #     def find_best_threshold(y_true, probas):
# # #         # 在 [0.01, 0.99] 搜一圈；也可用分位數切點更快
# # #         cand = np.linspace(0.01, 0.99, 99)
# # #         best = (0.0, 0.0)  # (best_f1, best_th)
# # #         for th in cand:
# # #             y_hat = (probas >= th).astype(int)
# # #             f1 = f1_score(y_true, y_hat, zero_division=0)
# # #             if f1 > best[0]:
# # #                 best = (f1, th)
# # #         return best

# # #     # pred
# # #     p_val = model.predict_proba(X_valid)[:, 1]
# # #     best_f1, th = find_best_threshold(y_valid, p_val)
# # #     y_pred = (p_val >= th).astype(int)

# # #     f1 = f1_score(y_valid, y_pred, zero_division=0)
# # #     recall = recall_score(y_valid, y_pred, zero_division=0)
# # #     precision = precision_score(y_valid, y_pred, zero_division=0)

# # #     print(f"[RESULT] Val F1={f1:.4f} | Recall={recall:.4f} | Precision={precision:.4f}")
# # #     return model, {"f1": f1, "recall": recall, "precision": precision}

# # # src/model_xgb.py
# # import numpy as np
# # import pandas as pd
# # from xgboost import XGBClassifier
# # from sklearn.metrics import f1_score, recall_score, precision_score

# # def _prepare_X_y(df: pd.DataFrame):
# #     feats = [c for c in df.columns if c not in ["acct", "label", "event_date"]]
# #     X = df[feats]
# #     y = df["label"].values
# #     return X, y, feats

# # def find_best_threshold(y_true, probas):
# #     cand = np.linspace(0.01, 0.99, 99)
# #     best = (0.0, 0.5)
# #     for th in cand:
# #         y_hat = (probas >= th).astype(int)
# #         f1 = f1_score(y_true, y_hat, zero_division=0)
# #         if f1 > best[0]:
# #             best = (f1, th)
# #     return best  # (best_f1, best_th)

# # def topk_by_count(probas, k):
# #     idx = np.argsort(-probas)[:k]
# #     y_hat = np.zeros_like(probas, dtype=int)
# #     y_hat[idx] = 1
# #     return y_hat

# # def train_xgb_baseline(df_tr: pd.DataFrame, df_va: pd.DataFrame, seed=42):
# #     X_tr, y_tr, feats = _prepare_X_y(df_tr)
# #     X_va, y_va, _     = _prepare_X_y(df_va)

# #     neg = (y_tr == 0).sum()
# #     pos = (y_tr == 1).sum()
# #     spw = max(1.0, neg / (pos + 1e-9))
# #     print(f"[INFO] scale_pos_weight={spw:.2f} (neg={neg}, pos={pos})")

# #     model = XGBClassifier(
# #         n_estimators=500,
# #         learning_rate=0.08,
# #         max_depth=7,
# #         subsample=0.8,
# #         colsample_bytree=0.8,
# #         reg_lambda=1.0,
# #         objective="binary:logistic",
# #         scale_pos_weight=spw,
# #         random_state=seed,
# #         n_jobs=-1,
# #         tree_method="hist",      # 有 GPU 可改 'gpu_hist'
# #         eval_metric="logloss"
# #     )

# #     model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

# #     p_va = model.predict_proba(X_va)[:, 1]

# #     # Best-Threshold
# #     best_f1, best_th = find_best_threshold(y_va, p_va)
# #     y_pred_th = (p_va >= best_th).astype(int)
# #     f1  = f1_score(y_va, y_pred_th, zero_division=0)
# #     rec = recall_score(y_va, y_pred_th, zero_division=0)
# #     pre = precision_score(y_va, y_pred_th, zero_division=0)
# #     print(f"[RESULT] BestTH={best_th:.3f} | F1={f1:.4f} | R={rec:.4f} | P={pre:.4f}")

# #     # Top-K（以驗證集陽性數為基準）
# #     k = int(y_va.sum())  # 也可 *2 或 *3
# #     y_pred_topk = topk_by_count(p_va, k)
# #     f1k  = f1_score(y_va, y_pred_topk, zero_division=0)
# #     reck = recall_score(y_va, y_pred_topk, zero_division=0)
# #     prek = precision_score(y_va, y_pred_topk, zero_division=0)
# #     print(f"[RESULT] Top-{k} | F1={f1k:.4f} | R={reck:.4f} | P={prek:.4f}")

# #     metrics = {
# #         "best_th": best_th,
# #         "best_f1": f1,
# #         "best_recall": rec,
# #         "best_precision": pre,
# #         "topk_k": k,
# #         "topk_f1": f1k,
# #         "topk_recall": reck,
# #         "topk_precision": prek
# #     }
# #     return model, metrics, feats

# import numpy as np
# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.metrics import f1_score, recall_score, precision_score

# def _prepare_X_y(df: pd.DataFrame):
#     feats = [c for c in df.columns if c not in ["acct", "label", "event_date"]]
#     return df[feats], df["label"].values, feats

# def find_best_threshold(y_true, probas):
#     cand = np.linspace(0.01, 0.99, 99)
#     best = (0.0, 0.5)
#     for th in cand:
#         f1 = f1_score(y_true, (probas >= th).astype(int), zero_division=0)
#         if f1 > best[0]:
#             best = (f1, th)
#     return best  # (f1, th)

# def search_best_k(y_true, probas, ks):
#     order = np.argsort(-probas)
#     best = (0.0, None)
#     for k in ks:
#         y_hat = np.zeros_like(probas, dtype=int)
#         y_hat[order[:k]] = 1
#         f1 = f1_score(y_true, y_hat, zero_division=0)
#         if f1 > best[0]:
#             best = (f1, k)
#     return best  # (f1, k)

# def train_xgb_baseline(df_tr: pd.DataFrame, df_va: pd.DataFrame, seed=42):
#     X_tr, y_tr, feats = _prepare_X_y(df_tr)
#     X_va, y_va, _     = _prepare_X_y(df_va)

#     neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
#     spw = max(1.0, neg / (pos + 1e-9))
#     print(f"[INFO] scale_pos_weight={spw:.2f} (neg={neg}, pos={pos})")

#     model = XGBClassifier(
#         n_estimators=900,
#         learning_rate=0.08,
#         max_depth=8,
#         min_child_weight=12,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         gamma=1.0,
#         reg_lambda=1.0,
#         objective="binary:logistic",
#         scale_pos_weight=spw,
#         random_state=seed,
#         n_jobs=-1,
#         tree_method="hist",   # GPU 可改 'gpu_hist'
#         eval_metric="logloss"
#     )
#     model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

#     p_va = model.predict_proba(X_va)[:, 1]

#     # ① Best-Threshold
#     f1_th, best_th = find_best_threshold(y_va, p_va)
#     y_th = (p_va >= best_th).astype(int)
#     print(f"[RESULT] BestTH={best_th:.3f} | F1={f1_th:.4f} | R={recall_score(y_va,y_th,zero_division=0):.4f} | P={precision_score(y_va,y_th,zero_division=0):.4f}")

#     # ② Best-TopK（對數刻度掃描）
#     ks = np.unique(np.int32(np.logspace(2, 5, 60)))  # 100~100000
#     f1_k, best_k = search_best_k(y_va, p_va, ks)
#     # 回報該 K 的 R/P
#     order = np.argsort(-p_va)
#     y_topk = np.zeros_like(p_va, dtype=int); y_topk[order[:best_k]] = 1
#     print(f"[RESULT] BestTopK={best_k} | F1={f1_k:.4f} | R={recall_score(y_va,y_topk,zero_division=0):.4f} | P={precision_score(y_va,y_topk,zero_division=0):.4f}")

#     metrics = {"best_th": best_th, "best_th_f1": f1_th, "best_k": int(best_k), "best_k_f1": f1_k}
#     return model, metrics, feats
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score, precision_score

def _prepare_X_y(df: pd.DataFrame):
    feats = [c for c in df.columns if c not in ["acct","label","event_date"]]
    return df[feats], df["label"].values, feats

def find_best_threshold(y_true, probas):
    cand = np.linspace(0.01, 0.99, 99)
    best = (0.0, 0.5)
    for th in cand:
        f1 = f1_score(y_true, (probas >= th).astype(int), zero_division=0)
        if f1 > best[0]:
            best = (f1, th)
    return best

def search_best_k(y_true, probas, ks):
    order = np.argsort(-probas)
    best = (0.0, None)
    for k in ks:
        y_hat = np.zeros_like(probas, dtype=int)
        y_hat[order[:k]] = 1
        f1 = f1_score(y_true, y_hat, zero_division=0)
        if f1 > best[0]:
            best = (f1, k)
    return best

def _fit_eval_once(X_tr, y_tr, X_va, y_va, spw, seed=42):
    model = XGBClassifier(
        n_estimators=1000, learning_rate=0.08, max_depth=8,
        min_child_weight=12, subsample=0.8, colsample_bytree=0.8,
        gamma=1.0, reg_lambda=1.0, objective="binary:logistic",
        scale_pos_weight=spw, random_state=seed, n_jobs=-1,
        tree_method="hist", eval_metric="logloss"
    )
    # Fit with early stopping when available; fallback for older xgboost versions
    try:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_va, y_va)],
            verbose=False,
            early_stopping_rounds=50,
        )
    except TypeError:
        # Older xgboost may not accept early_stopping_rounds in sklearn API
        try:
            import xgboost as xgb
            cb = [xgb.callback.EarlyStopping(rounds=50)]
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_tr, y_tr), (X_va, y_va)],
                verbose=False,
                callbacks=cb,
            )
        except Exception:
            # Final fallback: train without early stopping
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_tr, y_tr), (X_va, y_va)],
                verbose=False,
            )
    p_va = model.predict_proba(X_va)[:, 1]

    # A) Best-Threshold
    f1_th, best_th = find_best_threshold(y_va, p_va)
    y_th = (p_va >= best_th).astype(int)
    r_th = recall_score(y_va, y_th, zero_division=0)
    p_th = precision_score(y_va, y_th, zero_division=0)

    # B) Best-TopK（對數刻度掃描）
    ks = np.unique(np.int32(np.logspace(2, 5, 60)))
    f1_k, best_k = search_best_k(y_va, p_va, ks)
    order = np.argsort(-p_va); y_k = np.zeros_like(p_va, dtype=int); y_k[order[:best_k]] = 1
    r_k = recall_score(y_va, y_k, zero_division=0)
    p_k = precision_score(y_va, y_k, zero_division=0)

    # 以兩者較佳者為主（你也可以只選一種）
    if f1_k >= f1_th:
        metrics = {"mode":"topk","best_k":int(best_k),"f1":float(f1_k),"recall":float(r_k),"precision":float(p_k),
                   "best_th":None,"f1_th":float(f1_th),"recall_th":float(r_th),"precision_th":float(p_th)}
    else:
        metrics = {"mode":"threshold","best_k":None,"f1":float(f1_th),"recall":float(r_th),"precision":float(p_th),
                   "best_th":float(best_th),"f1_k":float(f1_k),"recall_k":float(r_k),"precision_k":float(p_k)}
    return model, metrics

def train_xgb_spw_scan(df_tr: pd.DataFrame, df_va: pd.DataFrame, seed=42, multipliers=(0.5, 1.0, 1.5, 2.0)):
    X_tr, y_tr, feats = _prepare_X_y(df_tr)
    X_va, y_va, _     = _prepare_X_y(df_va)
    neg, pos = (y_tr==0).sum(), (y_tr==1).sum()
    base_spw = max(1.0, neg / (pos + 1e-9))

    best = None
    for m in multipliers:
        spw = max(1.0, base_spw * m)
        model, metrics = _fit_eval_once(X_tr, y_tr, X_va, y_va, spw, seed=seed)
        print(f"[SCAN] spw={spw:.2f} | mode={metrics['mode']} | F1={metrics['f1']:.4f} | R={metrics['recall']:.4f} | P={metrics['precision']:.4f}")
        if (best is None) or (metrics["f1"] > best[1]["f1"]):
            best = (model, metrics, spw)

    model, metrics, spw = best
    print(f"[BEST] spw={spw:.2f} => mode={metrics['mode']} | F1={metrics['f1']:.4f} | R={metrics['recall']:.4f} | P={metrics['precision']:.4f}")
    # 補回必要資訊，方便後續 submission
    if metrics["mode"] == "threshold":
        metrics["best_th"] = metrics["best_th"]
        metrics["best_k"]  = None
    else:
        metrics["best_k"]  = metrics["best_k"]
        metrics["best_th"] = None
    return model, metrics, feats

# 為了與 main.py 既有流程相容，提供薄包裝：
def train_xgb_baseline(df_train: pd.DataFrame, df_val: pd.DataFrame, seed=42):
    model, metrics, _ = train_xgb_spw_scan(df_train, df_val, seed=seed)
    return model, metrics
