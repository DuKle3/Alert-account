import pandas as pd 

def build_labeled_accounts(df_agg: pd.DataFrame, df_alert: pd.DataFrame) -> pd.DataFrame:
    """
    將 df_agg 依照 acct 合併 df_alert，產生帳戶標籤 label (1/0), 並保留 event_date (警示日期)
    df_alert: acct, event_date
    """
    df_label = df_alert.copy()
    df_label["label"] = 1 
    df_label = df_label[["acct", "label", "event_date"]]

    df_label = df_agg.merge(df_label, on="acct", how="left")
    df_label["label"] = df_label["label"].fillna(0).astype(int)

    print(f"[INFO] Finish build_labeld_accounts")
    return df_label


from sklearn.model_selection import StratifiedGroupKFold
def train_val_split(df_lab: pd.DataFrame, df_test: pd.DataFrame, seed=42):
    """
    帳戶級 stratified split：確保相同 acct 不同切到 train/val（防洩漏）
    同時排除 test 清單中的帳戶（僅以官方樣板的策略）
    """
    # 先排除 test 帳戶（不拿來訓練）
    test_accts = set(df_test["acct"].unique())
    df_train_cand = df_lab[~df_lab["acct"].isin(test_accts)].copy()
    df_eval_cand  = df_lab[df_lab["acct"].isin(test_accts)].copy()

    # 帳戶層級 stratified split
    X = df_train_cand.drop(columns=["acct", "label"])
    y = df_train_cand["label"].values
    groups = df_train_cand["acct"].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    # 取第一折作為 val（簡單起手式）
    for train_idx, val_idx in sgkf.split(X, y, groups):
        tr = df_train_cand.iloc[train_idx].reset_index(drop=True)
        va = df_train_cand.iloc[val_idx].reset_index(drop=True)
        break

    print(f"[INFO] Train accounts: {tr['acct'].nunique():,}, Val accounts: {va['acct'].nunique():,}")
    print(f"[INFO] Class ratio train: pos={tr['label'].sum():,}/{len(tr):,}, val: pos={va['label'].sum():,}/{len(va):,}")
    return tr, va, df_eval_cand  # df_eval_cand 不是官方 test.csv，要預測時另處理
