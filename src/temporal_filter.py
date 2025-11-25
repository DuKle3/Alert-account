import pandas as pd 
import numpy as np

def filter_transactions_by_event_date(df_txn: pd.DataFrame, df_alert: pd.DataFrame) -> pd.DataFrame:
    """
    Goal: 用 event_date-1 做為每個帳戶的交易截止日
    - if acct is alert: 保留 txn_date < event_date
    - if acct isn't alert: 全部保留
    - if 交易雙方皆為 alert: 取兩者的 event_date (cutoff) 最小值
    """

    df_alert = df_alert.copy()
    df_alert["cutoff_day"] = df_alert["event_date"] - 1
    cutoff_map = df_alert.set_index("acct")["cutoff_day"]

    from_cut = df_txn["from_acct"].map(cutoff_map)
    to_cut = df_txn["to_acct"].map(cutoff_map)

    # 用 DataFrame 的逐列最小值（跳過 NaN），避免用 np.inf
    # 結果 dtype 會變成 float（含 NaN），這沒關係
    pair = pd.concat([from_cut, to_cut], axis=1)
    min_cut = pair.min(axis=1, skipna=True)  # 兩端都有值取較小；一端 NaN 取另一端；兩端都 NaN → NaN

    # mask：若 min_cut 為 NaN（兩端都非警示），則保留全部；否則要求 txn_date <= min_cut
    mask = min_cut.isna() | (df_txn["txn_date"] <= min_cut)

    df_filtered = df_txn[mask].copy()
    print(f"[INFO] Temporal filter (event_date): kept {len(df_filtered):,}/{len(df_txn):,} transactions.")
    return df_filtered

    # 交易雙方皆非 alert
    both_nan = from_cut.isna() & to_cut.isna()

    # 有一方是 alert -> 取較小 cutoff
    # 先用 np.fmin 把 NaN 當成 inf，再還原
    min_cut = np.fmin(from_cut.fillna(np.inf), to_cut.fillna(np.inf))
    min_cut = pd.Series(min_cut, index=df_txn.index).replace(np.inf, np.nan)

    mask = both_nan | (df_txn["txn_date"] <= min_cut)
    out = df_txn[mask].copy()
    print(f"[INFO] Temporal filter (event_date): kept {len(out):,}/{len(df_txn):,} transactions.")
    return out

