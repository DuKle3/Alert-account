import os
import pandas as pd 

def load_data(dir_path="data/raw/"):
    """
    Args:
        dif_path (str): 資料夾路徑，需包含:
            - acct_transaction.csv
            - acct_alert.csv
            - acct_predict.csv
    Returns:
        df_txn   (DataFrame): 交易資料
        df_alert (DataFrame): 警示帳戶 
        df_test  (DataFrame): 預測目標帳戶
    """

    # check is dir_path exist
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"🤡 Directory not found: {dir_path}")

    # load data
    df_txn = pd.read_csv(os.path.join(dir_path, "acct_transaction.csv"))
    df_alert = pd.read_csv(os.path.join(dir_path, "acct_alert.csv"))
    df_test = pd.read_csv(os.path.join(dir_path, "acct_predict.csv"))

    # log
    print(f"[INFO] 🚾 Dataset Loaded from: {dir_path}")
    print(f" Transactions: {len(df_txn):,} rows x {df_txn.shape[1]} cols")
    print(f" Alert accounts: {len(df_alert):,}")
    print(f" Test accounts: {len(df_test):,}")


    # 處理 txn_date: 交易日期 (切齊第一天), event_date: 警示日期
    df_txn["txn_date"] = pd.to_numeric(df_txn["txn_date"], errors="coerce").astype("Int64")
    df_alert["event_date"] = pd.to_numeric(df_alert["event_date"], errors="coerce").astype("Int64")
    print(f"[INFO] day_index range: {int(df_txn['txn_date'].min())} -> {int(df_txn['txn_date'].max())}")

    # check missing value
    total_missing = df_txn.isna().sum().sum()
    if total_missing > 0:
        print(f"[WARN] Missing values in transaction data: {total_missing}")
    else:
        print(f"[INFO] No missing values")

    # 顯示 columns
    print(f"[INFO] Transaction Columns: {list(df_txn.columns)}")

    # check from_acct / to_acct 是否 overlap
    # 初步了解帳戶結構（寄出/收款與重疊）
    unique_from = df_txn["from_acct"].nunique()
    unique_to = df_txn["to_acct"].nunique()
    overlap = len(set(df_txn["from_acct"]) & set(df_txn["to_acct"]))
    print(f"Unique senders: {unique_from}, receivers: {unique_to}, overlap: {overlap}")

    return df_txn, df_alert, df_test