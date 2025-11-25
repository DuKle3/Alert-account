# import pandas as pd


# def aggregate_accounts(df_txn: pd.DataFrame, ref_day: int | None = None, windows=(7, 30, 90)) -> pd.DataFrame:
#     if ref_day is None:
#         ref_day = int(df_txn["txn_date"].max())

#     base = df_txn[df_txn["txn_date"] <= ref_day].copy()

#     # from acct
#     send = base.groupby("from_acct")["txn_amt"].agg(
#         total_send_amt="sum", avg_send_amt="mean",
#         max_send_amt="max", min_send_amt="min", send_count="count"
#     ).reset_index().rename(columns={"from_acct":"acct"})

#     # to acct
#     recv = base.groupby("to_acct")["txn_amt"].agg(
#         total_recv_amt="sum", avg_recv_amt="mean",
#         max_recv_amt="max", min_recv_amt="min", recv_count="count"
#     ).reset_index().rename(columns={"to_acct":"acct"})

#     df_agg = send.merge(recv, on="acct", how="outer").fillna(0)

#     # 額外 features: 
#     # 自轉帳的比例
#     self_ratio = (
#         base.groupby("from_acct")["is_self_txn"]
#         .apply(lambda x: (x == "Y").mean())
#         .rename("self_txn_ratio")
#         .reset_index()
#         .rename(columns={"from_acct": "acct"})
#     )

#     # 主動發的轉帳，用過幾種幣別 / 幾個通路
#     cur_div = base.groupby("from_acct")["currency_type"].nunique().rename("num_currency").reset_index().rename(columns={"from_acct": "acct"})
#     ch_div = base.groupby("from_acct")["channel_type"].nunique().rename("num_channel").reset_index().rename(columns={"from_acct": "acct"})

#     for extra_feature in (self_ratio, cur_div, ch_div):
#         df_agg = df_agg.merge(extra_feature, on="acct", how="left")
#     df_agg = df_agg.fillna(0)

#     # window

#     print(f"[INFO] Aggregated account table: {len(df_agg):,} rows x {df_agg.shape[1]} cols")

#     return df_agg
import pandas as pd

def aggregate_accounts(df_txn: pd.DataFrame, ref_day: int | None = None, windows=(7, 30, 90)) -> pd.DataFrame:
    """
    將交易層 -> 帳戶層。
    1) 全期間 (<=ref_day) 的 from/to 統計
    2) 對每個 w in windows：再做近 w 天的 from/to 統計
    3) 進階特徵：短期占比、唯一對手數、自轉比/多樣性 (w30)
    """
    if ref_day is None:
        ref_day = int(df_txn["txn_date"].max())

    base = df_txn[df_txn["txn_date"] <= ref_day].copy()

    # ---- 全期間統計 ----
    send = base.groupby("from_acct")["txn_amt"].agg(
        total_send_amt="sum", avg_send_amt="mean",
        max_send_amt="max", min_send_amt="min", send_count="count"
    ).reset_index().rename(columns={"from_acct":"acct"})

    recv = base.groupby("to_acct")["txn_amt"].agg(
        total_recv_amt="sum", avg_recv_amt="mean",
        max_recv_amt="max", min_recv_amt="min", recv_count="count"
    ).reset_index().rename(columns={"to_acct":"acct"})

    df_agg = send.merge(recv, on="acct", how="outer").fillna(0)

    # 自轉比例、多樣性（全期，from 端）
    self_ratio = (
        base.groupby("from_acct")["is_self_txn"].apply(lambda s: (s == "Y").mean())
        .rename("self_txn_ratio").reset_index().rename(columns={"from_acct":"acct"})
    )
    cur_div = base.groupby("from_acct")["currency_type"].nunique().rename("num_currency").reset_index().rename(columns={"from_acct":"acct"})
    ch_div  = base.groupby("from_acct")["channel_type"].nunique().rename("num_channel").reset_index().rename(columns={"from_acct":"acct"})
    for extra in (self_ratio, cur_div, ch_div):
        df_agg = df_agg.merge(extra, on="acct", how="left")
    df_agg = df_agg.fillna(0)

    # ---- 時間窗統計 ----
    for w in windows:
        win = df_txn[(df_txn["txn_date"] >= ref_day - (w - 1)) & (df_txn["txn_date"] <= ref_day)]
        send_w = win.groupby("from_acct")["txn_amt"].agg(
            **{f"send_sum_w{w}":"sum", f"send_mean_w{w}":"mean", f"send_cnt_w{w}":"count"}
        ).reset_index().rename(columns={"from_acct":"acct"})
        recv_w = win.groupby("to_acct")["txn_amt"].agg(
            **{f"recv_sum_w{w}":"sum", f"recv_mean_w{w}":"mean", f"recv_cnt_w{w}":"count"}
        ).reset_index().rename(columns={"to_acct":"acct"})
        df_agg = df_agg.merge(send_w, on="acct", how="left").merge(recv_w, on="acct", how="left").fillna(0)

    # ---- 進階特徵（高性價比）----
    eps = 1e-6
    # 短期占比（近7/30天相對於全期）
    if "send_sum_w7" in df_agg:
        df_agg["send_ratio_w7"] = df_agg["send_sum_w7"] / (df_agg["total_send_amt"] + eps)
        df_agg["recv_ratio_w7"] = df_agg["recv_sum_w7"] / (df_agg["total_recv_amt"] + eps)
    if "send_sum_w30" in df_agg:
        df_agg["send_ratio_w30"] = df_agg["send_sum_w30"] / (df_agg["total_send_amt"] + eps)
        df_agg["recv_ratio_w30"] = df_agg["recv_sum_w30"] / (df_agg["total_recv_amt"] + eps)

    # 唯一對手數（近30天）
    w = 30
    win30 = df_txn[(df_txn["txn_date"] >= ref_day - (w - 1)) & (df_txn["txn_date"] <= ref_day)]
    cp_from = win30.groupby("from_acct")["to_acct"].nunique().rename("uniq_cp_from_w30").reset_index().rename(columns={"from_acct":"acct"})
    cp_to   = win30.groupby("to_acct")["from_acct"].nunique().rename("uniq_cp_to_w30").reset_index().rename(columns={"to_acct":"acct"})
    df_agg = df_agg.merge(cp_from, on="acct", how="left").merge(cp_to, on="acct", how="left").fillna(0)

    # 近30天自轉率與多樣性（from 端）
    self_w30 = (win30.groupby("from_acct")["is_self_txn"].apply(lambda s: (s == "Y").mean())
                .rename("self_txn_ratio_w30").reset_index().rename(columns={"from_acct":"acct"}))
    cur_w30 = win30.groupby("from_acct")["currency_type"].nunique().rename("num_currency_w30").reset_index().rename(columns={"from_acct":"acct"})
    ch_w30  = win30.groupby("from_acct")["channel_type"].nunique().rename("num_channel_w30").reset_index().rename(columns={"from_acct":"acct"})
    for extra in (self_w30, cur_w30, ch_w30):
        df_agg = df_agg.merge(extra, on="acct", how="left")
    df_agg = df_agg.fillna(0)

    # === [新增1] 近30天金額波動度（std） ===
    w = 30
    win30 = df_txn[(df_txn["txn_date"] >= ref_day - (w - 1)) & (df_txn["txn_date"] <= ref_day)]

    send_std30 = win30.groupby("from_acct")["txn_amt"].std(ddof=0).rename("send_std_w30").reset_index().rename(columns={"from_acct":"acct"})
    recv_std30 = win30.groupby("to_acct")["txn_amt"].std(ddof=0).rename("recv_std_w30").reset_index().rename(columns={"to_acct":"acct"})
    df_agg = df_agg.merge(send_std30, on="acct", how="left").merge(recv_std30, on="acct", how="left").fillna(0)

    # === [新增2] 高於「歷史p95/p90/p99」的占比（近30天） ===
    # (a) 先算「歷史（<=ref_day）」各帳戶 from/to 的分位數當門檻
    base = df_txn[df_txn["txn_date"] <= ref_day].copy()
    p95_from = base.groupby("from_acct")["txn_amt"].quantile(0.95).rename("p95_from").reset_index()
    p95_to   = base.groupby("to_acct")["txn_amt"].quantile(0.95).rename("p95_to").reset_index()
    # 另外計算 p90 / p99 門檻
    p90_from = base.groupby("from_acct")["txn_amt"].quantile(0.90).rename("p90_from").reset_index()
    p90_to   = base.groupby("to_acct")["txn_amt"].quantile(0.90).rename("p90_to").reset_index()
    p99_from = base.groupby("from_acct")["txn_amt"].quantile(0.99).rename("p99_from").reset_index()
    p99_to   = base.groupby("to_acct")["txn_amt"].quantile(0.99).rename("p99_to").reset_index()

    # (b) 近30天資料 join 門檻後，計算「大額次數 / 窗口內總次數」
    win30_f = (win30
               .merge(p95_from, left_on="from_acct", right_on="from_acct", how="left")
               .merge(p90_from, left_on="from_acct", right_on="from_acct", how="left")
               .merge(p99_from, left_on="from_acct", right_on="from_acct", how="left"))
    win30_t = (win30
               .merge(p95_to,   left_on="to_acct",   right_on="to_acct",   how="left")
               .merge(p90_to,   left_on="to_acct",   right_on="to_acct",   how="left")
               .merge(p99_to,   left_on="to_acct",   right_on="to_acct",   how="left"))

    # from 端：大於歷史 p95/p90/p99 的比例
    over_from_95 = (win30_f.assign(over=lambda d: (d["txn_amt"] > d["p95_from"]).astype(int))
                           .groupby("from_acct")["over"].agg(["sum","count"]).reset_index())
    over_from_95["hi_amt_ratio_from_w30"] = over_from_95["sum"] / over_from_95["count"].replace(0, 1)
    over_from_95 = over_from_95[["from_acct","hi_amt_ratio_from_w30"]]

    over_from_90 = (win30_f.assign(over=lambda d: (d["txn_amt"] > d["p90_from"]).astype(int))
                           .groupby("from_acct")["over"].agg(["sum","count"]).reset_index())
    over_from_90["hi_amt_ratio_from_w30_p90"] = over_from_90["sum"] / over_from_90["count"].replace(0, 1)
    over_from_90 = over_from_90[["from_acct","hi_amt_ratio_from_w30_p90"]]

    over_from_99 = (win30_f.assign(over=lambda d: (d["txn_amt"] > d["p99_from"]).astype(int))
                           .groupby("from_acct")["over"].agg(["sum","count"]).reset_index())
    over_from_99["hi_amt_ratio_from_w30_p99"] = over_from_99["sum"] / over_from_99["count"].replace(0, 1)
    over_from_99 = over_from_99[["from_acct","hi_amt_ratio_from_w30_p99"]]

    over_from = (over_from_95
                 .merge(over_from_90, on="from_acct", how="outer")
                 .merge(over_from_99, on="from_acct", how="outer")
                 .rename(columns={"from_acct":"acct"}))

    # to 端：大於歷史 p95/p90/p99 的比例
    over_to_95 = (win30_t.assign(over=lambda d: (d["txn_amt"] > d["p95_to"]).astype(int))
                         .groupby("to_acct")["over"].agg(["sum","count"]).reset_index())
    over_to_95["hi_amt_ratio_to_w30"] = over_to_95["sum"] / over_to_95["count"].replace(0, 1)
    over_to_95 = over_to_95[["to_acct","hi_amt_ratio_to_w30"]]

    over_to_90 = (win30_t.assign(over=lambda d: (d["txn_amt"] > d["p90_to"]).astype(int))
                         .groupby("to_acct")["over"].agg(["sum","count"]).reset_index())
    over_to_90["hi_amt_ratio_to_w30_p90"] = over_to_90["sum"] / over_to_90["count"].replace(0, 1)
    over_to_90 = over_to_90[["to_acct","hi_amt_ratio_to_w30_p90"]]

    over_to_99 = (win30_t.assign(over=lambda d: (d["txn_amt"] > d["p99_to"]).astype(int))
                         .groupby("to_acct")["over"].agg(["sum","count"]).reset_index())
    over_to_99["hi_amt_ratio_to_w30_p99"] = over_to_99["sum"] / over_to_99["count"].replace(0, 1)
    over_to_99 = over_to_99[["to_acct","hi_amt_ratio_to_w30_p99"]]

    over_to = (over_to_95
               .merge(over_to_90, on="to_acct", how="outer")
               .merge(over_to_99, on="to_acct", how="outer")
               .rename(columns={"to_acct":"acct"}))

    df_agg = (df_agg
            .merge(over_from, on="acct", how="left")
            .merge(over_to,   on="acct", how="left")
            .fillna(0))

    print(f"[INFO] Aggregated account table: {len(df_agg):,} rows x {df_agg.shape[1]} cols")
    return df_agg
