import argparse
from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class ClusteringConfig:
    n_clusters: int = 10
    n_components: int = 50
    use_tfidf: bool = True
    random_state: int = 42
    date_bucket: str = "raw"
    keep_latest_per_fund: bool = False


def _bucket_ann_date(df, bucket: str, keep_latest_per_fund: bool):
    import numpy as np
    import pandas as pd

    df = df.copy()
    df["ANN_DATE"] = pd.to_datetime(df["ANN_DATE"])

    if bucket == "raw":
        bucket_date = df["ANN_DATE"]
    elif bucket == "month":
        bucket_date = df["ANN_DATE"].dt.to_period("M").dt.to_timestamp()
    elif bucket == "quarter":
        bucket_date = df["ANN_DATE"].dt.to_period("Q").dt.to_timestamp()
    elif bucket == "halfyear":
        y = df["ANN_DATE"].dt.year
        m = df["ANN_DATE"].dt.month
        # 1~4月归上一年12-31，5~12月归当年06-30
        bucket_date = pd.to_datetime(
            np.where(m <= 4, (y - 1).astype(str) + "-12-31", y.astype(str) + "-06-30")
        )
    else:
        raise ValueError("date_bucket 必须是 raw/month/quarter/halfyear")

    df["CLUSTER_DATE"] = bucket_date

    # 检查是否存在同一基金在同一 bucket 内有多个不同 ANN_DATE
    duplicate_dates = df.groupby(["S_INFO_WINDCODE", "CLUSTER_DATE"])["ANN_DATE"].nunique()
    duplicates = duplicate_dates[duplicate_dates > 1]
    if not duplicates.empty:
        print(f"\n[Warning] 发现 {len(duplicates)} 个 (基金, 桶) 组合内存在多个不同的 ANN_DATE！")
        print("前几个示例如下：")
        for (f_code, b_date), count in duplicates.head().items():
            dates = df[(df["S_INFO_WINDCODE"] == f_code) & (df["CLUSTER_DATE"] == b_date)]["ANN_DATE"].unique()
            print(f"  基金: {f_code}, 桶: {b_date.strftime('%Y-%m-%d')}, 包含的原始公告日: {[d.strftime('%Y-%m-%d') for d in dates]}")
        print()
    else:
        print("\n[Info] 检查完毕：没有发现任何同一基金在同一桶内有多个不同 ANN_DATE 的情况。\n")

    if keep_latest_per_fund:
        latest = (
            df.groupby(["S_INFO_WINDCODE", "CLUSTER_DATE"], sort=False)["ANN_DATE"]
            .max()
            .rename("LATEST_ANN_DATE")
            .reset_index()
        )
        df = df.merge(latest, on=["S_INFO_WINDCODE", "CLUSTER_DATE"], how="inner")
        df = df[df["ANN_DATE"] == df["LATEST_ANN_DATE"]].drop(columns=["LATEST_ANN_DATE"])

    df = df.drop(columns=["ANN_DATE"]).rename(columns={"CLUSTER_DATE": "ANN_DATE"})
    return df


def cluster_funds_by_holdings(df_final: "pd.DataFrame", config: ClusteringConfig = ClusteringConfig()):
    try:
        import numpy as np
        import pandas as pd
        from scipy.sparse import coo_matrix
        from sklearn.cluster import KMeans
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfTransformer
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "依赖缺失：需要 numpy/pandas/scipy/scikit-learn 才能运行聚类。"
        ) from e

    required = {"S_INFO_WINDCODE", "S_INFO_STOCKWINDCODE", "ANN_DATE"}
    missing = required - set(df_final.columns)
    if missing:
        raise ValueError(f"df_final 缺少列: {sorted(missing)}")

    df = df_final[list(required)].dropna().copy()
    df = df.drop_duplicates()
    df = _bucket_ann_date(df, bucket=config.date_bucket, keep_latest_per_fund=config.keep_latest_per_fund)

    labels_all: list[pd.DataFrame] = []
    for ann_date, g in df.groupby("ANN_DATE", sort=True):
        funds = g["S_INFO_WINDCODE"].unique()
        stocks = g["S_INFO_STOCKWINDCODE"].unique()

        if len(funds) < 2 or len(stocks) < 2:
            continue

        fund_to_i = {f: i for i, f in enumerate(funds)}
        stock_to_j = {s: j for j, s in enumerate(stocks)}

        row = g["S_INFO_WINDCODE"].map(fund_to_i).to_numpy()
        col = g["S_INFO_STOCKWINDCODE"].map(stock_to_j).to_numpy()
        data = np.ones(len(g), dtype=np.float32)

        x = coo_matrix((data, (row, col)), shape=(len(funds), len(stocks))).tocsr()
        if config.use_tfidf:
            x = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True).fit_transform(x)

        k = min(config.n_clusters, len(funds))
        if k < 2:
            continue

        comp = min(config.n_components, max(2, k - 1), max(2, x.shape[1] - 1))
        if comp >= 2 and x.shape[1] >= 3:
            x_emb = TruncatedSVD(n_components=comp, random_state=config.random_state).fit_transform(x)
        else:
            x_emb = x.toarray()

        cluster_id = KMeans(n_clusters=k, random_state=config.random_state, n_init=10).fit_predict(x_emb)

        labels_all.append(
            pd.DataFrame(
                {
                    "S_INFO_WINDCODE": funds,
                    "ANN_DATE": ann_date,
                    "cluster_id": cluster_id,
                }
            )
        )

    df_labels_by_date = (
        pd.concat(labels_all, ignore_index=True)
        if labels_all
        else pd.DataFrame(columns=["S_INFO_WINDCODE", "ANN_DATE", "cluster_id"])
    )

    if df_labels_by_date.empty:
        df_fund_main_cluster = pd.DataFrame(
            columns=["S_INFO_WINDCODE", "main_cluster_id", "support", "n_periods"]
        )
        return df_labels_by_date, df_fund_main_cluster

    def mode_1d(x: pd.Series) -> int:
        m = x.mode()
        return int(m.iloc[0])

    df_main = (
        df_labels_by_date.groupby("S_INFO_WINDCODE")["cluster_id"]
        .agg(main_cluster_id=mode_1d, n_periods="size")
        .reset_index()
    )

    df_support = (
        df_labels_by_date.merge(df_main[["S_INFO_WINDCODE", "main_cluster_id"]], on="S_INFO_WINDCODE")
        .assign(hit=lambda d: (d["cluster_id"] == d["main_cluster_id"]).astype(int))
        .groupby("S_INFO_WINDCODE")["hit"]
        .mean()
        .reset_index(name="support")
    )

    df_fund_main_cluster = df_main.merge(df_support, on="S_INFO_WINDCODE")[
        ["S_INFO_WINDCODE", "main_cluster_id", "support", "n_periods"]
    ]

    return df_labels_by_date, df_fund_main_cluster


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="输入CSV路径，至少包含 S_INFO_WINDCODE,S_INFO_STOCKWINDCODE,ANN_DATE")
    p.add_argument("--out-by-date", default="fund_cluster_by_date.csv")
    p.add_argument("--out-main", default="fund_main_cluster.csv")
    p.add_argument("--n-clusters", type=int, default=10)
    p.add_argument("--n-components", type=int, default=50)
    p.add_argument("--no-tfidf", action="store_true")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--date-bucket", default="raw", choices=["raw", "month", "quarter", "halfyear"])
    p.add_argument("--keep-latest-per-fund", action="store_true")
    return p


def main():
    try:
        import pandas as pd
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("依赖缺失：需要 pandas 才能从 CSV 读取输入。") from e

    args = _build_arg_parser().parse_args()
    df_final = pd.read_csv(args.input)

    config = ClusteringConfig(
        n_clusters=args.n_clusters,
        n_components=args.n_components,
        use_tfidf=not args.no_tfidf,
        random_state=args.random_state,
        date_bucket=args.date_bucket,
        keep_latest_per_fund=args.keep_latest_per_fund,
    )
    df_labels_by_date, df_fund_main_cluster = cluster_funds_by_holdings(df_final, config=config)

    df_labels_by_date.to_csv(args.out_by_date, index=False)
    df_fund_main_cluster.to_csv(args.out_main, index=False)


if __name__ == "__main__":
    main()
