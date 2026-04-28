import pandas as pd
import numpy as np
import os
import re
try:
    import jieba
except ModuleNotFoundError:
    jieba = None
try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
except ModuleNotFoundError:
    KMeans = None
    TfidfVectorizer = None
    normalize = None
try:
    from scipy.sparse import hstack, csr_matrix
except ModuleNotFoundError:
    hstack = None
    csr_matrix = None

def drop_high_nan_columns(df, threshold=0.6, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    cols = [c for c in df.columns if c not in set(exclude_cols)]
    nan_ratio = df[cols].isna().mean()
    drop_cols = nan_ratio[nan_ratio > threshold].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df, drop_cols

def build_factor_data(fin_data, price_data, instruments=None, start_date=None, end_date=None, target_dates=None):
    fin_data = fin_data.copy()
    price_data = price_data.copy()

    fin_data['datetime'] = pd.to_datetime(fin_data['datetime'])
    price_data['datetime'] = pd.to_datetime(price_data['datetime'])

    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        fin_data = fin_data[fin_data['datetime'] >= start_date]
        price_data = price_data[price_data['datetime'] >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        fin_data = fin_data[fin_data['datetime'] <= end_date]
        price_data = price_data[price_data['datetime'] <= end_date]

    if instruments is not None:
        instruments = pd.Index(instruments)
        fin_data = fin_data[fin_data['instrument'].isin(instruments)]
        price_data = price_data[price_data['instrument'].isin(instruments)]

    if target_dates is not None:
        target_dates = pd.to_datetime(pd.Series(target_dates).unique())
        price_data = price_data[price_data['datetime'].isin(target_dates)]

    fin_data['instrument'] = fin_data['instrument'].astype('category')
    price_data['instrument'] = price_data['instrument'].astype('category')

    fin_data = fin_data.sort_values(by=['datetime', 'instrument']).reset_index(drop=True)
    price_data = price_data.sort_values(by=['datetime', 'instrument']).reset_index(drop=True)

    factor_data = pd.merge_asof(
        left=price_data,
        right=fin_data,
        on='datetime',
        by='instrument',
        direction='backward'
    )
    return factor_data

def clean_fund_name(name):
    name = str(name)
    name = re.sub(r'[A-Z]+$', '', name)
    name = re.sub(r'(ETF|LOF|FOF|混合|灵活配置|配置|股票|债券|增强|指数|发起式|证券|投资|基金|回报|精选|优选)', '', name)
    return name

def tokenize_text(text):
    if jieba is None:
        raise ModuleNotFoundError("jieba 未安装")
    return " ".join(jieba.lcut(text))

def cluster_funds_kmeans(fund_holding, n_clusters=40):
    """
    根据基金名称进行 NLP 清洗和 K-Means 聚类，并将结果保存在本地
    """
    if KMeans is None or TfidfVectorizer is None or normalize is None or hstack is None or csr_matrix is None:
        raise ModuleNotFoundError("缺少 sklearn/scipy 依赖，无法运行 KMeans 聚类")
    print("正在进行基金名称 NLP 聚类 (K-Means)...")
    # 提取需要聚类的基金基础信息
    df_fund = fund_holding[['S_INFO_WINDCODE', 'F_INFO_FULLNAME', 'F_INFO_NAME', 'F_INFO_FIRSTINVESTTYPE', 'F_INFO_FIRSTINVESTSTYLE']].copy()
    df_fund = df_fund.drop_duplicates()

    # 1. 文本清洗与分词
    df_fund['CLEAN_NAME'] = df_fund['F_INFO_NAME'].apply(clean_fund_name)
    df_fund['TOKEN_NAME'] = df_fund['CLEAN_NAME'].apply(tokenize_text)

    # 2. 提取文本特征：使用词级别的 TF-IDF
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=300)
    text_sparse = vectorizer.fit_transform(df_fund['TOKEN_NAME'])

    # 3. 提取类别特征：One-Hot 编码
    cat_features = pd.get_dummies(df_fund[['F_INFO_FIRSTINVESTSTYLE']])
    cat_sparse = csr_matrix(cat_features.values)

    # 4. 特征合并与归一化
    X_combined = hstack([cat_sparse * 1, text_sparse])
    X_normalized = normalize(X_combined)

    # 5. 重新执行 K-Means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    df_fund['Cluster_Opt'] = kmeans.fit_predict(X_normalized)

    print("【聚类完成】各个类别的基金数量分布前 5：\n", df_fund['Cluster_Opt'].value_counts().head(5))
    
    import os
    os.makedirs('fund', exist_ok=True)
    df_fund.to_csv("fund/fund_cluster.csv", encoding='utf-8-sig', index=False)
    return df_fund

def cluster_funds_llm(fund_holding, n_clusters=40, output_csv="fund/fund_cluster_llm.csv"):
    df_fund = fund_holding[['S_INFO_WINDCODE', 'F_INFO_NAME', 'F_INFO_FIRSTINVESTTYPE', 'F_INFO_FIRSTINVESTSTYLE']].copy()
    df_fund = df_fund.drop_duplicates()
    df_fund = df_fund.rename(columns={
        'S_INFO_WINDCODE': 'fund_code',
        'F_INFO_NAME': 'fund_name',
        'F_INFO_FIRSTINVESTTYPE': 'invest_type',
        'F_INFO_FIRSTINVESTSTYLE': 'invest_style',
    })
    from llm_fund_cluster import LLMFundClusterer, save_cluster_outputs
    clusterer = LLMFundClusterer(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        timeout=int(os.getenv("OPENAI_TIMEOUT", "240")),
    )
    clustering = clusterer.cluster(df_fund.to_dict(orient="records"), n_clusters=n_clusters)
    save_cluster_outputs(
        df_fund=df_fund,
        clustering=clustering,
        output_csv=output_csv,
        output_definitions_json="fund/fund_cluster_llm_definitions.json",
    )
    out = pd.read_csv(output_csv)
    return out

def build_df_for_xgb(fund_holding4, factor_data, neg_ratio=10, random_state=42):
    rng = np.random.default_rng(random_state)

    pos = fund_holding4[['S_INFO_WINDCODE', 'S_INFO_STOCKWINDCODE', 'F_PRT_ENDDATE']].drop_duplicates().copy()
    pos = pos.rename(columns={
        'S_INFO_WINDCODE': 'fund',
        'S_INFO_STOCKWINDCODE': 'stock',
        'F_PRT_ENDDATE': 'date',
    })
    pos['label'] = 1

    universe = pd.Index(pd.Series(factor_data['instrument'].unique()).dropna().unique())
    neg_rows = []
    for (fund, date), g in pos.groupby(['fund', 'date'], sort=False):
        held = pd.Index(g['stock'].unique())
        candidates = universe.difference(held)
        if candidates.empty:
            continue
        n = int(min(len(candidates), max(1, len(held) * neg_ratio)))
        sampled = rng.choice(candidates.to_numpy(), size=n, replace=False)
        neg_rows.append(pd.DataFrame({'fund': fund, 'stock': sampled, 'date': date, 'label': 0}))

    neg = pd.concat(neg_rows, ignore_index=True) if neg_rows else pd.DataFrame(columns=['fund', 'stock', 'date', 'label'])
    base = pd.concat([pos, neg], ignore_index=True)

    base = base.rename(columns={'stock': 'instrument', 'date': 'datetime'})
    base['datetime'] = pd.to_datetime(base['datetime'])
    base = base.sort_values(by='datetime').reset_index(drop=True)

    factor_data = factor_data.copy()
    factor_data['datetime'] = pd.to_datetime(factor_data['datetime'])
    factor_data = factor_data.sort_values(by='datetime').reset_index(drop=True)

    df = pd.merge_asof(
        left=base,
        right=factor_data,
        on='datetime',
        by='instrument',
        direction='backward'
    )
    df = df.rename(columns={'instrument': 'stock', 'datetime': 'date'})

    exclude_cols = ['fund', 'stock', 'date', 'label']
    df, _ = drop_high_nan_columns(df, threshold=0.6, exclude_cols=exclude_cols)

    feature_cols = [c for c in df.columns if c not in set(exclude_cols)]
    return df, feature_cols

def create_feature_matrix_x(return_intermediate=False, cluster_method="kmeans", n_clusters=40):
    """
    将基金持仓明细作为基础样本池，与高低频因子数据合并，得到最终的特征矩阵 X
    """
    print("1. 读取并过滤基金持仓数据...")
    try:
        fund_holding = pd.read_csv("fund/processed_fund_holdings.csv")
        
        if cluster_method == "llm":
            fund_cluster = cluster_funds_llm(fund_holding, n_clusters=n_clusters)
        else:
            fund_cluster = cluster_funds_kmeans(fund_holding, n_clusters=n_clusters)
        
        fund_cluster3 = fund_cluster[fund_cluster['Cluster_Opt'] == 0]
        fund_holding3 = fund_holding[fund_holding['S_INFO_WINDCODE'].isin(list(fund_cluster3['S_INFO_WINDCODE']))]            
        fund_holding4 = fund_holding3[fund_holding3['ANN_DATE'] >= '2015-01-01']
        
        start_date = fund_holding4['F_PRT_ENDDATE'].min()
        end_date = fund_holding4['F_PRT_ENDDATE'].max()
        stocks = fund_holding4['S_INFO_STOCKWINDCODE'].unique()
        print(f"   提取了 {len(fund_holding4)} 条持仓记录，涉及 {len(stocks)} 只股票，报告期: {start_date} 至 {end_date}")
    except FileNotFoundError:
        print("   [警告] 找不到真实的基金持仓数据，将使用模拟数据代替以演示流程。")
        # 构造模拟的 fund_holding4 以防本地测试报错
        fund_holding4 = pd.DataFrame({
            'S_INFO_WINDCODE': ['000423.OF'] * 3,
            'S_INFO_STOCKWINDCODE': ['000538.SZ', '600048.SH', '600348.SH'],
            'F_PRT_ENDDATE': ['2014-12-31', '2014-12-31', '2014-12-31'],
            'ANN_DATE': ['2015-03-27', '2015-03-27', '2015-03-27'],
            'REPORT_TYPE': ['中/年报'] * 3
        })
        
    # ---------------------------------------------------------
    # 2. 读取高低频因子数据 (这里用随机数据模拟，真实环境中你用 pd.read_csv)
    # ---------------------------------------------------------
    print("\n2. 读取并对齐因子数据 (财务 + 量价)...")
    # 假设这些是你的真实因子表
    # fin_data = pd.read_csv("fin_data.csv")
    # price_data = pd.read_csv("price_data.csv")
    
    # 模拟数据: 确保日期包含持仓记录中的日期 (2014-12-31附近)
    dates = pd.date_range('2014-12-25', '2015-01-05', freq='B')
    price_data = pd.DataFrame({
        'instrument': ['000538.SZ']*len(dates) + ['600048.SH']*len(dates) + ['600348.SH']*len(dates),
        'datetime': list(dates)*3,
        'price_factor_1': np.random.randn(len(dates)*3),
        'price_factor_2': np.random.randn(len(dates)*3)
    })
    
    fin_data = pd.DataFrame({
        'instrument': ['000538.SZ', '600048.SH', '600348.SH'],
        'datetime': ['2014-12-01', '2014-12-01', '2014-12-01'], # 月初发布的财务数据
        'fin_factor_1': np.random.randn(3),
        'fin_factor_2': np.random.randn(3)
    })
    
    instruments = fund_holding4['S_INFO_STOCKWINDCODE'].dropna().unique()
    target_dates = fund_holding4['F_PRT_ENDDATE'].dropna().unique()
    factor_data = build_factor_data(
        fin_data=fin_data,
        price_data=price_data,
        instruments=instruments,
        start_date=fund_holding4['F_PRT_ENDDATE'].min(),
        end_date=fund_holding4['F_PRT_ENDDATE'].max(),
        target_dates=target_dates
    )
    print(f"   合并后的全局因子库形状: {factor_data.shape}")

    # ---------------------------------------------------------
    # 3. 将因子数据拼接到基金持仓记录上，生成最终的 X
    # ---------------------------------------------------------
    print("\n3. 将因子库与基金持仓记录合并，生成特征矩阵 X...")
    
    # 提取需要的列，重命名以匹配因子表的列名
    # 我们用 F_PRT_ENDDATE (报告期截止日) 去匹配当天的最新因子
    base_df = fund_holding4[['S_INFO_WINDCODE', 'S_INFO_STOCKWINDCODE', 'F_PRT_ENDDATE', 'ANN_DATE', 'REPORT_TYPE']].copy()
    base_df = base_df.rename(columns={
        'S_INFO_STOCKWINDCODE': 'instrument',  # 对齐股票列
        'F_PRT_ENDDATE': 'datetime'            # 对齐时间列
    })
    
    # 【强制要求】：转换时间类型，并全局按 datetime 排序！
    base_df['datetime'] = pd.to_datetime(base_df['datetime'])
    base_df = base_df.sort_values(by='datetime').reset_index(drop=True)
    
    # 因为 factor_data 之前合并后可能乱序了，再次确保它是按时间排序的
    factor_data = factor_data.sort_values(by='datetime').reset_index(drop=True)
    
    # 进行最终拼接：为每一条基金持仓记录，匹配截止日当天的最新因子数据
    X = pd.merge_asof(
        left=base_df,
        right=factor_data,
        on='datetime',
        by='instrument',
        direction='backward'  # 寻找截止日及以前最近的因子数据
    )
    
    # 恢复列名，方便后续识别
    X = X.rename(columns={
        'instrument': 'S_INFO_STOCKWINDCODE',
        'datetime': 'F_PRT_ENDDATE'
    })

    X, dropped_cols = drop_high_nan_columns(
        X,
        threshold=0.6,
        exclude_cols=['S_INFO_WINDCODE', 'S_INFO_STOCKWINDCODE', 'F_PRT_ENDDATE', 'ANN_DATE', 'REPORT_TYPE']
    )
    print(f"\n4. 缺失值列过滤: 删除 NaN 占比 > 60% 的列，共删除 {len(dropped_cols)} 列")
    
    print(f"\n✅ 成功！最终特征矩阵 X 形状: {X.shape}")
    print("\n--- 最终 X 的前 3 行预览 ---")
    print(X[['S_INFO_WINDCODE', 'S_INFO_STOCKWINDCODE', 'F_PRT_ENDDATE', 'price_factor_1', 'fin_factor_1']].head(3))
    
    if return_intermediate:
        return X, fund_holding4, factor_data
    return X

if __name__ == "__main__":
    X, fund_holding4, factor_data = create_feature_matrix_x(return_intermediate=True)
    df, feature_cols = build_df_for_xgb(fund_holding4, factor_data, neg_ratio=10)
    print(f"\n5. 生成用于训练的 df 完成，形状: {df.shape}，特征列数: {len(feature_cols)}")
    print(df[['fund', 'stock', 'date', 'label']].head(3))

    try:
        from xgb_shap_pipeline import train_and_explain
        train_and_explain(df, feature_cols, target_col='label')
    except Exception as e:
        print(f"\n[提示] 未执行训练/SHAP（可忽略）。原因: {e}")
