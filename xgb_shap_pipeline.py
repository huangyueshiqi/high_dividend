import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def select_features_by_shap_coverage(importance_df, coverage=0.8):
    importance_df = importance_df.copy()
    total = float(importance_df['shap_importance'].sum())
    if total <= 0:
        importance_df['importance_ratio'] = 0.0
        importance_df['cumulative_ratio'] = 0.0
        return importance_df.iloc[:0].copy(), importance_df
    importance_df['importance_ratio'] = importance_df['shap_importance'] / total
    importance_df['cumulative_ratio'] = importance_df['importance_ratio'].cumsum()
    cut_idx = int(importance_df['cumulative_ratio'].ge(coverage).idxmax())
    selected = importance_df.iloc[: cut_idx + 1].copy()
    return selected, importance_df

def generate_mock_data(n_samples=5000, n_features=1000):
    """
    构造包含 [基金, 股票, 日期] + 1000列因子特征 + [标签] 的模拟数据
    """
    print(f"正在生成 {n_samples} 条模拟数据，包含 {n_features} 个特征...")
    np.random.seed(42)
    
    # 1. 基础维度列：基金、股票、日期
    funds = [f'Fund_{i:02d}' for i in range(10)]
    stocks = [f'Stock_{i:04d}' for i in range(500)]
    dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
    
    base_data = {
        'fund': np.random.choice(funds, n_samples),
        'stock': np.random.choice(stocks, n_samples),
        'date': np.random.choice(dates, n_samples),
        'label': np.random.randint(0, 2, n_samples)  # 二分类标签，例如：是否进入基金重仓池
    }
    df = pd.DataFrame(base_data)
    
    # 2. 生成1000维特征列
    feature_cols = [f'factor_{i}' for i in range(1, n_features + 1)]
    # 使用正态分布生成特征数据
    features_data = np.random.randn(n_samples, n_features)
    
    # 为了让模型有东西可学，人为让前10个特征和label存在一定的线性关系
    for i in range(10):
        features_data[:, i] += df['label'].values * np.random.uniform(0.5, 1.5)
        
    features_df = pd.DataFrame(features_data, columns=feature_cols)
    
    # 合并基础维度和特征
    df = pd.concat([df, features_df], axis=1)
    
    # 按照日期排序（模拟真实时间序列）
    df = df.sort_values(by='date').reset_index(drop=True)
    
    return df, feature_cols

def train_and_explain(df, feature_cols, target_col='label', train_full=False, shap_coverage=0.8):
    """
    训练 XGBoost 模型，并使用 SHAP 进行特征归因分析
    """
    if train_full:
        X_train, y_train = df[feature_cols], df[target_col]
        X_eval, y_eval = None, None
        print(f"\n使用全量数据训练: {X_train.shape}")
    else:
        print("\n划分训练集和测试集 (按时间序列前80%作为训练集)...")
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_eval, y_eval = test_df[feature_cols], test_df[target_col]
        print(f"训练集大小: {X_train.shape}, 测试集大小: {X_eval.shape}")
    
    # 2. 模型训练
    print("\n开始训练 XGBoost 模型...")
    # XGBoost 参数设置 (二分类任务)
    model = xgb.XGBClassifier(
        n_estimators=100,       # 树的数量
        max_depth=5,            # 树的最大深度
        learning_rate=0.05,     # 学习率
        subsample=0.8,          # 样本采样率，防止过拟合
        colsample_bytree=0.8,   # 特征采样率
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'   # 评估指标
    )
    
    fit_kwargs = {'verbose': False}
    if X_eval is not None and y_eval is not None:
        fit_kwargs['eval_set'] = [(X_eval, y_eval)]
    model.fit(X_train, y_train, **fit_kwargs)
    
    # 3. 模型评估
    if X_eval is not None and y_eval is not None:
        preds = model.predict(X_eval)
        preds_proba = model.predict_proba(X_eval)[:, 1]
        print("\n=== 模型在测试集上的表现 ===")
        print(classification_report(y_eval, preds))
        print(f"ROC-AUC Score: {roc_auc_score(y_eval, preds_proba):.4f}")
        print(f"Accuracy Score: {accuracy_score(y_eval, preds):.4f}")
        print(f"Recall Score (Positive class): {recall_score(y_eval, preds):.4f}")
    else:
        preds = model.predict(X_train)
        preds_proba = model.predict_proba(X_train)[:, 1]
        print("\n=== 模型在全量训练集上的表现 ===")
        print(classification_report(y_train, preds))
        print(f"ROC-AUC Score: {roc_auc_score(y_train, preds_proba):.4f}")
        print(f"Accuracy Score: {accuracy_score(y_train, preds):.4f}")
        print(f"Recall Score (Positive class): {recall_score(y_train, preds):.4f}")
    
    # 4. SHAP 分析与特征重要性提取
    print("\n开始计算 SHAP 值 (可能需要几秒钟)...")
    # 对于树模型，推荐使用 TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    X_shap = X_train if train_full else X_eval
    try:
        shap_values = explainer(X_shap, check_additivity=False)
    except TypeError:
        shap_values = explainer(X_shap)
    
    print("提取特征重要性排名...")
    # 计算每个特征在所有样本上的平均绝对 SHAP 值
    # shap_values.values 包含每个样本每个特征的 SHAP 值
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'shap_importance': mean_abs_shap
    })
    
    # 降序排列
    importance_df = importance_df.sort_values(by='shap_importance', ascending=False).reset_index(drop=True)
    
    print("\n=== Top 10 最重要的特征 ===")
    print(importance_df.head(10))
    
    # 5. 保存特征重要性到 CSV
    output_csv = "feature_shap_importance.csv"
    importance_df.to_csv(output_csv, index=False)
    print(f"\n特征重要性已保存至: {output_csv}")

    selected_df, importance_with_ratio = select_features_by_shap_coverage(importance_df, coverage=shap_coverage)
    selected_path = "selected_features_80pct.csv"
    selected_df.to_csv(selected_path, index=False)
    print(f"累计重要性覆盖 {int(shap_coverage * 100)}% 的特征已保存至: {selected_path} (共 {len(selected_df)} 列)")
    
    # 6. 绘制并保存 SHAP 摘要图 (Summary Plot)
    print("生成并保存 SHAP 摘要图...")
    plt.figure(figsize=(10, 8))
    # plot_type="dot" 是标准的蜜蜂图，显示正负影响
    shap.summary_plot(shap_values, X_shap, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=300)
    plt.close()
    print("SHAP摘要图已保存至: shap_summary_plot.png")

if __name__ == "__main__":
    # 1. 生成模拟数据 (包含 基金、股票、日期 + 1000列因子)
    df, features = generate_mock_data(n_samples=5000, n_features=1000)
    
    # 查看生成的数据结构
    print("\n数据预览 (前5行):")
    # 打印基础维度和前5个特征
    cols_to_show = ['fund', 'stock', 'date', 'label'] + features[:5]
    print(df[cols_to_show].head())
    
    # 2. 训练并进行 SHAP 归因
    train_and_explain(df, features, target_col='label', train_full=True, shap_coverage=0.8)
