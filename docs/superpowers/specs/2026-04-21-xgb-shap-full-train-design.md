## 目标

将 [xgb_shap_pipeline.py](file:///workspace/xgb_shap_pipeline.py) 调整为面向“复原基金持仓”的特征筛选流程：

- 使用全量数据训练一个二分类 XGBoost 模型（不再划分测试集）。
- 在全量样本上计算 SHAP（TreeExplainer），得到每个特征的 mean(|SHAP|) 重要性。
- 输出“累计重要性覆盖 80%”的最小特征集合，供后续复原持仓/选股特征工程使用。

## 输入/输出

### 输入

- DataFrame `df`：包含至少四列 `fund, stock, date, label` 以及若干数值因子特征列。
- `feature_cols`：特征列名列表（仅数值因子列）。

### 输出文件

- `feature_shap_importance.csv`：全部特征的 mean(|SHAP|) 排序结果。
- `selected_features_80pct.csv`：累计重要性达到 80% 的特征子集（含排序与累计占比）。
- `shap_summary_plot.png`：SHAP summary plot（默认 max_display=20）。

## 关键逻辑

1. 训练：使用全量数据 `X_all=df[feature_cols]`、`y_all=df[label]`。
2. SHAP：`shap.TreeExplainer(model)` 并在全量样本上计算；为提速可设置 `check_additivity=False`。
3. 特征选择：对 mean(|SHAP|) 归一化后计算累计占比，取最小集合使累计占比 >= 0.8。

## 约束与注意事项

- `feature_cols` 必须为数值列，避免将 `fund/stock/date` 等非数值列传入模型。
- 对样本量约 8k、特征约 2k 的规模，全量 SHAP 通常可接受；若后续规模扩大再引入抽样开关。

