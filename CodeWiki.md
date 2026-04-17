# 高股息股票预测算法 Code Wiki

## 1. 项目整体架构

本项目旨在利用机器学习方法，通过分析高股息基金持仓、多因子数据（量价、财务等）、以及宏观/行业数据，对高股息股票进行筛选与预测模型构建。

整体架构主要分为四个核心阶段：
1. **数据采集与接入 (Data Acquisition)**：通过 `cx_Oracle` 连接 Oracle 数据库，获取 Wind 数据中的高股息基金信息、基金持仓明细以及多张因子特征表（如 `ZQ_TABLE_FACTOR_BASE`、`ZQ_TABLE_FACTOR`）。
2. **数据清洗与特征工程 (Data Processing & Feature Engineering)**：清洗缺失值，过滤ST或新上市不超过半年的股票，通过金融数据表（利润表、现金流量表、资产负债表等）及量价数据构建股票的多维特征。结合股票未来收益率（相对中证500/800红利指数）进行好坏样本标签打标。
3. **模型训练与评估 (Model Training & Evaluation)**：针对正负样本不平衡问题，采用 `RandomUnderSampler` 欠采样技术。使用 `XGBoost` 算法训练多个基分类器，最后通过 `VotingClassifier`（软投票机制）进行模型集成预测。采用 `SHAP` 值进行模型解释与特征重要性分析。
4. **股票池预测与输出 (Stock Pool Prediction)**：加载训练好的投票模型，对最新时间截面（目标月份）的股票因子数据进行预测，输出高概率被划分为优质高股息的股票池清单。

---

## 2. 主要模块职责

- **数据获取模块** ([`other/get_ggx_hold.py`](file:///workspace/other/get_ggx_hold.py))：负责从 Oracle 数据库中直接拉取基金名称带有“股息”字样的基金清单，及其定期报告中的重仓股明细。
- **数据处理与探索模块** ([`other/High_dividend_eda.py`](file:///workspace/other/High_dividend_eda.py), [`other/process.py`](file:///workspace/other/process.py), `*.ipynb`):
  - `High_dividend_eda.py`：负责核心的特征预处理，包括从数据库拉取不同维度的因子数据、清洗过滤缺失值超标的特征、拼接股票基础信息，并生成供模型训练使用的 `EDA` 宽表数据集。
  - `process.py`：提供基础的 DataFrame 数据校验、缺失值统计及除重工具函数。
  - `cashflow_*.ipynb`、`balance_quarter.ipynb`、`income_quarter.ipynb`：负责解析与转换财务三大表（利润表、资产负债表、现金流量表）数据，将其季度化和年度化。
- **模型构建与训练模块** ([`other/High_dividend_model.py`](file:///workspace/other/High_dividend_model.py), `high_dividend_v1.ipynb`, `high_dividend_v2.ipynb`):
  - `High_dividend_model.py`：负责读取预处理后的特征集，执行欠采样，训练多个 `XGBoost` 基模型，组合成 `VotingClassifier` 集成模型，并使用 `SHAP` 评估特征重要性，最后输出混淆矩阵与评估指标。
  - `high_dividend_v*.ipynb`：提供了完整的研究链路分析原型和不同版本的算法实验与特征比对。
- **股票池预测模块** ([`other/stockpool_predict.py`](file:///workspace/other/stockpool_predict.py)): 在模型上线后，基于目标月份输入，获取对应时间窗口内的因子特征，经过预处理后交由加载好的投票模型进行推理预测，最后生成预测为1（被判定为优质高股息）的股票清单。

---

## 3. 关键类与函数说明

### 3.1 特征生成 ([`other/High_dividend_eda.py`](file:///workspace/other/High_dividend_eda.py))
- `read_fund_name()` & `read_high_dividend(fund_name)`：连接 Oracle 数据库，获取高股息基金清单及各基金历史的持仓明细。
- `process_stock_data()`：按基金名筛选有效股票（以`.SH`和`.SZ`结尾），获取各股票的最早和最晚交易日期并按月聚合。
- `process_monthly_stocks()`：扩展时间维度，按月份聚合及对齐交易时间范围内的目标股票集。
- `read_features(start_date_query, end_date_query)`：根据起止日期从数据库抽取三大因子特征表（`ZQ_TABLE_FACTOR_BASE`, `ZQ_TABLE_FACTOR`, `ZQ_TABLE_FACTOR_QAP`）。
- `process_data()`：进行特征表与股票标签的合并（Join），并在过程中剔除 ST 股票及上市不超过半年的股票，输出单月的完整样本特征。

### 3.2 模型训练 ([`other/High_dividend_model.py`](file:///workspace/other/High_dividend_model.py))
- `model_train(X, y)`：核心训练函数，循环迭代 30 次，每次使用 `RandomUnderSampler` 对负样本进行欠采样（正负比例1:10），训练 `XGBClassifier`，并使用 `shap.Explainer` 计算 SHAP 值以输出特征重要性，最终保存这 30 个基模型文件。
- `train_voting_model(X_train_rus, y_train_rus)`：加载前面训练好的 30 个 `XGBoost` 模型，通过 `VotingClassifier(voting='soft')` 组装为软投票分类器并进行训练和保存。
- `analyze_stock_feature()`：汇总各基模型输出的 SHAP 值文件，计算特征重要性平均值及方差，并利用 `matplotlib` 绘制 Top 15 特征的水平条形图。
- `metric_score(test_y, pre_y)`：计算分类模型的核心评估指标（Accuracy, AUC, Recall, Precision, F1-Score）并打印分类报告。

### 3.3 股票池预测 ([`other/stockpool_predict.py`](file:///workspace/other/stockpool_predict.py))
- `process_and_predict(target_month)`：核心预测流控制函数。
  1. 加载持久化的集成模型 `voting_{fund_name}_model.pkl`。
  2. 根据 `target_month` 转换出查询特征的日期范围。
  3. 读取数据库因子特征，执行缺失值清理及填充，并删除共线性过高的特征列（如 `CJJG_DP`, `CJJG_EBIT2EV` 等）。
  4. 调用 `model.predict(X)` 输出预测标签。
  5. 筛选出预测标签为 1 的股票，关联中文股票名，将股票池结果输出至 CSV 文件 `model_predict{target_month}.csv`。

---

## 4. 依赖关系

- **Python 版本**: 3.7+
- **核心数据处理**: `pandas`, `numpy`
- **机器学习库**: `scikit-learn` (模型评估、投票集成), `xgboost` (核心预测算法), `imblearn` (不平衡样本处理，下采样)
- **模型可解释性**: `shap` (分析因子特征对预测结果的边际贡献度)
- **数据库连接**: `cx_Oracle` (用于直连 Oracle 数据库，需本地配置好 Oracle Client 环境)
- **模型保存及可视化**: `joblib` (模型序列化), `matplotlib`, `seaborn` (数据可视化)

---

## 5. 项目运行方式

**前置准备：**
1. **安装依赖库**：
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn shap cx_Oracle matplotlib seaborn joblib
   ```
2. **数据库配置**：确保配置好 Oracle 数据库环境与权限（代码中通过硬编码方式连接如 `10.6.60.114:1521/wind`，需确保内网网络畅通及账号密码有效）。
3. **目录结构**：确保项目根目录下存在基础静态数据表（如 `data/stockcode_name.csv`, `data/ST_and_NewListings.csv` 等），并根据代码逻辑建立相应的数据存储目录。

**标准执行流程：**

1. **获取及生成特征数据集**
   执行 `High_dividend_eda.py`，从数据库读取各基金对应的股票因子，处理缺失值与剔除 ST 股票，生成各个月份对应的特征集 `.csv` 文件至 `data/fund/eda/24/`。
   ```bash
   python other/High_dividend_eda.py
   ```

2. **训练模型与特征重要性分析**
   执行 `High_dividend_model.py`，读取上一步生成的特征文件，自动划分样本、进行下采样、训练 XGBoost 基模型以及投票集成模型，生成模型文件 `.pkl` 和特征重要性 SHAP 柱状图。
   ```bash
   python other/High_dividend_model.py
   ```

3. **输出股票池预测结果**
   执行 `stockpool_predict.py`，配置 `target_month`（目标月份），脚本将提取实时因子，调用训练好的投票模型进行推理预测，最终在 `data/month/` 目录下输出高股息推荐股票池清单。
   ```bash
   python other/stockpool_predict.py
   ```

4. **使用 Notebook 进行研究分析 (可选)**
   若需要修改因子算法、特征打标逻辑或做进一步探索性分析，可以通过 Jupyter 启动 Notebooks 脚本进行迭代：
   ```bash
   jupyter notebook high_dividend_v2.ipynb
   ```