# 高股息股票分析工具

## 项目简介

本项目是一个用于分析和预测高股息股票的工具集，主要通过分析基金持仓数据、市场因子和股票特征，构建股票池并进行预测模型训练。项目使用机器学习方法分析高股息股票的特征和表现，旨在为投资决策提供数据支持。

## 功能特点

- 从数据库中获取高股息基金的持仓信息
- 提取和处理股票特征数据
- 构建机器学习模型预测高股息股票
- 时间序列分析和数据可视化
- 基于SHAP值分析股票特征重要性
- 生成投资组合推荐

## 文件结构说明

- **数据处理与分析文件**
  - `process.py`: 数据预处理基础功能工具
  - `get_ggx_hold.py`: 获取高股息基金的持仓数据
  - `High_dividend_eda.py`: 高股息股票的探索性数据分析
  - `util.ipynb`: 工具函数集合（Jupyter Notebook格式）

- **模型文件**
  - `High_dividend_model.py`: 高股息预测模型的实现
  - `stockpool_predict.py`: 股票池预测功能实现

- **数据分析笔记本**
  - `high_dividend_analyze.ipynb`: 高股息分析过程笔记本

- **数据文件**
  - `high_dividend_test.csv`: 高股息测试数据
  - `JY9011-投资经理持仓明细导出*.xlsx`: 投资经理持仓明细数据
  - `高股息.xlsx`, `高股息交易流水.xlsx`: 高股息相关数据

## 环境要求

- Python 3.7+
- 依赖库:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - xgboost
  - shap
  - cx_Oracle (用于数据库连接)
  - joblib

## 使用指南

### 数据准备

1. 准备数据库连接（项目使用Oracle数据库）
   ```python
   connection = cx_Oracle.connect('username', 'password', 'host:port/service')
   ```

2. 获取高股息基金数据
   ```bash
   python get_ggx_hold.py
   ```

### 数据分析

1. 运行探索性数据分析
   ```bash
   python High_dividend_eda.py
   ```

2. 也可以通过Jupyter Notebook查看分析过程
   ```bash
   jupyter notebook high_dividend_analyze.ipynb
   ```

### 模型训练与预测

1. 训练高股息预测模型
   ```bash
   python High_dividend_model.py
   ```

2. 预测股票池
   ```bash
   python stockpool_predict.py
   ```

## 数据库表结构

项目中使用了以下主要数据表:
- `ChinaMutualFundDescription`: 基金描述信息
- `ChinaMutualFundStockPortfolio`: 基金持仓信息
- `ZQ_TABLE_FACTOR_BASE`: 基础因子数据
- `ZQ_TABLE_FACTOR`: 因子数据
- `ZQ_TABLE_FACTOR_QAP`: QAP因子数据

## 注意事项

- 本项目需要Oracle数据库连接
- 部分数据路径可能需要根据实际环境进行调整
- 模型训练过程可能需要较长时间，请耐心等待
