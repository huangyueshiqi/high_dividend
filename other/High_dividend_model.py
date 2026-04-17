import itertools
import time
from collections import Counter

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shap
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, \
    recall_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

sns.set_style('whitegrid')
from matplotlib import font_manager

font_path = '/usr/share/fonts/cjkuni-uming/uming.ttc'
matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False

import warnings

warnings.filterwarnings('ignore')


def count_fund_names(dir):
    # 初始化计数器
    fund_name_counter = Counter()
    for filename in os.listdir(dir):
        if filename.endswith('.csv'):
            split_fund_name = filename.split('_eda_')
            fund_name_counter[split_fund_name[0]] += 1
            print(split_fund_name[0])
    return fund_name_counter


# 读取基金名称和月份数据，合并并清理空值，将每行的基金名称和其对应的非空月份分别存入两个列表
def clean_and_split_months(fund_names_path, months_path):
    # 读取数据
    months_df = pd.read_csv(fund_names_path)
    fund_names_df = pd.read_csv(months_path)
    # 合并数据框
    months_with_fund = pd.concat([fund_names_df, months_df], axis=1)
    # 清理空值并将每一行的非空值提取为列表
    result_cleaned = months_with_fund.apply(lambda row: row.dropna().tolist(), axis=1)
    # 分离基金名称和其对应的非空月份
    res = []
    for x in result_cleaned:
        if len(x) > 1:  # 确保至少有基金名称和至少一个月份
            res.append(x)
    fund_names = [sublist[0] for sublist in res]
    months_list = [sublist[1:] for sublist in res]

    return fund_names, months_list


# 模型评估指标函数，打印Recall和F1
def metric_score(test_y, pre_y):
    # 模型评估
    accuracy = accuracy_score(test_y, pre_y)
    auroc = roc_auc_score(test_y, pre_y)
    aupr = average_precision_score(test_y, pre_y)
    f1 = f1_score(test_y, pre_y)
    precision = precision_score(test_y, pre_y)
    recall = recall_score(test_y, pre_y)

    # 分类报告
    report = classification_report(test_y, pre_y)

    # print(report)
    print("Recall:%.3f" % recall)
    print("Precision:%.3f" % precision)
    print("F1:%.3f" % f1)


def plot_confusion_matrix(cm, classes, fund_name, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'{fund_name}_confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.savefig(f'data/fund/fig/24/{fund_name}_confusion_matrix.svg')
    plt.close()
    print(f'{fund_name}的混淆矩阵绘制完成')


# 读取相关股票信息
def read_stock_info(ST_and_NewListings_path, stockcode_name_path, factor_name_path):
    # ST股票和上市不超过半年的股票
    ST_and_NewListings = pd.read_csv(ST_and_NewListings_path)
    print(f'ST和上市不超过半年的股票信息：{ST_and_NewListings}')

    stockcode_name = pd.read_csv(stockcode_name_path)
    print(f'股票名字信息：{stockcode_name}')

    # 加载因子名字文件
    factor_name = pd.read_csv(factor_name_path)
    print(f'因子名字信息：{factor_name}')

    # 统一因子字段为大写
    factor_name['COLUMNAME'] = factor_name['COLUMNAME'].str.upper()
    # 创建映射字典
    mapping_dict = factor_name.set_index('COLUMNAME')['FACTORNAME'].to_dict()
    return ST_and_NewListings, stockcode_name, mapping_dict


# 训练多个模型并存储,以及计算分析SHAP值
def model_train(X, y):
    # 模型存储列表
    models = []

    for i in range(30):
        rus = RandomUnderSampler(sampling_strategy=0.1)  # 比例1：10
        X_train_rus, y_train_rus = rus.fit_resample(X, y)
        xgb_model = XGBClassifier()
        xgb_model.fit(X_train_rus, y_train_rus)

        # 特征重要性（SHAP值）
        explainer = shap.Explainer(xgb_model)
        shape_value = explainer.shap_values(X)
        shap_df = pd.DataFrame(shape_value, columns=X_train_rus.columns)
        # 输出每个特征的SHAP值（绝对值）
        features = []
        shap_values_means = []
        for index, feature in enumerate(X_train_rus.columns):
            shap_values_mean = np.abs(shap_df.values[:, index]).mean()
            features.append(feature)
            shap_values_means.append(shap_values_mean)
        tt = pd.DataFrame(features, columns=['features'])
        tt['shap_values'] = shap_values_means
        tt_sort = tt.sort_values(by='shap_values', ascending=False)

        base_dir = f'data/fund/model/24/{fund_name}'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        tt_sort.to_csv(f'{base_dir}/xgb_{fund_name}_feature_importance{i}.csv', index=False)

        # 保存模型
        model_filename = f'{base_dir}/xgb_{fund_name}_model_{i}.pkl'
        joblib.dump(xgb_model, model_filename)
        models.append(xgb_model)
    # 作为投票模型的训练集
    X_train_rus, y_train_rus = rus.fit_resample(X, y)

    return X_train_rus, y_train_rus


# 训练投票模型
def train_voting_model(X_train_rus, y_train_rus):
    base_dir = f'data/fund/model/24/{fund_name}'
    # 加载所有保存的模型
    models = [joblib.load(f'{base_dir}/xgb_{fund_name}_model_{i}.pkl') for i in range(30)]

    # 对所有模型进行投票集成,选择硬投票或软投票(基于概率）
    voting_clf = VotingClassifier(estimators=[(f'xgb_{fund_name}_model_{i}', model) for i, model in enumerate(models)],
                                  voting='soft')
    voting_clf.fit(X_train_rus, y_train_rus)
    # 保存模型
    model_filename = f'{base_dir}/voting_{fund_name}_model.pkl'
    joblib.dump(voting_clf, model_filename)
    y_pred = voting_clf.predict(X_test)
    metric_score(test_y=y_test, pre_y=y_pred)
    return y_pred


# 查看模型预测为1，实际为0的股票
def show_pred(y_pred):
    # 原始数据处理
    result_df1 = y_code.reset_index()
    result_copy = result_df1.copy()
    result_copy['y_true'] = y_test
    result_copy['y_pred'] = y_pred

    false_pos = result_copy[(result_copy['y_pred'] == 1) & (result_copy['y_true'] == 0)]
    # 合并股票名称
    false_pos_df1 = pd.merge(false_pos['CJJG_CODE'], stockcode_name, left_on='CJJG_CODE', right_on='STOCKCODE',
                             how='inner')
    # 添加重复次数列
    false_pos_df1['count'] = false_pos_df1.groupby('STOCKCODE')['STOCKCODE'].transform('count')
    false_pos_df1 = false_pos_df1[['STOCKCODE', 'SESNAME', 'count']].drop_duplicates().reset_index(drop=True)
    base_dir = f'data/fund/model/24/{fund_name}'
    false_pos_df1.to_csv(f'{base_dir}/voting_{fund_name}_model_predict.csv', index=False)

    # 默认阈值一般为0.5
    y_pred_recall = y_pred[0:] > 0.5
    cnf_matrix = confusion_matrix(y_test, y_pred_recall)
    class_names = [0, 1]
    plot_confusion_matrix(cnf_matrix, classes=class_names, fund_name=fund_name)


# 查看实际为0，模型预测为1的股票名，以及特征重要性分析画图
def analyze_stock_feature():
    # 读取股票名文件
    base_dir = f'data/fund/model/24/{fund_name}'
    voting_result = pd.read_csv(f'{base_dir}/voting_{fund_name}_model_predict.csv')
    print(f'实际为0，模型预测为1的股票名结果{voting_result}')

    # 特征重要性读取
    feature_file_names = [f'{base_dir}/xgb_{fund_name}_feature_importance{i}.csv' for i in range(30)]

    importance_features = []
    for file in feature_file_names:
        df = pd.read_csv(file)
        importance_features.append(df)

    # 横向拼接所有数据
    importance_all = pd.concat(importance_features)

    # 按特征分组，计算特征重要性平均值，并排序
    importance_summary = (
        importance_all.groupby('features')['shap_values'].agg(['mean', 'std']).reset_index().sort_values(by='mean',
                                                                                                         ascending=False))
    print(f'特征重要性结果：{importance_summary}')

    # 替换特征名字，如果找不到映射，使用原名字
    mapping_features = [mapping_dict.get(feature) for feature in importance_summary['features'][:15]]
    print(f'特征重要性特征映射结果：{mapping_features}')

    plt.barh(mapping_features, importance_summary['mean'][:15], xerr=importance_summary['std'][:15], capsize=5)
    plt.xlabel(f'{fund_name}_Importance(Mean)', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'data/fund/fig/24/{fund_name}_Importance(Mean).svg')
    plt.close()
    print(f'{fund_name}的特征重要性图绘制完成')


if __name__ == "__main__":
    # 相关股票数据信息读取与处理
    fund_names_path = 'data/fund/eda/24/months_df.csv'
    months_path = 'data/fund/eda/24/fund_names_df.csv'
    fund_names, months_list = clean_and_split_months(fund_names_path, months_path)

    ST_and_NewListings_path = 'data/ST_and_NewListings.csv'
    stockcode_name_path = 'data/stockcode_name.csv'
    factor_name_path = 'data/factor_name.csv'
    ST_and_NewListings, stockcode_name, mapping_dict = read_stock_info(ST_and_NewListings_path, stockcode_name_path,
                                                                       factor_name_path)
    # 加载time_simple_eda.ipynb处理好的特征标签文件time_series_eda_202411.csv  2022-09---2024-11
    # 加载和合并数据
    start_time = time.time()  # 开始计时
    for i, fund_name in enumerate(fund_names):
        all_data = []  # 存储所有月份的数据
        for month in months_list[i]:
            month = int(month)
            df = pd.read_csv(f'data/fund/eda/24/{fund_name}_eda_{month}.csv')
            df['month'] = month
            all_data.append(df)
        data = pd.concat(all_data, ignore_index=True)
        print(f'{fund_name}的eda数据shape是{data.shape}')
        # 特征与标签,股票代码列分离
        X = data.drop(columns=['CJJG_CODE', 'label', 'month'])
        # 删除共线性高的特征
        drop_columns = ['CJJG_DP', 'CJJG_EBIT2EV', 'CJJG_CP_YOY', 'CJJG_PORE', 'CJJG_EBITDA2EV', 'CJJG_NOPLAT2EV',
                        'CJJG_SALES2EV', 'CJJG_TR_10', 'CJJG_TR_20', 'CJJG_MBPM', 'CJJG_POE', 'CJJG_DIO', 'CJJG_DSO',
                        'CJJG_ILQ', 'CJJG_ILQ_10', 'CJJG_ILQ_20', 'CJJG_KU_60']
        X = X.drop(columns=drop_columns)
        print(f'{fund_name}的X数据shape是{X.shape}')
        y = data['label']
        print(f'{fund_name}的y数据shape是{y.shape}')
        y_code = data[['CJJG_CODE', 'label']]

        # 训练集测试集划分，训练集放入全部正样本，负样本进行欠采样，预测时放入全部样本
        # 保留正样本，欠采样负样本
        rus = RandomUnderSampler(sampling_strategy=0.1)  # 比例1：10
        X_train_rus, y_train_rus = rus.fit_resample(X, y)
        X_test, y_test = X, y

        # 统计当前类别占比情况
        print(f"基金{fund_name}在Before undersampling:{Counter(y)}")

        # 欠采样后类别占比情况
        print(f"基金{fund_name}在After undersampling:{Counter(y_train_rus)}")

        if Counter(y)[1] / Counter(y)[0] < 0.005:
            print(f'基金{fund_name}的正负样本比例太小，只有{Counter(y)[1] / Counter(y)[0]}，参考意义较小，不参与训练')
            continue

        # 训练投票模型
        X_train_rus, y_train_rus = model_train(X, y)

        y_pred = train_voting_model(X_train_rus, y_train_rus)

        # 展示预测结果
        show_pred(y_pred)

        # 分析预测结果并画图
        analyze_stock_feature()
    end_time = time.time()  # 结束计时
    print(f'代码运行时间为{end_time - start_time:.2f}秒')
