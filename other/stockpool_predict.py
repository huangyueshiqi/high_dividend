import os.path

import cx_Oracle
import matplotlib
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from joblib import load

from matplotlib import font_manager

font_path = '/usr/share/fonts/cjkuni-uming/uming.ttc'
matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
matplotlib.rcParams['axes.unicode_minus'] = False


# 加载特征标签文件，并筛选出label为1的股票信息
def load_and_filter_data(start_date, end_date):
    # 加载time_simple_eda.ipynb处理好的特征标签文件time_series_eda_202411.csv  2022-09---2024-11
    all_data = []  # 存储所有月份的数据
    # 生成从202209到202411的所有有效月份
    months = pd.date_range(start=start_date, end=end_date, freq='MS').strftime("%Y%m").tolist()
    for month in months:
        file_path = f'data/fund/eda/{fund_name[:6]}_eda_{month}.csv'
        if os.path.exists(file_path):  # 检查文件是否存在
            df = pd.read_csv(file_path)
            df['month'] = month
            all_data.append(df)
        else:
            print(f'{month}不存在')

    data = pd.concat(all_data, ignore_index=True)
    # 筛选出label为1的高股息股票信息
    y_label = data[['CJJG_CODE', 'label', 'month']]
    y_label = y_label[y_label['label'] == 1]

    return y_label


# 清理和填充缺失值
def clean_and_fill_nan(dataframes, threshold, fill_value):
    cleaned_dataframes = []
    for df in dataframes:
        # 删除超过80%行全为NaN的行
        df = df[df.isnull().mean(axis=1) <= threshold]

        # 找出超过80%列全为NaN的列
        deal_col1 = df.isnull().mean(axis=0) > threshold
        columns_to_fill = deal_col1[deal_col1].index.tolist()

        if columns_to_fill:
            for col in columns_to_fill:
                df[col] = df[col].fillna(fill_value)
                print(f'{target_month}的月份特征数据列{col}存在超过80%的缺失')
        cleaned_dataframes.append(df)
        print(df.shape)
    return cleaned_dataframes


# 根据指定时间范围，从数据库中读取因子特征信息
def read_features(start_date_query, end_date_query):
    connection = cx_Oracle.connect('jylh', 'jylh', '10.6.60.114:1521/wind')
    cursor = connection.cursor()
    sql = "select * from ZQ_TABLE_FACTOR_BASE where CJJG_DT between :start_date_query AND :end_date_query"
    cursor.execute(sql, start_date_query=start_date_query.strftime('%Y-%m-%d'),
                   end_date_query=end_date_query.strftime('%Y-%m-%d'))
    columns = [col[0] for col in cursor.description]
    results = cursor.fetchall()
    X_df1 = pd.DataFrame(results, columns=columns)

    # ZQ_TABLE_FACTOR表读取。选取离高股息最近时间。
    connection = cx_Oracle.connect('jylh', 'jylh', '10.6.60.114:1521/wind')
    cursor = connection.cursor()
    sql = "select * from ZQ_TABLE_FACTOR where CJJG_DT between :start_date_query AND :end_date_query"
    cursor.execute(sql, start_date_query=start_date_query.strftime('%Y-%m-%d'),
                   end_date_query=end_date_query.strftime('%Y-%m-%d'))
    columns = [col[0] for col in cursor.description]
    results = cursor.fetchall()
    X_df2 = pd.DataFrame(results, columns=columns)

    # ZQ_TABLE_FACTOR_QAP表读取。选取离高股息最近时间。
    connection = cx_Oracle.connect('jylh', 'jylh', '10.6.60.114:1521/wind')
    cursor = connection.cursor()
    sql = "select * from ZQ_TABLE_FACTOR_QAP where CJJG_DT between :start_date_query AND :end_date_query"
    cursor.execute(sql, start_date_query=start_date_query.strftime('%Y-%m-%d'),
                   end_date_query=end_date_query.strftime('%Y-%m-%d'))
    columns = [col[0] for col in cursor.description]
    results = cursor.fetchall()
    X_df3 = pd.DataFrame(results, columns=columns)

    print(X_df1.shape)
    print(X_df2.shape)
    print(X_df3.shape)

    dataframes = []
    dataframes.append(X_df1)
    dataframes.append(X_df2)
    dataframes.append(X_df3)
    return dataframes


# 2022-09---2024-11，筛选模型预测为1但真实标签为0的假阳性，并合并股票代码和股票名称
# 其它时间段没有真实标签，直接获取预测为1的结果
def filter_false_positives(result_df, y_pred, stockcode_name, target_month):
    # 重置索引并复制数据
    result_df1 = result_df.reset_index()
    result_copy = result_df1.copy()
    # 添加预测结果列
    result_copy['y_pred'] = y_pred

    # 获取预测为1的结果
    false_pos = result_copy[result_copy['y_pred'] == 1]

    # # 2022-09---2024-11 在这个时间段的训练数据中有真实标签为1的股票，需要剔除这些股票的预测结果
    # target_month1 = target_month.replace('-', '')
    # if 201712 <= int(target_month1) <= 202406:
    #     y_label_month = y_label[y_label['month'] == target_month1]
    #     false_pos = false_pos[~false_pos['CJJG_CODE'].isin(y_label_month['CJJG_CODE'])]

    # 合并获取模型预测的股票代码和股票名字
    false_pos_df1 = pd.merge(false_pos['CJJG_CODE'], stockcode_name, left_on='CJJG_CODE', right_on='STOCKCODE',
                             how='inner')
    false_pos_df1 = false_pos_df1[['STOCKCODE', 'SESNAME']].drop_duplicates()

    return false_pos_df1


# 根据目标月份加载模型、数据并预测股票池，输出结果为csv文件
def process_and_predict(target_month):
    # 加载时间段的投票模型
    model = load(f'data/fund/model/{fund_name}/voting_{fund_name}_model.pkl')

    # 加载真实label及对应月份
    # y_label = load_and_filter_data(start_date='2017-12', end_date='2024-06')

    # 加载股票名字
    stockcode_name = pd.read_csv('data/stockcode_name.csv')

    # 加载2016年到现在的月份特征数据
    # 目标月份转换为起始日期和结束日期
    start_date = pd.to_datetime(f'{target_month}-01')
    end_date = (start_date + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
    start_date_query = (start_date - pd.DateOffset(months=1))
    end_date_query = (end_date - pd.Timedelta(days=10))

    # 读取数据
    dataframes = read_features(start_date_query, end_date_query)

    # 预处理数据
    cleaned_dataframes = clean_and_fill_nan(dataframes, threshold=0.8, fill_value=0)
    X_df1, X_df2, X_df3 = cleaned_dataframes

    # 当前月份特征数据
    X_feature_merged = pd.merge(X_df1, X_df2, on='CJJG_CODE', how='outer')
    X_feature_merged = pd.merge(X_feature_merged, X_df3, on='CJJG_CODE', how='outer')

    # 特征与标签,股票代码列分离
    X = X_feature_merged.drop(columns=['CJJG_DT', 'CJJG_DT_x', 'CJJG_DT_y', 'CJJG_CODE'])
    # 删除共线性高的特征
    drop_columns = ['CJJG_DP', 'CJJG_EBIT2EV', 'CJJG_CP_YOY', 'CJJG_PORE', 'CJJG_EBITDA2EV', 'CJJG_NOPLAT2EV',
                    'CJJG_SALES2EV', 'CJJG_TR_10', 'CJJG_TR_20', 'CJJG_MBPM', 'CJJG_POE', 'CJJG_DIO', 'CJJG_DSO',
                    'CJJG_ILQ', 'CJJG_ILQ_10', 'CJJG_ILQ_20', 'CJJG_KU_60']
    X = X.drop(columns=drop_columns)
    y_code = X_feature_merged['CJJG_CODE']

    # 对未来月份数据进行预测
    y_pred = model.predict(X)

    # 筛选预测为1的股票
    # false_pos_df1 = filter_false_positives(y_code, y_pred, y_label,stockcode_name, target_month)
    false_pos_df1 = filter_false_positives(y_code, y_pred, stockcode_name, target_month)
    print(f'{target_month}的股票池列表已经成功输出，总共{len(false_pos_df1)}个股票')

    # 存储结果
    base_dir = f'data/month/{fund_name[:6]}'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    false_pos_df1.to_csv(f'data/month/{fund_name[:6]}/model_predict{target_month}.csv', index=False)


if __name__ == "__main__":
    # 需要修改训练数据中label为1对应的月份
    # 单个月份
    # target_month='2024-01'
    # process_and_predict(target_month)

    # 生成从201601到202411的所有有效月份
    months = pd.date_range(start='2016-01', end='2024-11', freq='MS').strftime("%Y-%m").tolist()
    fund_name = '008177.OF'
    for target_month in months:
        print(target_month)
        process_and_predict(target_month)
