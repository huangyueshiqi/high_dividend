import cx_Oracle
import pandas as pd
import matplotlib.pyplot
import os

def read_data(file_path,sep='\t'):
    df=pd.read_csv(file_path,sep=sep)

def check_columns_equal(df,col1,col2):
    is_equal=df[col1]==df[col2]
    print(f'比较结果：{is_equal}')
    equal_count=is_equal.sum()
    print(f'两列值相同的行数：{equal_count}')
    # return is_equal,equal_count

def count_nan_columns(df,colname):
    total_count=len(df)
    nan_count=df[colname].isna().sum()
    nan_ratio=nan_count/total_count
    print(f'{colname}的NAN值数量：{nan_count}')
    print(f'{colname}的NAN值比例：{nan_ratio:.2%}')




if __name__ == "__main__":
    pd.set_option('display.max_columns',None)
    folder_path='../data'
    for root,dirs,files in os.walk(folder_path):
        for file in files:
            print(os.path.join(root,file))
    Balance_data=pd.read_csv('../data/AShareBalanceSheet_data.csv',sep='\t')
    CashFlow_data = pd.read_csv('../data/AShareCashFlow_data.csv', sep='\t')
    EarningEst_data = pd.read_csv('../data/AShareEarningEst_data.csv', sep='\t')
    EXRightDividendRecord_data = pd.read_csv('../data/AShareEXRightDividendRecord_data.csv', sep='\t')
    # print(Balance_data.head())
    # print(CashFlow_data.head())
    # print(EarningEst_data.head())
    # print(EXRightDividendRecord_data.head())

    # print(Balance_data.columns)
    # print(CashFlow_data.columns)
    # print(EarningEst_data.columns)
    # print(EXRightDividendRecord_data.columns)

    # print(Balance_data.info())
    # print(CashFlow_data.info())
    # print(EarningEst_data.info())
    # print(EXRightDividendRecord_data.info())

    # for col in EXRightDividendRecord_data.columns:
    #     count_nan_columns(EXRightDividendRecord_data,col)

    # check_columns_equal(EarningEst_data,'S_INFO_WINDCODE','WIND_CODE')
    # print(df_merged.shape)

    #去重
    Balance_data1=Balance_data.drop_duplicates()
    CashFlow_data1 = CashFlow_data.drop_duplicates()

    #去除NAN值
    CashFlow_data2=CashFlow_data1.dropna()

    # print(Balance_data1.describe())
    # print(CashFlow_data1.describe())
    # print(EarningEst_data.describe())
    print(EXRightDividendRecord_data.describe())

    # print(Balance_data1.groupby('WIND_CODE')['TOT_SHR'].describe())

    print('hello')