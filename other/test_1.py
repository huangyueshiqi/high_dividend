import cx_Oracle
import pandas as pd

if __name__ == "__main__":
    Balance_data=pd.read_csv('../data/AShareBalanceSheet_data.csv',sep='\t')
    CashFlow_data = pd.read_csv('../data/AShareCashFlow_data.csv', sep='\t')
    EarningEst_data = pd.read_csv('../data/AShareEarningEst_data.csv', sep='\t')
    EXRightDividendRecord_data = pd.read_csv('../data/AShareEXRightDividendRecord_data.csv', sep='\t')
    print(Balance_data.head())
    print(CashFlow_data.head())
    print(EarningEst_data.head())
    print(EXRightDividendRecord_data.head())
    print('hello')