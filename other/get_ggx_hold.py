import cx_Oracle
import pandas as pd

# 获取高股息的基金代码
def get_fund_data():
    connection = cx_Oracle.connect('wind', 'wind', '10.6.60.114:1521/wind')
    cursor = connection.cursor()
    sql = "select S_INFO_OUTERCODE from ChinaMutualFundDescription where F_INFO_FULLNAME like '%股息%'"
    print('从数据库中获取高股息基金数据: ' + sql)
    cursor.execute(sql)
    columns = [col[0] for col in cursor.description]
    results = cursor.fetchall()
    df = pd.DataFrame(results, columns=columns)
    print(df)
    cursor.close()
    connection.close()
    return df

# 获取高股息的基金的持股信息
def get_fund_detail_data(fund_name):
    connection = cx_Oracle.connect('wind', 'wind', '10.6.60.114:1521/wind')
    cursor = connection.cursor()
    sql = "select F_PRT_ENDDATE, S_INFO_STOCKWINDCODE, F_PRT_STKVALUETONAV from ChinaMutualFundStockPortfolio where S_INFO_OUTERCODE = '" + fund_name + "'and REPORT_TYPE = '中/年报' order by F_PRT_ENDDATE desc"
    print('从数据库中获取基金持股数据: ' + sql)
    cursor.execute(sql)
    columns = [col[0] for col in cursor.description]
    results = cursor.fetchall()
    df = pd.DataFrame(results, columns=columns)
    print(df)
    cursor.close()
    connection.close()
    return df

# 对股票池进行行业/市值的区分
def analy_fund_data(fund_names):
    connection = cx_Oracle.connect('jylh', 'jylh', '10.6.60.114:1521/wind')
    cursor = connection.cursor()
    sql = "select S_INFO_OUTERCODE from ChinaMutualFundDescription where F_INFO_FULLNAME like '%股息%'"
    print('从数据库中获取高股息基金数据: ' + sql)
    cursor.execute(sql)
    columns = [col[0] for col in cursor.description]
    results = cursor.fetchall()
    df = pd.DataFrame(results, columns=columns)
    print(df)
    cursor.close()
    connection.close()
    return df
    pass

if __name__ == "__main__":
    fund_names = get_fund_data()
    # for fund_name in fund_names['S_INFO_OUTERCODE']:
    #     get_fund_detail_data(fund_name)
