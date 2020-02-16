import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './cred.json'
import pandas as pd

def write():
    df = pd.read_csv('a.csv')
    print(df)
    table_id = 'ten2.hyper1'
    df.to_gbq(table_id, if_exists='append')

def read():
    sql = """
    SELECT *
    FROM `ten2.hyper1`
    """
    df = pd.read_gbq(sql, dialect='standard')
    print(df)

read()
#write()
