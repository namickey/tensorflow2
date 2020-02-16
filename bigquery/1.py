import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './cred.json'
import pandas as pd

def write():
    df = pd.DataFrame({
        'id' : [8,9],
        'dropout' : [8, 8],
        'loss' : [1, 2]
    })
    df.to_gbq('ten2.hyper1', if_exists='replace') # append or replace

def writeCsv():
    df = pd.read_csv('a.csv')
    print(df)
    df.to_gbq('ten2.hyper1', if_exists='append')

def read():
    sql = """
    SELECT *
    FROM `ten2.hyper1`
    """
    df = pd.read_gbq(sql, dialect='standard')
    print(df)

write()
#writeCsv()
read()
