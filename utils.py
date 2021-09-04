# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: 'Python 3.9.6 64-bit (''base'': conda)'
#     name: python3
# ---


import pandas as pd
from sqlalchemy import create_engine


class mysqlDatabase:
    def __init__(self, database_username, database_password, database_ip, database_name):
        self.database_username = database_username
        self.database_password = database_password
        self.database_ip       = database_ip
        self.database_name     = database_name
    def get_engine(self):
        sql_engine = create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.format(self.database_username, self.database_password, self.database_ip, self.database_name))
        return sql_engine
    def select_table(self, sql):
        engine = self.get_engine()
        df = pd.read_sql(sql, con = engine)
        print('Successfully select from Bigdata table')
        return df
    def insert_table(self, df, table_name):
        engine = self.get_engine()
        df.astype(str).to_sql(name=table_name, con=engine, if_exists = 'append', index=False)
        print('Successfully insert into Bigdata table: ' + table_name)
        return df
    def upsert_table(self, df, table_name):
        engine = self.get_engine()
        connection = engine.connect()
        # create tmp table 暫存要插入的所有資料
        df.astype(str).to_sql(name='upsert_tmp', con=engine, if_exists = 'replace', index=False)
        try:
            # 刪除會被更新的資料
            sql_safe = '''SET SQL_SAFE_UPDATES=0'''
            sql = '''
            delete from Bigdata.:table_name where exists (select id from Bigdata.upsert_tmp where upsert_tmp.id = :table_name.id)
            '''
            sql = sql.replace(':table_name', table_name)
            connection.execute(sql_safe)
            connection.execute(sql)
            # 插入所有資料完成更新
            df.astype(str).to_sql(name=table_name, con=engine, if_exists = 'append', index=False)
            print('Successfully upsert into Bigdata table: ' + table_name)
        except:
            print('oops, upsert failed!')