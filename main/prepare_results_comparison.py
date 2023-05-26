# Databricks notebook source
import numpy as np
import pandas as pd
from utils.load_data import load_and_clean_data

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data

# COMMAND ----------

df = pd.read_csv('../input/HTL_input_0519.csv')

#删除多于列
drop_cols = []
for col in df.columns:
  if col[:7]=='Unnamed':
    drop_cols.append(col)
df.drop(columns=drop_cols, inplace=True)

#删除全部为空的行
df = df.dropna(how='all')    

#去掉列名中的空格, 并转换为小写
remove_space_dict = {}
for col in df.columns:
  remove_space_dict[col] = col.replace(' ', '_').lower()
df.rename(columns=remove_space_dict, inplace=True)

#处理缺失列
if 're_qty' not in df.columns:
  print(f're_qty is missing, use re_qty=sku_qty')
  df['re_qty'] = df['sku_quantity']

display(df)

# COMMAND ----------

df.columns

# COMMAND ----------

group_by_cols = ['oracle_batch','dimension_group']
df_agg = df.groupby(group_by_cols).agg(job_number_cnt=('job_number','count'),)

# COMMAND ----------

df[df['Fix_Orientation'].isna()]

# COMMAND ----------


