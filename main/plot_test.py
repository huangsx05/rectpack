# Databricks notebook source
import numpy as np
import pandas as pd
from utils.plot import *

# COMMAND ----------

df = pd.read_csv('../input/manual_batch.csv') ######
display(df)

# COMMAND ----------

agg_dict = {'quantity':'sum', 'ups':'sum'}
df_agg = df.groupby('oracle_batch').agg(agg_dict).reset_index()
display(df_agg)

# COMMAND ----------

colors = list(mcolors.TABLEAU_COLORS.values())

# COMMAND ----------

#H880282
plot_rectangle(x=0,y=0,w=678,h=528,color='black')
plot_n_rectangle_full_sheet_height(x=0,y=0,w=39,h=64,w_sheet=678,h_sheet=528,n_rec=112,color='orange')
plot_n_rectangle_full_sheet_height(x=546,y=0,w=39,h=68.5,w_sheet=678,h_sheet=528,n_rec=21,color='orange')

# COMMAND ----------

#H880099
plot_rectangle(x=0,y=0,w=678,h=528,color='black')
plot_n_rectangle_full_sheet_height(x=0,y=0,w=44,h=65,w_sheet=678,h_sheet=528,n_rec=112,color='orange')
plot_rectangle(x=616,y=0,w=30,h=528,color='black')
plot_n_rectangle_full_sheet_height(x=646,y=0,w=29,h=40.4,w_sheet=678,h_sheet=528,n_rec=13,color='blue')

# COMMAND ----------

#H880131
plot_rectangle(x=0,y=0,w=678,h=528,color=colors[0])
plot_n_rectangle_full_sheet_height(x=0,y=0,w=30,h=29,w_sheet=678,h_sheet=528,n_rec=252,color=colors[1])
plot_rectangle(x=420,y=0,w=30,h=528,color='black')
plot_n_rectangle_full_sheet_height(x=450,y=0,w=28,h=41,w_sheet=678,h_sheet=528,n_rec=96,color=colors[3])

# COMMAND ----------

#H880132
plot_rectangle(x=0,y=0,w=678,h=528,color='black')
plot_n_rectangle_full_sheet_height(x=0,y=0,w=28,h=36,w_sheet=678,h_sheet=528,n_rec=112,color='orange')
plot_rectangle(x=224,y=0,w=30,h=528,color='black')
plot_n_rectangle_full_sheet_height(x=254,y=0,w=28,h=36,w_sheet=678,h_sheet=528,n_rec=84,color='green')
plot_rectangle(x=422,y=0,w=30,h=528,color='black')
plot_n_rectangle_full_sheet_height(x=452,y=0,w=39,h=45,w_sheet=678,h_sheet=528,n_rec=11,color='blue')
plot_n_rectangle_full_sheet_height(x=491,y=0,w=29,h=46,w_sheet=678,h_sheet=528,n_rec=66,color='blue')

# COMMAND ----------

#H880166
plot_rectangle(x=0,y=0,w=582,h=482,color='black')
plot_n_rectangle_full_sheet_height(x=0,y=0,w=97,h=39,w_sheet=582,h_sheet=482,n_rec=72,color='orange')

# COMMAND ----------

582/97, 482/39

# COMMAND ----------


