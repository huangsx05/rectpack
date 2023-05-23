# Databricks notebook source
# MAGIC %md
# MAGIC v2:  
# MAGIC 去掉filter  
# MAGIC update obj为base on batch_the_pds  

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from utils.load_data import load_config, load_and_clean_data, initialize_dg_level_results, initialize_sku_level_results
from utils.plot import plot_ups_ocod
from utils.postprocess import prepare_dg_level_results, prepare_sku_level_results
from model.ocod_solver import filter_id_by_criteria_ocod, sheet_size_selection_ocod, check_criteria_ocod, allocate_sku_ocod

# COMMAND ----------

start_time = datetime.now()
print(start_time)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 0: Common Initial Session

# COMMAND ----------

# UI Inputs
dbutils.widgets.dropdown("Request Type", '1_Batching_Proposal', ['1_Batching_Proposal', '2_Direct-Layout'])
request_type = dbutils.widgets.get("Request Type")

dbutils.widgets.dropdown("Batching Type", '1_OCOD', ['1_OCOD', '2_OCMD', '3_MCMD_Seperater', '4_MCMD_No_Seperater'])
batching_type = dbutils.widgets.get("Batching Type")

dbutils.widgets.text("ABC Plates", "1", "ABC Plates")
n_abc = dbutils.widgets.get("ABC Plates")

dbutils.widgets.text("Films", "678x528, 582x482, 522x328", "Films")
sheet_size_list = dbutils.widgets.get("Films")

n_abc = int(n_abc)
sheet_size_list = sheet_size_list.split(', ')
sheet_size_list = [sorted([int(i.split('x')[0]), int(i.split('x')[1])],reverse=True) for i in sheet_size_list]

params_dict = {
  'batching_type':batching_type,
  'n_abc':n_abc,
  'request_type':request_type,
  'sheet_size_list':sheet_size_list
  }

# for k,v in params_dict.items():
#   print(f'{k}: {v}')

# COMMAND ----------

#algo inputs
config_file = f"../config/config_panyu_htl.yaml"
params_dict = load_config(config_file, params_dict)
for k,v in params_dict.items():
  print(f'{k}: {v}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 1: specific test data and params

# COMMAND ----------

input_file = "../input/HTL_input_0419.csv"
filter_Color_Group = ["CG_24","CG_35"]

# COMMAND ----------

#sample config
criteria = params_dict['criteria']
n_abc = params_dict['n_abc']
OCOD_filter = params_dict['OCOD_filter']
OCOD_criteria_check = params_dict['OCOD_criteria_check']

# COMMAND ----------

df = pd.read_csv(input_file)
df = df[df['Color_Group'].isin(filter_Color_Group)]
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main

# COMMAND ----------

#inputs
#clean intput data
df = load_and_clean_data(df)
display(df)

#aggregation by cg_dg
#以cg_dg_id分组，其实应该以dg_id分组就可以，也就是说dg_id是cg_id的下一级
cols_to_first = ['cg_id', 'dimension_group', 'fix_orientation','overall_label_width', 'overall_label_length']
agg_dict = {'re_qty':'sum'}
for c in cols_to_first:
  agg_dict[c] = 'first'
df_1 = df.groupby(['cg_dg_id']).agg(agg_dict).reset_index()
display(df_1)

# COMMAND ----------

#outputs
#初始化和PPC结果比较的results data - 同时也是Files_to_ESKO的file-1
df_res = initialize_dg_level_results(df)
display(df_res)

#初始化结果文件Files_to_ESKO的file-3
res_file_3 = initialize_sku_level_results(df)
display(res_file_3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2  OCOD Layout
# MAGIC 对于每一个ocod id, 选择最优的sheet_size，并决定在该sheet_size上的layout  
# MAGIC  - 遍历sheet_size_list中的sheet_size, 选择pds*sheet_area最小的sheet_size  
# MAGIC  - for both fix_orientation==0 and fix_orientation==1, 遍历旋转和不旋转两种情况

# COMMAND ----------

#前接口
df_1_2 = df_1.copy()

# COMMAND ----------

#寻找最优sheet_size并计算ups和pds
for cg_dg_rotate_id in sorted(df_1['cg_dg_id'].unique()):
  print(f'------ calculating for {cg_dg_rotate_id} ------')
  df_temp = df_1_2[df_1_2['cg_dg_id']==cg_dg_rotate_id]

  #准备dg level的inputs
  label_size = [df_temp['overall_label_width'].values[0], df_temp['overall_label_length'].values[0]]
  fix_orientation = int(df_temp['fix_orientation'].values[0])
  re_qty = df_temp['re_qty'].values[0]
  print(sheet_size_list, label_size, re_qty, fix_orientation) #print函数的input for check

  #准备sku level的inputs
  cols = ['cg_dg_id','cg_id', 'dimension_group', 'fix_orientation','overall_label_width', 'overall_label_length','sku_id','re_qty']
  df_1_5_temp = df[df['cg_dg_id']==cg_dg_id][cols]

  #主函数
  best_sheet_size, res = sheet_size_selection_ocod(sheet_size_list, label_size, re_qty, fix_orientation, df_1_5_temp) ###--->>>遍历sheet_size
  print(f'res = {res}')
  
  #formulate results
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'sheet_width'] = best_sheet_size[0]
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'sheet_length'] = best_sheet_size[1]
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'cg_dg_layout'] = str(res)
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'ups'] = res['ups']
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'pds'] = res['pds'] 
  
#根据criteria再次确认结果
df_1_2['pds_check'] = df_1_2.apply(lambda x: check_criteria_ocod(x, criteria['one_cg_one_dg']), axis=1)
df_1_2['checkpoint_1.1'] = df_1_2['pds_check']
print('results for all ocod ids')
display(df_1_2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.3 allocate SKU  

# COMMAND ----------

# #get data for this step
# cols = ['cg_dg_id','cg_id', 'dimension_group', 'fix_orientation','overall_label_width', 'overall_label_length','sku_id','re_qty']
# df_1_5 = df[cols]
#按照n_abc*ups分配allocate sku
for cg_dg_id in pass_cg_dg_ids_1_2:
  # df_1_5_temp = df_1_5[df_1_5['cg_dg_id']==cg_dg_id]
  # print(f'------ sku allocation for {cg_dg_id} ------')
  res_1_5 = allocate_sku(dict(zip(df_1_5_temp['sku_id'], df_1_5_temp['re_qty'].astype('int'))), ups_1_2_dict[cg_dg_id]*n_abc) ### --->>>
  print(res_1_5)
  con_1 = df_1_5['cg_dg_id']==cg_dg_id
  for sku_id, sku_res_dict in res_1_5.items():
    con_2 = df_1_5['sku_id']==sku_id
    df_1_5.loc[con_1&con_2, 'sku_ups'] = sku_res_dict['ups']
    df_1_5.loc[con_1&con_2, 'sku_pds'] = sku_res_dict['pds']

df_1_5 = df_1_5.sort_values(['cg_dg_id','sku_pds']).reset_index().drop(columns=['index'])
df_1_5['cum_sum_ups'] = df_1_5.groupby(['cg_dg_id'])['sku_ups'].cumsum()   

#split ABC sheets
sets = ['Set A Ups','Set B Ups','Set C Ups','Set D Ups','Set E Ups','Set F Ups','Set G Ups','Set H Ups']
for cg_dg_id in pass_cg_dg_ids_1_2:
  print(f'------ calculating for {cg_dg_id} ------')
  df_1_5_temp = df_1_5[df_1_5['cg_dg_id']==cg_dg_id].reset_index().drop(columns=['index'])  
  n = 1
  cur_set_index = 0
  sum_pds = 0
  for i in range(len(df_1_5_temp)):
    cur_ups_thres = n*ups_1_2_dict[cg_dg_id]
    sku_id = df_1_5_temp.loc[i,'sku_id']
    set_name = sets[cur_set_index]
    # print(cur_set_index, set_name)      
    # print(df_1_5_temp.loc[i,'cum_sum_ups'],cur_ups_thres)
    if df_1_5_temp.loc[i,'cum_sum_ups']<=cur_ups_thres:
      df_1_5.loc[df_1_5['sku_id']==sku_id, set_name] = df_1_5['sku_ups']
    else:
      sum_pds += df_1_5_temp.loc[i,'sku_pds']
      pre_sku_ups = cur_ups_thres - df_1_5_temp.loc[i-1,'cum_sum_ups']
      df_1_5.loc[df_1_5['sku_id']==sku_id, set_name] = pre_sku_ups #pre sheet
      next_sku_ups = df_1_5_temp.loc[i,'sku_ups'] - pre_sku_ups
      n += 1
      cur_set_index += 1  
      set_name = sets[cur_set_index]
      # print(cur_set_index, set_name)   
      df_1_5.loc[df_1_5['sku_id']==sku_id, set_name] = next_sku_ups #next_sheet       
  # sum_pds += df_1_5_temp['sku_pds'].values[-1]

  for set_name in sets:
    if set_name in df_1_5.columns:
      df_1_5.fillna(0,inplace=True)
      print(f'sum_ups for {set_name} = {np.sum(df_1_5[set_name])}') #确认每个set的ups相等

df_1_5 = df_1_5.sort_values(['cg_dg_id','sku_pds'])

# COMMAND ----------

#后接口
display(df_1_5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4 Prepare Results

# COMMAND ----------

# ups layout
scale = 100
for cg_dg_id in pass_cg_dg_ids_1_2:
  df_temp = df_1_2[df_1_2['cg_dg_id']==cg_dg_id]
  label_width = df_temp['overall_label_width'].values[0]
  label_length = df_temp['overall_label_length'].values[0]
  sheet_width = int(df_temp['sheet_width'].values[0])
  sheet_length = int(df_temp['sheet_length'].values[0])  
  layout_dict = eval(df_temp['cg_dg_layout'].values[0])
  print(f'------ plot ups for {cg_dg_id} ------')  
  print(f'sheet_size = {sheet_width}x{sheet_length}')
  print(layout_dict)  

  plt.figure(figsize=(sheet_width/scale, sheet_length/scale))
  plt.title(f"{cg_dg_id}, 'sheet_size={sheet_width}x{sheet_length}'") 
  plot_ups_ocod([label_width, label_length], [sheet_width, sheet_length], layout_dict) ###### --->>>
  # plt.show() 

# COMMAND ----------

#DG level results - 更新结果和PPC比较的结果df_res，同时也是Files_to_ESKO的file-1
df_res = prepare_dg_level_results(df_res, df_1_2, df_1_5, pass_cg_dg_ids_1_2)
display(df_res)

# COMMAND ----------

#sku level results - 更新结果文件Files_to_ESKO的file-3
res_file_3 = prepare_sku_level_results(res_file_3, df_1_5)
display(res_file_3)

# COMMAND ----------

end_time = datetime.now()
print(start_time)
print(end_time)
print('running time =', (end_time-start_time).seconds, 'seconds')
