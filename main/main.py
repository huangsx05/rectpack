# Databricks notebook source
      # ppc - iter - minA - maxA
# 0419: 416 - 404  - 415  - 411
# 0519: 304 - 276  - 273  - 272
# 0614: 787 - 763  - 771  - 763

# COMMAND ----------

# MAGIC %md
# MAGIC #### main

# COMMAND ----------

pip install sympy

# COMMAND ----------

import json
import pandas as pd
import numpy as np
from datetime import datetime
from utils.tools import allocate_sku
from utils.load_data import convert_jobs_df_to_json, load_user_params, load_config, initialize_input_data, convert_jobs_input_into_df
from model.shared_solver import split_abc_ups

# COMMAND ----------

#ui inputs
user_params_path = "../config/ui_inputs.json"
with open(user_params_path, "r", encoding="utf-8") as f:
  input_params = json.load(f) 
print(input_params)

#job inputs
input_file = '../input/HTL_input_0419.csv' #'../input/HTL_input_0419.csv','../input/HTL_input_0519.csv', 0614
# filter_Color_Group = [] #空代表不筛选，全部计算
filter_Color_Group = ['CG_22', 'CG_23', 'CG_24', 'CG_26', 'CG_27', 'CG_28', 'CG_29', 'CG_30']

# COMMAND ----------

def main(input_params):
  #get user params
  user_params = load_user_params(input_params)  
  batching_type = user_params["batching_type"]
  print(f"batching_type = {batching_type}")

  #get algo config
  config_path = f"../config/config.json"  
  params_dict = load_config(config_path)[batching_type]
  params_dict['user_params'] = user_params
  for k,v in params_dict.items():
    print(f"{k}:")
    for ki, vi in v.items():
      print(f"  {ki}: {vi}")

  #jobs input
  # print(f"[{datetime.now()}]: preparing jobs input")
  input_mode = params_dict['algo_params']['input_mode']
  print(f"input_mode = {input_mode}")
  if input_mode == 'csv':
    df, df_1 = initialize_input_data(input_mode, filter_Color_Group, input_file=input_file) #------ 数据清洗部分可以转移到GPM完成
  elif input_mode == 'json':
    input_params = convert_jobs_df_to_json(input_params, input_file, filter_Color_Group) #JSON IUPUT
    df, df_1 = initialize_input_data(input_mode, filter_Color_Group, jobs_dict_list=input_params['jobInfo'])
  # print(df_1.columns)

  #main
  start_time = datetime.now() 
  print(f"[{datetime.now()}]: ------ start main batching ------")
  #---------------------------------------------------------------------------------------------------------
  if batching_type == '1_OCOD':
    pass
  elif batching_type == '2_OCMD':
    pass
  elif batching_type == '3_MCMD_Seperater':  
    from sub_main.runner_3_mcmd_seperater import runner_3_mcmd_seperator_sku_pds
    best_index, best_batch, best_res = runner_3_mcmd_seperator_sku_pds(start_time, params_dict, df, df_1)
  elif batching_type == '4_MCMD_No_Seperater':
    pass
  #---------------------------------------------------------------------------------------------------------  
  print(f"[{datetime.now()}]: ------ end main batching ------")
  return df, df_1, params_dict, best_index, best_batch, best_res

if __name__ == "__main__":
  df, df_3, params_dict, best_index, best_batch, best_res = main(input_params)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot for the best batch

# COMMAND ----------

import matplotlib.pyplot as plt
from utils.plot import plot_full_height_for_each_dg_with_ink_seperator

# COMMAND ----------

#根据上一环节的结果得到最优batch的batch_name
# best_batch_name = min(res_metric_3_2, key=res_metric_3_2.get)
best_batch_name = best_index
print(f"best_batch_name = '{best_batch_name}'")

# COMMAND ----------

# best_batch = batches_dict[best_batch_name]
# res = res_detail_3_2[best_batch_name]
res = best_res
print(best_batch)
print(res) 

# COMMAND ----------

params_dict

# COMMAND ----------

ink_seperator_width = int(params_dict["user_params"]["ink_seperator_width"])
ink_seperator_width

# COMMAND ----------

#plot上面的最优解
for sub_batch_id in best_batch.keys():
  best_sheet = res[sub_batch_id]['best_sheet']
  best_comb = res[sub_batch_id]['best_comb']
  dg_id = [i[:-2] for i in best_comb.split('<+>')]
  dg_orient = [i[-1] for i in best_comb.split('<+>')]
  cg_id = []
  label_w_list  =[]
  label_h_list = []
  for i in range(len(dg_id)):
    dg = dg_id[i]
    orient = dg_orient[i]
    label_w = df_3.loc[df_3['dg_id']==dg, 'overall_label_width'].values[0]
    label_h = df_3.loc[df_3['dg_id']==dg, 'overall_label_length'].values[0]
    cg_id.append(df_3.loc[df_3['dg_id']==dg, 'cg_id'].values[0])
    if orient=='w':
      label_w_list.append(label_w)
      label_h_list.append(label_h)
    else:
      label_w_list.append(label_h)
      label_h_list.append(label_w)        
  ups_list = list(res[sub_batch_id]['best_res']['ups'])
  pds_list = list(res[sub_batch_id]['best_res']['pds'])
  re_qty = list(res[sub_batch_id]['best_res']['re_qty'])  
  if len(dg_id)==1:
    n_cols = list([res[sub_batch_id]['best_res']['n_cols']])
  else:
    n_cols = list(res[sub_batch_id]['best_res']['n_cols'])
  # print(best_sheet, ink_seperator_width, dg_id, cg_id, label_w_list, label_h_list, n_cols, ups_list)
  left_sheet_width = best_sheet[0]-(len(set(cg_id))-1)*ink_seperator_width-np.sum(np.multiply(label_w_list, n_cols))
  print(f'label_w_list = {label_w_list}')  
  print(f'n_cols = {n_cols}')
  print(f're_qty = {re_qty}')     
  print(f'ups_list = {ups_list}')      
  print(f'pds_list = {pds_list}')    
  print(f'left_sheet_width = {left_sheet_width}')      

  scale = 100
  plt.figure(figsize=(best_sheet[0]/scale, best_sheet[1]/scale))
  plt.title(f"{best_comb}, {str(best_sheet)}") 
  plot_full_height_for_each_dg_with_ink_seperator(best_sheet, ink_seperator_width, dg_id, cg_id, label_w_list, label_h_list, n_cols, ups_list)
  plt.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3 allocate sku
# MAGIC 只需要在DG内部分分配sku

# COMMAND ----------

df_3_3 = df[['dimension_group','sku_id','re_qty']]
df_3_3 = df_3_3.sort_values(list(df_3_3.columns))

#revert字典，将batch号加入df_3_3
batch_revert = {}
for k,v in best_batch.items():
  for i in v:
    batch_revert[i] = k
df_3_3 = df_3_3[df_3_3['dimension_group'].isin(batch_revert.keys())]
df_3_3['sub_batch_id'] = df_3_3['dimension_group'].apply(lambda x: batch_revert[x])

df_3_3 = df_3_3.sort_values(['sub_batch_id','dimension_group','sku_id','re_qty'])
display(df_3_3)

# COMMAND ----------

n_abc = int(params_dict["user_params"]["n_abc"])

# COMMAND ----------

#在每一个dg内部分配做sku的ups分配
df_i_list = [] #存放每一个dg的ups分配情况
print('n_abc = ', n_abc)  

for sub_batch_id in best_batch.keys(): #for b0, b1, ...
  #一个sub_batch的sku allocation
  df_i = df_3_3[df_3_3['sub_batch_id']==sub_batch_id]
  # display(df_i)
  # dg_id = best_batch[sub_batch_id] #不能用这个，顺序会不一样
  best_comb = res[sub_batch_id]['best_comb'] #每个dg的旋转方向
  dg_id = [i[:-2] for i in best_comb.split('<+>')]  
  dg_orient = [i[-1] for i in best_comb.split('<+>')]
  ups_list = list(res[sub_batch_id]['best_res']['ups'])
  # print()
  # print(f'dg_id = {dg_id}')
  # print(f'ups_list = {ups_list}')  
  # print()

  for sub_dg_index in range(len(dg_id)): #在每一个dg内部分配做sku的ups分配
    sub_dg = dg_id[sub_dg_index]
    df_i_sub = df_i[df_i['dimension_group']==sub_dg]
    # print(f'sub_dg = {sub_dg}')

    #step 1: 按照n_abc*ups分配allocate sku
    sku_qty_dict = dict(zip(df_i_sub['sku_id'],df_i_sub['re_qty']))
    n_ups = ups_list[sub_dg_index]
    # print(f'sku_qty_dict = {sku_qty_dict}')    
    # print(f'n_ups = {n_ups}')       
    res_dict = allocate_sku(sku_qty_dict, n_ups*n_abc) ###--->>>
    # print(f'res_dict = {res_dict}')

    for sku_id in res_dict.keys():
      df_i_sub.loc[df_i_sub['sku_id']==sku_id, 'sku_ups'] = res_dict[sku_id]['ups']
      df_i_sub.loc[df_i_sub['sku_id']==sku_id, 'sku_pds'] = res_dict[sku_id]['pds']  

    # print(f"sum_sku_ups = {np.sum(df_i_sub['sku_ups'])}")
    # print(f"max_sku_ups = {np.max(df_i_sub['sku_ups'])}")
    # display(df_i_sub)

    df_i_sub = df_i_sub.sort_values(['dimension_group','sku_pds']).reset_index().drop(columns=['index']) #按照pds从小到大排序 - ppc要求
    df_i_sub['cum_sum_ups'] = df_i_sub.groupby(['dimension_group'])['sku_ups'].cumsum()   

    #step 2: 做ABC版的ups分割（每个版的ups应该相等）split ABC sheets
    df_i_sub = split_abc_ups(sub_id=sub_dg, sub_id_colname='dimension_group', df=df_i_sub, ups_dict={sub_dg:n_ups})

    #存放结果
    df_i_list.append(df_i_sub)

df_3_3_res = pd.concat(df_i_list).sort_values(['sub_batch_id','dimension_group','sku_id','re_qty'])
df_3_3_res['job_number'] = df_3_3_res['sku_id'].apply(lambda x: x.split('<+>')[0])
df_3_3_res['sku_seq'] = df_3_3_res['sku_id'].apply(lambda x: x.split('<+>')[1])
df_3_3_res = df_3_3_res[['sub_batch_id', 'dimension_group', 'sku_id', 'job_number', 'sku_seq', 're_qty', 'sku_ups', 'sku_pds', 'Set A Ups']].sort_values(['sub_batch_id', 'dimension_group', 'sku_id'])
print(f"sum_sku_ups = {np.sum(df_3_3_res['sku_ups'])}")
print(f"max_sku_ups = {np.max(df_3_3_res['sku_ups'])}")
display(df_3_3_res)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4 Prepare Results

# COMMAND ----------

from utils.load_data import initialize_dg_level_results, initialize_sku_level_results

# COMMAND ----------

#outputs
df_res = initialize_dg_level_results(df) #初始化和PPC结果比较的results data - 同时也是Files_to_ESKO的file-1
display(df_res)
res_file_3 = initialize_sku_level_results(df) #初始化结果文件Files_to_ESKO的file-3
display(res_file_3)

# COMMAND ----------

df_3_3_agg = df_3_3_res.groupby(['sub_batch_id','dimension_group']).agg({'sku_ups':'sum','sku_pds':'max'}).reset_index()
df_3_3_agg['sub_batch_id'] = 'mcmd<+>'+df_3_3_agg['sub_batch_id']
df_3_3_agg.rename(columns={'dimension_group':'DG'},inplace=True)
df_res = df_res.merge(df_3_3_agg, how='left', on='DG')
df_res.loc[df_res['batch_id'].isna(),'batch_id'] = df_res['sub_batch_id']
df_res.loc[df_res['ups'].isna(),'ups'] = df_res['sku_ups']
df_res.loc[df_res['pds'].isna(),'pds'] = df_res['sku_pds']
df_res.drop(columns=['sub_batch_id','sku_ups','sku_pds'],inplace=True)
df_res['the_pds'] = np.ceil(df_res['qty_sum']/df_res['ups'])

for k,v in best_batch.items():
  res_k = res[k]
  orient_list = [i[-1] for i in res_k['best_comb'].split('<+>')]
  best_sheet = res_k['best_sheet']  
  for i in range(len(v)):
    df_res.loc[df_res['DG']==v[i],'Printing_area'] = str(int(best_sheet[0]))+'x'+str(int(best_sheet[1]))
    if orient_list[i]=='w':
      df_res.loc[df_res['DG']==v[i],'orient'] = 'horizon'
    elif orient_list[i]=='h':
      df_res.loc[df_res['DG']==v[i],'orient'] = 'vertical'

#中离数
df_res_agg = df_res.groupby(['batch_id'])['Color Group'].nunique().reset_index()
n_seperator_dict = dict(zip(df_res_agg['batch_id'],df_res_agg['Color Group']))
for k,v in n_seperator_dict.items():
  df_res.loc[df_res['batch_id']==k,'中离数'] = int(v)-1

df_res = df_res.sort_values(['batch_id','Color Group','DG','Req_Qty'])
display(df_res)

# COMMAND ----------

df_res['weight'] = 1
df_res.loc[df_res['Printing_area']=='522x328','weight'] = 0.5
df_res['weighted_pds'] = df_res['weight']*df_res['pds']

# COMMAND ----------

df_res.groupby(['batch_id']).agg({'weighted_pds':'max'}).reset_index()

# COMMAND ----------

metrics_3_3 = np.sum(df_res.groupby(['batch_id']).agg({'weighted_pds':'max'}).values)
# metrics_3_3 = np.sum(df_3_3_res.groupby(['sub_batch_id']).agg({'weighted_sku_pds':'max'}).values)
n_batch = df_res['batch_id'].nunique()
print(f'sum_pds = {metrics_3_3+int(params_dict["user_params"]["add_pds_per_sheet"])*n_batch}') #在只有一种sheet_size的情况下只看sum_pds

# COMMAND ----------

#0319 case: 1000000, 10 dg, 5 grp, 6620s, 9091 sample/min
#0519 case: 1000, 13 dg, 5 grp, 6200s, 9.4 sample/min

# COMMAND ----------


