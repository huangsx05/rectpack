# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from utils.load_data import load_config, load_and_clean_data, agg_to_get_dg_level_df, initialize_dg_level_results, initialize_sku_level_results
from utils.plot import plot_ups_ocod
from utils.postprocess import prepare_dg_level_results, prepare_sku_level_results
from model.ocod_solver import filter_id_by_criteria_ocod, sheet_size_selection_ocod, check_criteria_ocod
from model.shared_solver import allocate_sku, split_abc_ups

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

dbutils.widgets.text("Films", "678x528, 582x482, 522x328", "Films") #678x528, 582x482, 522x328
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

# COMMAND ----------

#algo inputs - add to params_dict
config_file = f"../config/config_panyu_htl.yaml"
params_dict = load_config(config_file, params_dict)
# for k,v in params_dict.items():
#   print(f'{k}: {v}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 1: specific test data and params

# COMMAND ----------

filter_Color_Group = ["CG_08"]
# filter_Color_Group = [] #空代表不筛选，全部计算

# COMMAND ----------

#sample config
criteria = params_dict['criteria']
input_file = params_dict['input_file']
n_abc = params_dict['n_abc']
OCOD_filter = params_dict['OCOD_filter']
OCOD_criteria_check = params_dict['OCOD_criteria_check']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main

# COMMAND ----------

# MAGIC %md
# MAGIC #### - inputs

# COMMAND ----------

df = pd.read_csv(input_file)
if len(filter_Color_Group)>0:
  df = df[df['Color_Group'].isin(filter_Color_Group)]
display(df) #源数据，未经任何代码处理。须在Excel中填充缺失值和去空格（用下划线代替）

# COMMAND ----------

#clean intput data
df = load_and_clean_data(df)
display(df) #数据清洗后的，以sku为颗粒度的数据 - 整个计算的基础数据

# COMMAND ----------

#aggregation by cg_dg 以cg_dg_id分组，其实应该以dg_id分组就可以，也就是说dg_id是cg_id的下一级
df_1 = agg_to_get_dg_level_df(df)
display(df_1) #dg颗粒度的input data

# COMMAND ----------

# MAGIC %md
# MAGIC #### - outputs

# COMMAND ----------

#初始化和PPC结果比较的results data - 同时也是Files_to_ESKO的file-1
df_res = initialize_dg_level_results(df)
display(df_res)

# COMMAND ----------

#初始化结果文件Files_to_ESKO的file-3
res_file_3 = initialize_sku_level_results(df)
display(res_file_3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1  OCOD Filter
# MAGIC 过滤无论如何不可能满足criteria的id  
# MAGIC 比较鸡肋，因为在这个阶段即使遍历情况也不多  
# MAGIC 强烈建议使用OCOD_filter==False

# COMMAND ----------

#前接口
id_list_1 = sorted(df_1['cg_dg_id'].tolist()) #所有candidate ids
print(f'candidate_id_list = {id_list_1}')

# COMMAND ----------

print(f'OCOD_filter = {OCOD_filter}')

if OCOD_filter:
  #计算每一组cg_dg_rotation的最优sheet_size以及最优layout
  id_list_1_1 = [] #所有有可能满足criteria的ids
  for cg_dg_rotate_id in id_list_1:
    label_width = df_1.loc[df_1['cg_dg_id']==cg_dg_rotate_id, 'overall_label_width']
    label_length = df_1.loc[df_1['cg_dg_id']==cg_dg_rotate_id, 'overall_label_length']
    re_qty = df_1.loc[df_1['cg_dg_id']==cg_dg_rotate_id, 're_qty']
    fail_criteria = filter_id_by_criteria_ocod(label_width, label_length, re_qty, sheet_size_list, criteria['one_cg_one_dg']) ###### --->>> 
    if fail_criteria == False:
      id_list_1_1.append(cg_dg_rotate_id)
  # print(f'qualified_id_list = {id_list_1_1}')
  # #输出：id_list_1_1（有可能满足criteria，进入下一步的id列表）
else:
  id_list_1_1 = id_list_1.copy()

# COMMAND ----------

#后接口
print(f'qualified_id_list = {id_list_1_1}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2  OCOD Layout
# MAGIC 对于每一个ocod id, 选择最优的sheet_size，并决定在该sheet_size上的layout  
# MAGIC  - 遍历sheet_size_list中的sheet_size, 选择pds x sheet_area最小的sheet_size  
# MAGIC    (实际的目标是pds x sheet_area最小，且多一个batch相当于多N个pds。但在OCOD的情况下，全是一个cg_dg一个batch，所以目标简化为pds*sheet_area最小。）
# MAGIC  - for both fix_orientation==0 and fix_orientation==1, 遍历旋转和不旋转两种情况

# COMMAND ----------

#前接口
df_1_2 = df_1[df_1['cg_dg_id'].isin(id_list_1_1)] #根据1.1的结果筛选需要进行此计算步骤的id

# COMMAND ----------

#寻找最优sheet_size并计算ups和pds
for cg_dg_rotate_id in id_list_1_1:
  # print(f'------ calculating for {cg_dg_rotate_id} ------')
  df_temp = df_1_2[df_1_2['cg_dg_id']==cg_dg_rotate_id]
  label_size = [df_temp['overall_label_width'].values[0], df_temp['overall_label_length'].values[0]]
  fix_orientation = int(df_temp['fix_orientation'].values[0])
  # print(f'label_size={label_size}, fix_orientation={fix_orientation}')
  re_qty = df_temp['re_qty'].values[0]
  best_sheet_size, res = sheet_size_selection_ocod(sheet_size_list, label_size, re_qty, fix_orientation) ###--->>>遍历sheet_size
  # print(f'res = {res}')
  
  #formulate results
  cur_label_w = df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'overall_label_width']
  cur_label_h = df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'overall_label_length']  
  if res['is_rotated']==True:
    df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'overall_label_width'] = cur_label_h
    df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'overall_label_length'] = cur_label_w
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'sheet_width'] = best_sheet_size[0]
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'sheet_length'] = best_sheet_size[1]
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'cg_dg_layout'] = str(res)
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'ups'] = res['ups']
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'pds'] = res['pds'] 
  
#根据criteria再次确认结果
df_1_2['pds_check'] = df_1_2.apply(lambda x: check_criteria_ocod(x, criteria['one_cg_one_dg']), axis=1)
df_1_2['checkpoint_ocod'] = df_1_2['pds_check']
print('results for all ocod ids')
display(df_1_2)

# COMMAND ----------

#后接口
#把df拆分为两部分，第一部在ocod做sku分配，第二部分做ocmd
if OCOD_criteria_check:
  df_1_2 = df_1_2[df_1_2['checkpoint_ocod']==True] #for sku allocation
  print('results for qualified ocod ids')
  display(df_1_2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.5 allocate SKU  

# COMMAND ----------

#前接口
pass_cg_dg_ids_1_2 = list(df_1_2['cg_dg_id'].unique())
ups_1_2_dict = dict(zip(df_1_2['cg_dg_id'], df_1_2['ups'].astype('int')))
print('ups: ', ups_1_2_dict)

# COMMAND ----------

#所有n_abc适用的通用方案
if len(pass_cg_dg_ids_1_2)==0:
  df_1_5 = pd.DataFrame()
else:
  #get data for this step
  df_1_5 = df[df['cg_dg_id'].isin(pass_cg_dg_ids_1_2)]
  cols = ['cg_dg_id','cg_id', 'dimension_group', 'fix_orientation','overall_label_width', 'overall_label_length','sku_id','re_qty']
  df_1_5 = df_1_5[cols]
  #按照n_abc*ups分配allocate sku
  for cg_dg_id in pass_cg_dg_ids_1_2: #在每一个cg_dg内部为sku分配ups
    df_1_5_temp = df_1_5[df_1_5['cg_dg_id']==cg_dg_id]
    # print(f'------ sku allocation for {cg_dg_id} ------')
    #计算sku allocation结果 - 按照ups*n_abc分配
    res_1_5 = allocate_sku(dict(zip(df_1_5_temp['sku_id'], df_1_5_temp['re_qty'].astype('int'))), ups_1_2_dict[cg_dg_id]*n_abc) ### --->>>
    # print(res_1_5)
    #更新sku allocation结果到df
    con_1 = df_1_5['cg_dg_id']==cg_dg_id
    for sku_id, sku_res_dict in res_1_5.items():
      con_2 = df_1_5['sku_id']==sku_id
      df_1_5.loc[con_1&con_2, 'sku_ups'] = sku_res_dict['ups']
      df_1_5.loc[con_1&con_2, 'sku_pds'] = sku_res_dict['pds']

  df_1_5 = df_1_5.sort_values(['cg_dg_id','sku_pds']).reset_index().drop(columns=['index']) #按照pds从小到大排序 - ppc要求
  df_1_5['cum_sum_ups'] = df_1_5.groupby(['cg_dg_id'])['sku_ups'].cumsum()   

  #做ABC版的ups分割（每个版的ups应该相等）split ABC sheets
  sets = ['Set A Ups','Set B Ups','Set C Ups','Set D Ups','Set E Ups','Set F Ups','Set G Ups','Set H Ups'] #预设版数
  for cg_dg_id in pass_cg_dg_ids_1_2:
    df_1_5 = split_abc_ups(sub_id=cg_dg_id, sub_id_colname='cg_dg_id', df=df_1_5, ups_dict=ups_1_2_dict) ###--->>>

# COMMAND ----------

#后接口
display(df_1_5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4 Prepare Results

# COMMAND ----------

#plot ups layout (optional)
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
  plt.title(f"{cg_dg_id}, 'label_size={label_width}x{label_length}', 'sheet_size={sheet_width}x{sheet_length}'") 
  plot_ups_ocod([label_width, label_length], [sheet_width, sheet_length], layout_dict) ###--->>>

# COMMAND ----------

#DG level results - 更新结果和PPC比较的结果df_res，同时也是Files_to_ESKO的file-1
df_res = prepare_dg_level_results(df_res, df_1_2, df_1_5, pass_cg_dg_ids_1_2) ###--->>>
display(df_res)

# COMMAND ----------

#sku level results - 更新结果文件Files_to_ESKO的file-3
res_file_3 = prepare_sku_level_results(res_file_3, df_1_5) ###--->>>
display(res_file_3)

# COMMAND ----------

end_time = datetime.now()
print(start_time)
print(end_time)
print('running time =', (end_time-start_time).seconds, 'seconds')
