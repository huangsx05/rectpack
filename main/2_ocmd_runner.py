# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

from utils.load_data import load_config, load_and_clean_data, agg_to_get_dg_level_df, initialize_dg_level_results, initialize_sku_level_results
from utils.plot import plot_rectangle, plot_full_height_for_each_dg_with_ink_seperator
from utils.postprocess import prepare_dg_level_results, prepare_sku_level_results
from utils.tools import get_all_dg_combinations_with_orientation
from model.ocmd_solver import only_keep_cg_with_mul_dgs, filter_id_by_criteria_ocmd
from model.shared_solver import iterate_to_solve_min_total_sheet_area, allocate_sku, split_abc_ups

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

# COMMAND ----------

#algo inputs
config_file = f"../config/config_panyu_htl.yaml"
params_dict = load_config(config_file, params_dict)
# for k,v in params_dict.items():
#   print(f'{k}: {v}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 1: specific test data and params

# COMMAND ----------

filter_Color_Group = ["CG_09"]
# filter_Color_Group = [] #空代表不筛选，全部计算

# COMMAND ----------

#sample config
criteria = params_dict['criteria']
ink_seperator_width = params_dict['ink_seperator_width']
input_file = params_dict['input_file']
n_abc = params_dict['n_abc']
OCMD_filter = params_dict['OCMD_filter']
OCMD_criteria_check = params_dict['OCMD_criteria_check']
OCMD_layout_mode = params_dict['OCMD_layout_mode']

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
# MAGIC #### 2.1  Filter

# COMMAND ----------

#前接口
df_2 = only_keep_cg_with_mul_dgs(df_1)
print(f"candidate_id_list = {df_2['cg_id'].unique()}")
display(df_2)

# COMMAND ----------

id_list_2_1 = [] #所有有可能满足criteria的ids

if OCMD_filter:
  for cg_id in df_2['cg_id'].unique():
    fail_criteria = filter_id_by_criteria_ocmd(df_2, cg_id, sheet_size_list, criteria['one_cg_mul_dg']) ###### --->>> 
    if fail_criteria == False:
      id_list_2_1.append(cg_id)
else:
  id_list_2_1 = df_2['cg_id'].values

print(f'OCMD_filter = {OCMD_filter}')
print(f'qualified_id_list = {id_list_2_1}')

# COMMAND ----------

#后接口
df_2_1 = df_2[df_2['cg_id'].isin(id_list_2_1)]
if len(df_2_1)>0:
  display(df_2_1)
else:
  dbutils.notebook.exit('no candidate id to layout')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2 Layout
# MAGIC 假设：  
# MAGIC 1. 同一个cg_id，不管下面有多少种dimensions，都group在一起  
# MAGIC 2. OCMD_layout_mode == 'one_dg_one_column': 每个DG排整列，同一列上不同DG不混排
# MAGIC 3. OCMD_layout_mode == 'mul_dg_one_column': 同一列上不同DG可以混排

# COMMAND ----------

#前接口
#筛选进入这一步计算的ids
cg_id_list_2_2 = sorted(df_2_1['cg_id'].unique())
print(f'candidate cd_ids = {cg_id_list_2_2}')

# COMMAND ----------

#初始化
qualified_res_2_2 = {}
pass_cg_ids_2_2 = []
fail_cg_ids_2_2 = []
ups_2_2_dict = {} #存放每个dg对应的ups数，用于之后的sku分配

#遍历CG
for cg_id in cg_id_list_2_2:
  #获取cg对应df
  df_i = df_2_1[df_2_1['cg_id']==cg_id].sort_values(['cg_id','dimension_group'])
  if 'index' in df_i.columns:
    df_i.drop(columns=['index'], inplace=True)
  df_i = df_i.reset_index()  

  #获取所有dg rotation的combinations
  batch_name = 'batch_ocmd<+>'+cg_id
  print(f'$$$$$$ calculating batch = {batch_name} $$$$$$')  
  dg_id = df_i['dimension_group'].values.tolist()
  fix_orientation = df_i['fix_orientation'].values.tolist()
  label_width = df_i['overall_label_width'].values.tolist()
  label_length = df_i['overall_label_length'].values.tolist()
  # print(dg_id,fix_orientation,label_width,label_length)
  comb_names, comb_res_w, comb_res_h = get_all_dg_combinations_with_orientation(dg_id,fix_orientation,label_width,label_length) ###--->>>
  # print(f'len(comb_names) = {len(comb_names)}')
  # print(f'comb_names = {comb_names}')
  # print(f'comb_res_w = {comb_res_w}')
  # print(f'comb_res_h = {comb_res_h}')  #check w和h的对应关系

  #遍历所有comb和sheet_size，选择最优的comb+sheet_size的组合。简化目标：pds*sheet_area最小
  cg_id = [cg_id]
  re_qty = df_i['re_qty'].values.tolist()
  # print(f'sheet_size_list,comb_names,comb_res_w,comb_res_h,dg_id,cg_id,re_qty,ink_seperator_width = ')
  # print(sheet_size_list,comb_names,comb_res_w,comb_res_h,dg_id,cg_id,re_qty,ink_seperator_width)
  print(f'OCMD_layout_mode = {OCMD_layout_mode}')
  if OCMD_layout_mode=='one_dg_one_column': #不推荐使用
    best_comb, best_sheet, res, min_tot_area = iterate_to_solve_min_total_sheet_area(sheet_size_list,comb_names,comb_res_w,comb_res_h,dg_id,cg_id,re_qty,ink_seperator_width,
                                                                                      check_criteria=OCMD_criteria_check,criteria_dict=criteria['one_cg_mul_dg'],
                                                                                      mode='one_dg_one_column') ###
  elif OCMD_layout_mode=='mul_dg_one_column': #推荐使用，符合PPC要求的互扣式混排
    best_comb, best_sheet, res, min_tot_area = iterate_to_solve_min_total_sheet_area(sheet_size_list,comb_names,comb_res_w,comb_res_h,dg_id,cg_id,re_qty,ink_seperator_width,
                                                                                      check_criteria=OCMD_criteria_check,criteria_dict=criteria['one_cg_mul_dg'],
                                                                                      mode='mul_dg_one_column') ### mode与上面不同
  else:
    print(f'unrecognized OCMD_layout_mode = {OCMD_layout_mode}')   
    stop_flag = 1/0 
  
  res_for_print = {key: res[key] for key in ['re_qty', 'ups', 'pds', 'dg_layout_seq']}
  print(f'****** best_comb={best_comb}, best_sheet={best_sheet}, best_res={res_for_print}, min_tot_area={min_tot_area}') 
  #best_comb=dg_098_w<+>dg_099_w, best_sheet=[582, 482], best_res={'re_qty': [4145, 690], 'n_rows': [10, 10], 'n_cols': [16, 3], 'ups': [160, 30], 'pds': [26.0, 23.0]}, min_tot_area=7293624.0 

  #添加结果
  if best_comb!='dummy':
    pass_cg_ids_2_2.append(cg_id[0])
    max_pds = np.max(res['pds'])
    qualified_res_2_2[batch_name] = {'best_comb':best_comb, 'best_sheet':best_sheet, 'best_res':res, 'max_pds':max_pds, 'min_tot_area':min_tot_area}
    #prepare ups_dict for sku allocation use
    dg_id_list = [i[:-2] for i in best_comb.split('<+>')]
    ups_2_2_dict.update(dict(zip(dg_id_list, res['ups'])))
  else:
    fail_cg_ids_2_2.append(cg_id[0])    

# COMMAND ----------

#后接口
print('-'*50)
print(f'fail_cg_ids_2_2 = {fail_cg_ids_2_2}')
print(f'pass_cg_ids_2_2 = {pass_cg_ids_2_2}')
# print(f'qualified_res_2_2 = {qualified_res_2_2}')

# COMMAND ----------

display(df_2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3 Allocate SKU

# COMMAND ----------

#前接口
#get data for this step
df_2_3 = df[df['cg_id'].isin(pass_cg_ids_2_2)]
cols = ['cg_dg_id','cg_id', 'dimension_group', 'fix_orientation','overall_label_width', 'overall_label_length','sku_id','re_qty']
df_2_3 = df_2_3[cols]
# display(df_2_3)

# COMMAND ----------

if len(pass_cg_ids_2_2)==0:
  df_2_3 = pd.DataFrame()
else:
  #step 1: 按照n_abc*ups分配allocate sku
  print('ups for one sheet: ', ups_2_2_dict)
  print('n_abc = ', n_abc)  
  for dg_id, ups in ups_2_2_dict.items():
    print(f'------ sku allocation for {dg_id} ------')
    df_2_3_i = df_2_3[df_2_3['dimension_group']==dg_id]
    res_2_3 = allocate_sku(dict(zip(df_2_3_i['sku_id'], df_2_3_i['re_qty'].astype('int'))), ups_2_2_dict[dg_id]*n_abc) ### --->>>
    print(f'sku ups allocation = {res_2_3}')
    con_1 = df_2_3['dimension_group']==dg_id
    for sku_id, sku_res_dict in res_2_3.items():
      con_2 = df_2_3['sku_id']==sku_id
      df_2_3.loc[con_1&con_2, 'sku_ups'] = sku_res_dict['ups']
      df_2_3.loc[con_1&con_2, 'sku_pds'] = sku_res_dict['pds']

  df_2_3 = df_2_3.sort_values(['dimension_group','sku_pds']).reset_index().drop(columns=['index']) #按照pds从小到大排序 - ppc要求
  df_2_3['cum_sum_ups'] = df_2_3.groupby(['dimension_group'])['sku_ups'].cumsum()   

  #step 2: 做ABC版的ups分割（每个版的ups应该相等）split ABC sheets
  for dg_id in ups_2_2_dict.keys():
    df_2_3 = split_abc_ups(sub_id=dg_id, sub_id_colname='dimension_group', df=df_2_3, ups_dict=ups_2_2_dict)

# COMMAND ----------

#后接口
display(df_2_3)  

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4 Prepare Results

# COMMAND ----------

best_batch = qualified_res_2_2
res = qualified_res_2_2
for k,v in best_batch.items():
  print(k,v)

# COMMAND ----------

#plot上面的最优解
for sub_batch_id in best_batch.keys():
  best_sheet = res[sub_batch_id]['best_sheet']
  best_comb = res[sub_batch_id]['best_comb']
  ups_res_list = res[sub_batch_id]['best_res']['ups_info_by_col']    

  scale = 100
  plt.figure(figsize=(best_sheet[0]/scale, best_sheet[1]/scale))
  plt.title(f"{best_comb}, {str(best_sheet)}") 
  for i in range(len(ups_res_list)):
    for rec in ups_res_list[i]:
      plot_rectangle(rec[0]/scale,rec[1]/scale,rec[2]/scale,rec[3]/scale,color='blue')
  plt.show() 

# COMMAND ----------

#DG level results - 更新结果和PPC比较的结果df_res，同时也是Files_to_ESKO的file-1
df_res = prepare_dg_level_results(df_res, df_2, df_2_3, pass_cg_ids_2_2)
display(df_res)

# COMMAND ----------

#sku level results - 更新结果文件Files_to_ESKO的file-3
res_file_3 = prepare_sku_level_results(res_file_3, df_2_3)
display(res_file_3)

# COMMAND ----------

end_time = datetime.now()
print(start_time)
print(end_time)
print('running time =', (end_time-start_time).seconds, 'seconds')

# COMMAND ----------



# COMMAND ----------


