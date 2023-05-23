# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

from utils.load_data import load_config, load_and_clean_data, initialize_dg_level_results, initialize_sku_level_results
from utils.plot import plot_full_height_for_each_dg_with_ink_seperator
# from utils.postprocess import prepare_dg_level_results, prepare_sku_level_results
from utils.tools import get_all_dg_combinations_with_orientation
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
sheet_size_list = sheet_size_list.split(', ') #678x528, 582x482, 522x328
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
filter_Color_Group = ["CG_22","CG_23","CG_24","CG_26","CG_27","CG_28","CG_29","CG_30"]

# COMMAND ----------

#sample config
criteria = params_dict['criteria']
ink_seperator_width = params_dict['ink_seperator_width']
n_abc = params_dict['n_abc']

# COMMAND ----------

df = pd.read_csv(input_file)
df = df[df['Color_Group'].isin(filter_Color_Group)]
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main

# COMMAND ----------

# MAGIC %md
# MAGIC 需要决策怎么组合，且每种组合怎么选sheet_size；  
# MAGIC 多颜色会产生中离；  
# MAGIC 假设：同一个颜色的不同dimensions必然放同一个cell  ---  #cg_id相同的肯定在一起   
# MAGIC 中离的情况下，简化为每个区域“one color one/two/more dimension”的问题；
# MAGIC  - step 1: 每个cg所需要的最小宽度；
# MAGIC  - step 2: 根据每个cg的最小宽度和sheet_width, 遍历所有可能的组合
# MAGIC  - 取所有组合中某个metric(margin, pds, ...)u最优的一个

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
# MAGIC #### 数据接口

# COMMAND ----------

# # df_2 = df_1[~df_1['cg_dg_id'].isin(pass_cg_dg_ids_1_2)]
# df_2 = df_1.copy()
# df_2 = df_2[['cg_dg_id']+cols_to_first+['re_qty']]

# #过滤：只考虑有多行的cg_id，因为单行的应该在OCOD考虑
# df_2_cg_count = df_2.groupby('cg_id')['cg_dg_id'].count().reset_index().sort_values('cg_dg_id', ascending=False)
# multi_dg_cg = df_2_cg_count[df_2_cg_count['cg_dg_id']>1]['cg_id'].values
# df_2 = df_2[df_2['cg_id'].isin(multi_dg_cg)].sort_values(['cg_id', 're_qty'], ascending=[True, False])
# display(df_2)

# COMMAND ----------

df_3 = df_1.copy()
df_3.rename(columns={'dimension_group':'dg_id'},inplace=True)
display(df_3)
cg_agg_cnt = df_3.groupby('cg_id')['cg_dg_id'].agg('count').reset_index()
cg_agg_cnt = dict(zip(cg_agg_cnt['cg_id'],cg_agg_cnt['cg_dg_id']))
print(cg_agg_cnt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Batching
# MAGIC Lawrence is working on this part

# COMMAND ----------

### codes added by laoluo
df_3_required_col = df_3[['dg_id','re_qty','overall_label_width','overall_label_length','fix_orientation']]
df_3_required_col['required_total_area'] = df_3_required_col.apply(lambda x: int(x.re_qty*x.overall_label_width*x.overall_label_length), axis=1)
display(df_3_required_col)

# COMMAND ----------

 dg_req_area_dict = dict(zip(df_3_required_col['dg_id'].tolist(), df_3_required_col['required_total_area'].tolist()))
 totalAreaSum = sum(dg_req_area_dict.values())
 maxiPossibleGrps = len(dg_req_area_dict) 
 amplf_factor = 1
 numgroups = 1
 batches_list = []
 while numgroups <= maxiPossibleGrps:
    approxiGroupSum = totalAreaSum/numgroups
    dg_req_area_working = {k: v for k, v in dg_req_area_dict.items()}
    groups_list = []
    while len(dg_req_area_working)>0:
      approxiGroupSum = approxiGroupSum * amplf_factor
      group_candidate = {}
      while sum(group_candidate.values())<approxiGroupSum and len(dg_req_area_working)>0:
        min_key = min(dg_req_area_working, key=dg_req_area_working.get)
        min_value = dg_req_area_working.get(min_key)
        group_candidate[min_key] = min_value
        del dg_req_area_working[min_key]
      groups_list.append(group_candidate) 
    numgroups = numgroups + 1  

    ## 检查分组是否满足不同color不操作多少个dg的限制条件

    ## 将分组按要求格式输出
    batch = {}
    for i in range(len(groups_list)):
      b_key = 'b'+str(i)
      batch[b_key] = list(groups_list[i].keys()) 
    print(batch)
    batches_list.append(batch)    
    ## 下一环节的确定sheetsize和各dg的ups
#print(batches_list)

print('---------------------------------- split boundary ------------------------------------')

numgroups = 1
amplf_factor = 1.5
while numgroups <= maxiPossibleGrps:
  approxiGroupSum = totalAreaSum/numgroups
  dg_req_area_working = {k: v for k, v in dg_req_area_dict.items()}
  groups_list = []
  while len(dg_req_area_working)>0:
    approxiGroupSum = approxiGroupSum * amplf_factor
    group_candidate = {}
    pickedSize = 0
    while sum(group_candidate.values())<approxiGroupSum and len(dg_req_area_working)>0:
      if pickedSize%2 == 0:
        min_key = min(dg_req_area_working, key=dg_req_area_working.get)
        min_value = dg_req_area_working.get(min_key)
        group_candidate[min_key] = min_value
        del dg_req_area_working[min_key]
      else:
        max_key = max(dg_req_area_working, key=dg_req_area_working.get)
        max_value = dg_req_area_working.get(max_key)
        group_candidate[max_key] = max_value
        del dg_req_area_working[max_key]
      pickedSize = pickedSize + 1      
    groups_list.append(group_candidate)
  numgroups = numgroups + 1  

  ## 检查分组是否满足不同color不操作多少个dg的限制条件

  ## 将分组按要求格式输出
  batch = {}
  for i in range(len(groups_list)):
    b_key = 'b'+str(i)
    batch[b_key] = list(groups_list[i].keys()) 
  print(batch)
  batches_list.append(batch)    
    ## 下一环节的确定sheetsize和各dg的ups
print(batches_list)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Layout
# MAGIC 遍历3.1中得到的batches，找出pds*sheet_size最小的batch做为最优解  
# MAGIC 对每一个batch，遍历dg的旋转组合情况  
# MAGIC 对每一个组合，遍历所有sheet_size
# MAGIC
# MAGIC 假设：每一个dg_id都排整列，同一列上不同dg_id不混排  
# MAGIC 决策：每一个dg是否旋转，每一个dg排几列  

# COMMAND ----------

#sample batch 输入
# batch = {'b0':['dg_084','dg_086'],
#         'b1':['dg_087','dg_088'],
#         'b2':['dg_091','dg_093'],
#         'b3':['dg_094','dg_095','dg_098','dg_099']}

# batches_list = [
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091', 'dg_086', 'dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091'], 'b1': ['dg_086', 'dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091'], 'b1': ['dg_086', 'dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093'], 'b1': ['dg_091', 'dg_086'], 'b2': ['dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094'], 'b1': ['dg_093', 'dg_091'], 'b2': ['dg_086'], 'b3': ['dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094'], 'b1': ['dg_093', 'dg_091'], 'b2': ['dg_086'], 'b3': ['dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094'], 'b1': ['dg_093', 'dg_091'], 'b2': ['dg_086'], 'b3': ['dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095'], 'b1': ['dg_094', 'dg_093'], 'b2': ['dg_091'], 'b3': ['dg_086'], 'b4': ['dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095'], 'b1': ['dg_094', 'dg_093'], 'b2': ['dg_091'], 'b3': ['dg_086'], 'b4': ['dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095'], 'b1': ['dg_094', 'dg_093'], 'b2': ['dg_091'], 'b3': ['dg_086'], 'b4': ['dg_088']},
# {'b0':['dg_084','dg_086'],
#         'b1':['dg_087','dg_088'],
#         'b2':['dg_091','dg_093'],
#         'b3':['dg_094','dg_095','dg_098','dg_099']} #ppc solution
# ]

old_batches = [
#   {'b0': ['dg_084', 'dg_087', 'dg_094', 'dg_095', 'dg_098', 'dg_099'], 'b1': ['dg_093', 'dg_091'], 'b2': ['dg_086'], 'b3': ['dg_088']}, #sum_pds = 849.0
#   # {'b0': ['dg_084', 'dg_087', 'dg_095', 'dg_098', 'dg_099'], 'b1': ['dg_094', 'dg_093'], 'b2': ['dg_091'], 'b3': ['dg_086'], 'b4': ['dg_088']}, #sum_pds = 367.0 --- current best
#   # {'b0': ['dg_084','dg_086'], 'b1':['dg_087','dg_088'], 'b2':['dg_091','dg_093'], 'b3':['dg_094','dg_095','dg_098','dg_099']}, #sum_pds = 376.0 --- ppc solution - 360
#   # {'b0': ['dg_084', 'dg_086', 'dg_087', 'dg_088', 'dg_091', 'dg_098', 'dg_099'], 'b1': ['dg_095', 'dg_093', 'dg_094']}, #sum_pds = 512.0
#   # {'b0': ['dg_084', 'dg_086', 'dg_087', 'dg_088', 'dg_099'], 'b1': ['dg_098', 'dg_091', 'dg_095', 'dg_093', 'dg_094']}, #sum_pds = 467.0
#   # {'b0': ['dg_087', 'dg_088'], 'b1': ['dg_099', 'dg_086', 'dg_084', 'dg_091'], 'b2': ['dg_098', 'dg_093', 'dg_095', 'dg_094']}, #sum_pds = 407.0
#   # {'b0': ['dg_087', 'dg_088'], 'b1': ['dg_099', 'dg_086', 'dg_084'], 'b2': ['dg_098', 'dg_091', 'dg_095', 'dg_093'], 'b3': ['dg_094']} #sum_pds = 400.0
  ]

#对batches_list根据dg名称排序
for i in range(len(batches_list)):
  dict_i = batches_list[i]
  k = list(dict_i.keys())[0]
  v = sorted(list(dict_i.values())[0])
  # print(k,v)
  dict_i[k] = v
  batches_list[i] = dict_i

batches = [b for b in batches_list if b not in old_batches]
print(len(batches_list),len(batches))
for b in batches:
  print(b)

# COMMAND ----------

#batches去重
print(f'before drop_duplicates, lane(batches) = {len(batches)}')
batches_drop_duplicates = []
unique_str = []
for i in range(len(batches)):
  if str(batches[i]) not in unique_str:
    unique_str.append(str(batches[i]))
    batches_drop_duplicates.append(batches[i])
batches = batches_drop_duplicates
print(f'after drop_duplicates, lane(batches) = {len(batches)}')

#根据dg名称排序
for i in range(len(batches)):
  dict_i = batches[i]
  k = list(dict_i.keys())[0]
  v = sorted(list(dict_i.values())[0])
  # print(k,v)
  dict_i[k] = v
  batches[i] = dict_i
# print(batches) #列表形式

#转化batches输入为字典
batches_dict = {}
for i in range(len(batches)):
  batch_name = 'batch_'+str(i)
  batches_dict[batch_name] = batches[i]
print()
print(batches_dict)
print()
for k in batches_dict.keys():
  print(batches_dict[k])

# COMMAND ----------

#遍历batches找最优batch
#sample batches_dict
#{'batch_0': {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091', 'dg_086', 'dg_088']}, 
# 'batch_1': {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091'], 'b1': ['dg_086', 'dg_088']}}
res_metric_3_2 = {}
res_detail_3_2 = {}
#遍历batches
for i in range(len(batches_dict)):
  break_flag = 0 #color#>5时break
  batch_name = 'batch_'+str(i)
  res_detail_3_2[batch_name] = {}
  batch = batches_dict[batch_name] #{'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091', 'dg_086', 'dg_088']}
  print()
  print(f'$$$$$$ calculating batch = {batch} $$$$$$')

  #revert字典，将batch号加入df_3
  batch_revert = {}
  for k,v in batch.items():
    for i in v:
      batch_revert[i] = k
  df_3['batch_id'] = df_3['dg_id'].apply(lambda x: batch_revert[x])
  # display(df_3.sort_values(['batch_id','cg_id','dg_id']))

  #遍历dg rotation和sheet size
  #对每一个batch，找到最优解- 中离
  for batch_id in batch.keys(): #这里的batch_id是sub_batch_id
    # print()
    # print(f'iterate to find best dg_rotate_comb and best sheet_size for batch = {batch_id}')
    #准备数据
    df_i = df_3[df_3['batch_id']==batch_id]
    df_i = df_i.sort_values(['cg_id', 'dg_id'])
    # display(df_i)
    cg_id = df_i['cg_id'].values.tolist() #cg相同的必须相邻
    if len(set(cg_id))>5:
      print('ERROR: nunique_color > 5, skip this case')
      break_flag = 1
      break
    dg_id = sorted(df_i['dg_id'].values.tolist())
    fix_orientation = df_i['fix_orientation'].values.tolist()
    label_width = df_i['overall_label_width'].values.tolist()
    label_length = df_i['overall_label_length'].values.tolist()
    re_qty = df_i['re_qty'].values.tolist()
    print(f'dg_id = {dg_id}，re_qty = {re_qty}')
    # print(cg_id, dg_id, fix_orientation, label_width, label_length, re_qty)

    #穷举该batch在一个sheet中排列的所有组合，考虑rotation  
    #每一个comb都包含batch中的所有dg，区别在于dg是否rotate
    comb_names, comb_res_w, comb_res_h = get_all_dg_combinations_with_orientation(dg_id,fix_orientation,label_width,label_length)
    # print(f'len(comb_names) = {len(comb_names)}')
    # print(f'comb_names = {comb_names}')
    # print(f'comb_res_w = {comb_res_w}')
    # print(f'comb_res_h = {comb_res_h}')  #check w和h的对应关系

    #遍历所有comb和sheet_size，选择总耗材面积最小的comb+sheet_size的组合
    best_comb, best_sheet, res, min_tot_area = iterate_to_solve_min_total_sheet_area(sheet_size_list,comb_names,comb_res_w,comb_res_h,dg_id,cg_id,re_qty,ink_seperator_width,
                                                                                    check_criteria=False,criteria_dict=criteria['mul_cg_mul_dg'],
                                                                                    mode='one_dg_one_column') ###
    print(f'****** best_comb={best_comb}, best_sheet={best_sheet}, best_res={res}, min_tot_area={min_tot_area}')  

    #结果添加至res_3_2字典
    max_pds = np.max(res['pds'])
    batch_the_pds = np.ceil(np.sum(res['re_qty'])/np.sum(res['ups']))
    res_detail_3_2[batch_name][batch_id] = {'best_comb':best_comb, 'best_sheet':best_sheet, 'best_res':res, 'max_pds':max_pds, 'batch_the_pds':batch_the_pds, 'min_tot_area':min_tot_area}
    #{'b0': {'best_comb': 'dg_084_h<+>dg_087_w<+>dg_095_w<+>dg_098_w<+>dg_099_w', 'best_sheet': [678, 528], 'best_res': {'re_qty': [1275, 440, 5794, 4145, 690], 'ups': [26, 13, 112, 77, 22], 'pds': [50.0, 34.0, 52.0, 54.0, 32.0], 'n_rows': [13, 13, 14, 11, 11], 'n_cols': [2, 1, 8, 7, 2]}, 'max_pds': 54.0, 'min_tot_area': 19331136.0}, 'b1': {'best_comb': 'dg_093_w<+>dg_094_w', 'best_sheet': [522, 328], 'best_res': {'re_qty': [10638, 7934], 'ups': [88, 63], 'pds': [121.0, 126.0], 'n_rows': [8, 9], 'n_cols': [11, 7]}, 'max_pds': 126.0, 'min_tot_area': 21573216.0}}

  if break_flag == 1:
    continue

  #计算当前batch的指标
  res_batch = res_detail_3_2[batch_name]
  # 'batch_5': {
  # 'b0': {'best_comb': 'dg_084_w<+>dg_086_h', 'best_sheet': [678, 528], 'best_res': {'n_rows': [7, 8], 'n_cols': [3, 14], 'ups': array([ 21, 112]), 'pds': [61.0, 91.0]}, 'max_pds': 91.0, 'min_tot_area': 32576544.0}, 
  # 'b1': {'best_comb': 'dg_087_w<+>dg_088_w', 'best_sheet': [678, 528], 'best_res': {'n_rows': [13, 8], 'n_cols': [1, 14], 'ups': array([ 13, 112]), 'pds': [34.0, 99.0]}, 'max_pds': 99.0, 'min_tot_area': 35440416.0}
  # }
  metric = 0
  for k,v in res_batch.items():
    metric += v['batch_the_pds']
  print(f'****** sum_batch_the_pds = {metric}')  
  res_metric_3_2[batch_name] = metric

print('-'*50)
print(res_detail_3_2)
print('-'*50)
print(res_metric_3_2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot for the best batch

# COMMAND ----------

#根据上一环节的结果得到最优batch的batch_name
best_batch_name = min(res_metric_3_2, key=res_metric_3_2.get)
print(f"best_batch_name = '{best_batch_name}'")

# COMMAND ----------

best_batch = batches_dict[best_batch_name]
res = res_detail_3_2[best_batch_name]
print(best_batch)
print(res)

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

best_batch

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
print(f"sum_sku_ups = {np.sum(df_3_3_res['sku_ups'])}")
print(f"max_sku_ups = {np.max(df_3_3_res['sku_ups'])}")
display(df_3_3_res)

# COMMAND ----------

print(best_batch)
print()
print(res)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4 Prepare Results

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
df_res_agg = df_res.groupby(['batch_id'])['Color Group'].count().reset_index()
n_seperator_dict = dict(zip(df_res_agg['batch_id'],df_res_agg['Color Group']))
for k,v in n_seperator_dict.items():
  df_res.loc[df_res['batch_id']==k,'中离数'] = int(v)

df_res = df_res.sort_values(['batch_id','Color Group','DG','Req_Qty'])
display(df_res)

# COMMAND ----------

metrics_3_3 = np.sum(df_3_3_res.groupby(['sub_batch_id']).agg({'sku_pds':'max'}).values)
print(f'sum_pds = {metrics_3_3}') #在只有一种sheet_size的情况下只看sum_pds

# COMMAND ----------

end_time = datetime.now()
print(start_time)
print(end_time)
print('running time =', (end_time-start_time).seconds, 'seconds')

# COMMAND ----------



# COMMAND ----------


