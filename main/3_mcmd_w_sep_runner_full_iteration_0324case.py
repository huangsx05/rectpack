# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

from utils.load_data import load_config, load_and_clean_data, agg_to_get_dg_level_df, initialize_dg_level_results, initialize_sku_level_results
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

dbutils.widgets.text("Colors", '1', "Colors")
n_color = dbutils.widgets.get("Colors")

n_abc = int(n_abc)

sheet_size_list = sheet_size_list.split(', ') #678x528, 582x482, 522x328
sheet_size_list = [sorted([int(i.split('x')[0]), int(i.split('x')[1])],reverse=True) for i in sheet_size_list]

n_color = int(n_color)
if batching_type!='4_MCMD_No_Seperater':
  add_pds_per_sheet = int((n_color+1)*3.5)
else:
  add_pds_per_sheet = int((n_color+2)*3.5)

params_dict = {
  'batching_type':batching_type,
  'n_abc':n_abc,
  'n_color':n_color,  
  'request_type':request_type,
  'sheet_size_list':sheet_size_list,
  }

# COMMAND ----------

#algo inputs
config_file = f"../config/config_panyu_htl_0419case.yaml"
params_dict = load_config(config_file, params_dict)
for k,v in params_dict.items():
  print(f'{k}: {v}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Part 1: specific test data and params

# COMMAND ----------

filter_Color_Group = ['CG_22', 'CG_23', 'CG_24', 'CG_26', 'CG_27', 'CG_28', 'CG_29', 'CG_30']
# filter_Color_Group = [] #空代表不筛选，全部计算

# COMMAND ----------

#sample config
criteria = params_dict['criteria']
input_file = params_dict['input_file']
ink_seperator_width = params_dict['ink_seperator_width']
n_abc = params_dict['n_abc']
n_color_limit = params_dict['n_color_limit'][batching_type]

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
# MAGIC #### 数据接口

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

# COMMAND ----------

# MAGIC %%time
# MAGIC import random
# MAGIC sample_num = 100000
# MAGIC M = min(5,df_3['dg_id'].nunique())  #分组数量上限
# MAGIC N = df_3['dg_id'].nunique() #元素数量
# MAGIC dg_sorted_list = sorted(df_3['dg_id'].tolist())
# MAGIC dg_cg_dict = dict(zip(df_3['dg_id'].tolist(), df_3['cg_id'].tolist()))
# MAGIC n_grp_lower = int(np.ceil(df_3['cg_id'].nunique()/n_color_limit)) #按照颜色数量决定分组下限
# MAGIC # print(f'n_grp_lower={n_grp_lower}')
# MAGIC
# MAGIC batches_list = []
# MAGIC v_set_list = []
# MAGIC combination_list = []
# MAGIC for n in range(N**M): #所有可能的组合的个数为N**M
# MAGIC   # print(f' ------ {n} -')
# MAGIC   combination = [[] for __ in range(M)] #初始化
# MAGIC   for i in range(N):
# MAGIC     combination[n // M**i % M].append(i)
# MAGIC
# MAGIC   combination_list.append(combination)
# MAGIC combination_list = random.sample(combination_list, sample_num)
# MAGIC print(len(combination_list))
# MAGIC
# MAGIC for combination in combination_list:
# MAGIC   batch = []
# MAGIC   for c in combination:
# MAGIC     if len(c)>0:
# MAGIC       sub_batch = [dg_sorted_list[i] for i in c]
# MAGIC       batch.append(sub_batch)
# MAGIC   # if len(batch)>=n_grp_lower:
# MAGIC   if len(batch)>=2:    
# MAGIC     #去掉颜色数大于limit的sub_batch    
# MAGIC     for sub_batch in batch:
# MAGIC       colors = [dg_cg_dict[s] for s in sub_batch]
# MAGIC       if len(set(colors))<=n_color_limit:
# MAGIC         batch_dict = {}
# MAGIC         for i in range(len(batch)):
# MAGIC           b_key = 'b'+str(i)
# MAGIC           batch_dict[b_key] = batch[i]         
# MAGIC         v_set = set([str(i) for i in batch_dict.values()])  
# MAGIC         if v_set not in v_set_list:
# MAGIC           v_set_list.append(v_set)
# MAGIC           batches_list.append(batch_dict)
# MAGIC           print(batch_dict)
# MAGIC print(len(batches_list))

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
# # {'b0': ['dg_087', 'dg_094', 'dg_095', 'dg_098', 'dg_099'], 'b1': ['dg_084', 'dg_086', 'dg_093'], 'b2': ['dg_088', 'dg_091']},
# # {'b0': ['dg_095', 'dg_098', 'dg_099'], 'b1': ['dg_087', 'dg_094'], 'b2': ['dg_088', 'dg_091'], 'b3': ['dg_084', 'dg_086', 'dg_093']},
# {'b0': ['dg_094', 'dg_098', 'dg_099'], 'b1': ['dg_087', 'dg_095'], 'b2': ['dg_086', 'dg_093'], 'b3': ['dg_084'], 'b4': ['dg_088', 'dg_091']}
# ]

ppc_batch = [
{'b0': ['dg_095', 'dg_098', 'dg_099'], 'b1': ['dg_084', 'dg_087'], 'b2': ['dg_088', 'dg_093'], 'b3': ['dg_091'], 'b4': ['dg_086', 'dg_094']},
{'b0': ['dg_087', 'dg_095', 'dg_098', 'dg_099'], 'b1': ['dg_084', 'dg_093'], 'b2': ['dg_088'], 'b3': ['dg_091'], 'b4': ['dg_086', 'dg_094']},
{'b0':['dg_084','dg_086'],
 'b1':['dg_087','dg_088'],
 'b2':['dg_091','dg_093'],
 'b3':['dg_094','dg_095','dg_098','dg_099']
 } #ppc solution - 0419
]
batches_list = ppc_batch+batches_list

old_batches = [
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

for b in batches:
  print(b)

# COMMAND ----------

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
    if len(set(cg_id))>n_color_limit: #这里可以优化代码效率，因为目前是算到color大于limit的sub_batch才会break, 前面的sub_batch还是被计算了
      print(f'ERROR: nunique_color > {n_color_limit}, skip this case')
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
    #考虑pds和sheet_weight
    sheet = v['best_sheet']
    sheet_name = str(int(sheet[0]))+'<+>'+str(int(sheet[1]))
    sheet_weight = criteria['mul_cg_mul_dg'][sheet_name]['weight']
    metric += v['max_pds']*sheet_weight
  #再考虑版数和pds之间的权衡
  add_metric = len(res_batch)*add_pds_per_sheet
  metric += add_metric

  print(f'****** weighted_sum_batch_the_pds + sheet_equivalent_pds = {metric}')  
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

# {'b0':['dg_084','dg_086'],
#  'b1':['dg_087','dg_088'],
#  'b2':['dg_091','dg_093'],
#  'b3':['dg_094','dg_095','dg_098','dg_099']
#  } - 426
# {'b0': ['dg_087', 'dg_094', 'dg_095', 'dg_098', 'dg_099'], 'b1': ['dg_084', 'dg_086', 'dg_093'], 'b2': ['dg_088', 'dg_091']}
# {'b0': ['dg_095', 'dg_098', 'dg_099'], 'b1': ['dg_087', 'dg_094'], 'b2': ['dg_088', 'dg_091'], 'b3': ['dg_084', 'dg_086', 'dg_093']}
# {'b0': ['dg_094', 'dg_098', 'dg_099'], 'b1': ['dg_087', 'dg_095'], 'b2': ['dg_086', 'dg_093'], 'b3': ['dg_084'], 'b4': ['dg_088', 'dg_091']} - 378
#{'batch_0': 426.0, 'batch_1': 365.0, 'batch_2': 369.5, 'batch_3': 378.0}

# COMMAND ----------


