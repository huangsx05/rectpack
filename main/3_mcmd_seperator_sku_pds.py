# Databricks notebook source
# MAGIC %md
# MAGIC 需要决策怎么组合，且每种组合怎么选sheet_size；  
# MAGIC 多颜色会产生中离；  
# MAGIC 假设：同一个颜色的不同dimensions必然放同一个cell  ---  #cg_id相同的肯定在一起   
# MAGIC 中离的情况下，简化为每个区域“one color one/two/more dimension”的问题；
# MAGIC  - step 1: 每个cg所需要的最小宽度；
# MAGIC  - step 2: 根据每个cg的最小宽度和sheet_width, 遍历所有可能的组合
# MAGIC  - 取所有组合中某个metric(margin, pds, ...)u最优的一个  
# MAGIC 当前方案下，每一个dg占用整列，即使是同颜色不同dg在同一列上也不会混排

# COMMAND ----------

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

from utils.load_data import load_config, load_and_clean_data, agg_to_get_dg_level_df, initialize_input_data, initialize_dg_level_results, initialize_sku_level_results
from utils.plot import plot_full_height_for_each_dg_with_ink_seperator
# from utils.postprocess import prepare_dg_level_results, prepare_sku_level_results
from utils.tools import allocate_sku, get_all_dg_combinations_with_orientation
from model.shared_solver import get_batches_with_filter, iterate_to_solve_min_total_sheet_area, split_abc_ups

# COMMAND ----------

start_time = datetime.now()
print(start_time)

# COMMAND ----------

# MAGIC %md
# MAGIC #### config

# COMMAND ----------

#inputs
dbutils.widgets.text("config_filename", 'config_panyu_htl.yaml')
config_filename = dbutils.widgets.get("config_filename")
print(config_filename)

# COMMAND ----------

params_dict = load_config('/tmp/'+config_filename)
for k,v in params_dict.items():
  print(f'{k}: {v}')

# COMMAND ----------

# 当前notebook会用到的params，其他的会再调用函数中直接传入params_dict
filter_Color_Group = params_dict['filter_Color_Group']
input_file = params_dict['input_file']

algo_time_limit = params_dict['algo_params']['algo_time_limit']
# sample_batch = params_dict['algo_params']['sample_batch'] #true/false
sample_batch_num = params_dict['algo_params']['sample_batch_num'] #考虑做成动态调整,并考虑在时间允许的范围内loop

ink_seperator_width = params_dict['business_params']['ink_seperator_width']
n_color_limit = params_dict['business_params']['n_color_limit']

add_pds_per_sheet = params_dict['user_params']['add_pds_per_sheet']
n_abc = params_dict['user_params']['n_abc']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main

# COMMAND ----------

#inputs
df_raw, df, df_1 = initialize_input_data(input_file, filter_Color_Group) #------ 数据清洗部分可以转移到GPM完成
print(f"input data before data cleaning:")
display(df_raw) #源数据，未经任何代码处理。须在Excel中填充缺失值和去空格（用下划线代替）
print(f"input data after data cleaning:")
display(df) #数据清洗后的，以sku为颗粒度的数据 - 整个计算的基础数据
print(f"aggregated input data at dg level:")
display(df_1) #dg颗粒度的input data #aggregation by cg_dg 以cg_dg_id分组，其实应该以dg_id分组就可以，也就是说dg_id是cg_id的下一级

# COMMAND ----------

#准备sku level的dict
dg_sku_qty_dict = {}
for dg_name in df['dimension_group'].unique(): #在每一个dg内部分配做sku的ups分配
  df_i_sub = df[df['dimension_group']==dg_name]
  sku_qty_dict = dict(zip(df_i_sub['sku_id'],df_i_sub['re_qty']))
  dg_sku_qty_dict[dg_name] = sku_qty_dict
# print(dg_sku_qty_dict)

# COMMAND ----------

#outputs
df_res = initialize_dg_level_results(df) #初始化和PPC结果比较的results data - 同时也是Files_to_ESKO的file-1
display(df_res)
res_file_3 = initialize_sku_level_results(df) #初始化结果文件Files_to_ESKO的file-3
display(res_file_3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 数据接口

# COMMAND ----------

df_3 = df_1.copy()
df_3.rename(columns={'dimension_group':'dg_id'},inplace=True)
# display(df_3)
cg_agg_cnt = df_3.groupby('cg_id')['cg_dg_id'].agg('count').reset_index()
cg_agg_cnt = dict(zip(cg_agg_cnt['cg_id'],cg_agg_cnt['cg_dg_id']))
print("dg count for each color group:")
print(cg_agg_cnt)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Batching

# COMMAND ----------

print(f"batch_generate_mode = {params_dict['algo_params']['batch_generate_mode']}")
batches_list = get_batches_with_filter(df_3, params_dict, n_color_limit)

# COMMAND ----------

# #sample batch 输入
# batches_list = [
# {'b0': ['dg_10', 'dg_11', 'dg_12', 'dg_13'], 'b1': ['dg_02'], 'b2': ['dg_01', 'dg_04', 'dg_09'], 'b3': ['dg_03', 'dg_05', 'dg_08'], 'b4': ['dg_06', 'dg_07']}
# # {'b0': ['dg_087', 'dg_098', 'dg_099'], 'b1': ['dg_088', 'dg_091'], 'b2': ['dg_084', 'dg_093'], 'b3': ['dg_094', 'dg_095'], 'b4': ['dg_086']}
# ]

# ppc_batch = [
# {'b0':['dg_01','dg_02','dg_03','dg_04'],'b1':['dg_05','dg_06','dg_07','dg_08','dg_09'],'b2':['dg_10'],'b3':['dg_11'],'b4':['dg_12','dg_13'] } #ppc solution - 0519
# # {'b0':['dg_084','dg_086'],'b1':['dg_087','dg_088'],'b2':['dg_091','dg_093'],'b3':['dg_094','dg_095','dg_098','dg_099']} #ppc solution - 0419
# ]
# batches_list = ppc_batch+batches_list

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Layout
# MAGIC 遍历3.1中得到的batches，找出pds*sheet_weight最小的batch做为最优解  
# MAGIC 对每一个batch，遍历dg的旋转组合情况  
# MAGIC 对每一个组合，遍历所有sheet_size
# MAGIC
# MAGIC 假设：每一个dg_id都排整列，同一列上不同dg_id不混排  
# MAGIC 决策：每一个dg是否旋转，每一个dg排几列  

# COMMAND ----------

old_batches = []
n_count = 0
n_current = 0

res_metric_3_2 = {}
res_detail_3_2 = {}
best_metric = 1e12
best_index = 0
best_batch = []
best_res = {}

while True: #时限未到
  #取样
  #remove batches in old_batches
  print()
  print(f'before dropping old batches, len(batches) = {len(batches_list)}')
  batches_list = [b for b in batches_list if b not in old_batches]
  print(f'after dropping old batches, len(batches) = {len(batches_list)}')
  sample_batch_num = np.min([sample_batch_num,len(batches_list)])
  batches = random.sample(batches_list, sample_batch_num)

  #转化batches输入为字典(列表转换为字典)
  batches_dict = {}
  for i in range(len(batches)):
    batch_name = 'batch_'+str(n_count+i)
    batches_dict[batch_name] = batches[i]
  n_count += len(batches)
  # print(batches_dict[list(batches_dict.keys())[0]]) #print a sample

  #主计算部分 -----------------------------------------------------------------------------------------------------------------------
  #遍历batches找最优batch
  #sample batches_dict
  #{'batch_0': {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091', 'dg_086', 'dg_088']}, 
  # 'batch_1': {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091'], 'b1': ['dg_086', 'dg_088']}}
  for i in range(len(batches_dict)):
    #初始化
    break_flag = 0 #用于控制结果不可能更优时退出当前batch
    batch_name = 'batch_'+str(n_current)
    res_detail_3_2[batch_name] = {}
    #获得batch
    batch = batches_dict[batch_name] #{'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091', 'dg_086', 'dg_088']}
    # print()
    print(f'{n_current}/{n_count} - {batch}')
    n_current += 1

    #revert字典，将batch号加入df_3，获得dg和sub_batch_id的对应关系
    batch_revert = {}
    for k,v in batch.items():
      for i in v:
        batch_revert[i] = k
    df_3['batch_id'] = df_3['dg_id'].apply(lambda x: batch_revert[x])
    # display(df_3.sort_values(['batch_id','cg_id','dg_id']))

    #遍历sub_batch: 对每一个sub_batch，找到中离方案最优解
    temp_sub_batch_metric = 0 #用于不满足条件时尽早结束计算
    for batch_id in batch.keys(): #这里的batch_id是sub_batch_id
      # print(f'sub_batch = {batch_id}, iterate to find best dg_rotate_comb and best sheet_size')
      #获得数据子集
      df_i = df_3[df_3['batch_id']==batch_id].sort_values(['dg_id']) #按照dg_id排序 - 这个很重要，保证所有数据的对应性
      # display(df_i)
      # #过滤不符合color limit的batch - filter batch时已考虑
      cg_id = df_i['cg_id'].values.tolist() #cg相同的必须相邻
      # if len(set(cg_id))>n_color_limit: #这里可以优化代码效率，因为目前是算到color大于limit的sub_batch才会break, 前面的sub_batch还是被计算了
      #   print(f'ERROR: nunique_color > {n_color_limit}, skip this case')
      #   break_flag = 1
      #   break
      #准备输入数据
      dg_id = df_i['dg_id'].values.tolist()
      fix_orientation = df_i['fix_orientation'].values.tolist()
      label_width = df_i['overall_label_width'].values.tolist()
      label_length = df_i['overall_label_length'].values.tolist()
      re_qty = df_i['re_qty'].values.tolist()
      # print(f'dg_id = {dg_id}，re_qty = {re_qty}')
      # print(cg_id, dg_id, fix_orientation, label_width, label_length, re_qty)

      #穷举该sub_batch所有rotation可能性的组合
      comb_names, comb_res_w, comb_res_h = get_all_dg_combinations_with_orientation(dg_id,fix_orientation,label_width,label_length)
      # print(f'comb_names = {comb_names}')
      # print(f'comb_res_w = {comb_res_w}')
      # print(f'comb_res_h = {comb_res_h}')  #check w和h的对应关系

      #遍历所有comb和sheet_size，选择对于该sub_batch最优的sheet_size和rotation_comb
      #这里min_tot_area只是一个代称，其实指的是metric，不一定是基于面积
      best_comb, best_sheet, res, min_tot_area = iterate_to_solve_min_total_sheet_area(#sheet_size_list, 
                                                                                      comb_names, comb_res_w, comb_res_h, dg_id, cg_id, re_qty, 
                                                                                      dg_sku_qty_dict, params_dict
                                                                                      #  check_criteria=False
                                                                                      ) ###--->>>
      max_pds = np.max(res['pds']) #这里是基于sku的max_pds    
      sheet_name = str(int(best_sheet[0]))+'<+>'+str(int(best_sheet[1]))
      sheet_weight = params_dict['business_params']['criteria'][sheet_name]['weight']
      temp_sub_batch_metric += max_pds*sheet_weight
      # print(f'temp_sub_batch_metric={temp_sub_batch_metric}, min_tot_area={min_tot_area}')  

      if temp_sub_batch_metric>best_metric: #虽然还没有计算完,但是结果已经不可能更好
        break_flag = 1
        print('temp_sub_batch_metric>best_metric')
        break
      # print(f'****** best_comb={best_comb}, best_sheet={best_sheet}, best_res={res}, min_tot_area={min_tot_area}')  

      ###########################################################################################################################
      
      #sub_batch结果添加至res_3_2字典
      # batch_the_pds = np.ceil(np.sum(res['re_qty'])/np.sum(res['ups']))
      res_detail_3_2[batch_name][batch_id] = {'best_comb':best_comb, 'best_sheet':best_sheet, 'best_res':res, 'max_pds':max_pds, 
                                              # 'batch_the_pds':batch_the_pds, 
                                              'min_tot_area':min_tot_area}
      #{'b0': {'best_comb': 'dg_084_h<+>dg_087_w<+>dg_095_w<+>dg_098_w<+>dg_099_w', 'best_sheet': [678, 528], 'best_res': {'re_qty': [1275, 440, 5794, 4145, 690], 'ups': [26, 13, 112, 77, 22], 'pds': [50.0, 34.0, 52.0, 54.0, 32.0], 'n_rows': [13, 13, 14, 11, 11], 'n_cols': [2, 1, 8, 7, 2]}, 'max_pds': 54.0, 'min_tot_area': 19331136.0}, 'b1': {'best_comb': 'dg_093_w<+>dg_094_w', 'best_sheet': [522, 328], 'best_res': {'re_qty': [10638, 7934], 'ups': [88, 63], 'pds': [121.0, 126.0], 'n_rows': [8, 9], 'n_cols': [11, 7]}, 'max_pds': 126.0, 'min_tot_area': 21573216.0}}

    if break_flag == 1:
      continue

    #计算当前batch的指标, 更新最优指标
    res_batch = res_detail_3_2[batch_name]
    # 'batch_5': {
    # 'b0': {'best_comb': 'dg_084_w<+>dg_086_h', 'best_sheet': [678, 528], 'best_res': {'n_rows': [7, 8], 'n_cols': [3, 14], 'ups': array([ 21, 112]), 'pds': [61.0, 91.0]}, 'max_pds': 91.0, 'min_tot_area': 32576544.0}, 
    # 'b1': {'best_comb': 'dg_087_w<+>dg_088_w', 'best_sheet': [678, 528], 'best_res': {'n_rows': [13, 8], 'n_cols': [1, 14], 'ups': array([ 13, 112]), 'pds': [34.0, 99.0]}, 'max_pds': 99.0, 'min_tot_area': 35440416.0}
    # }
    metric = 0
    # temp_metric=0
    for k,v in res_batch.items():
      #考虑pds和sheet_weight
      # sheet = v['best_sheet']
      metric += v['min_tot_area']
      # print(f'temp_metric={temp_metric}, metric={metric}')  
    #再考虑版数和pds之间的权衡
    add_metric = len(res_batch)*add_pds_per_sheet
    metric += add_metric
    res_metric_3_2[batch_name] = metric

    if metric<best_metric:
      best_metric = metric
      best_index = batch_name
      best_batch = batch      
      best_res = res_batch

    print(f'metric for {batch_name} = {metric}; current best metric = {best_metric}, current best batch = {best_index}')  

  # print('-'*50)
  # print(res_detail_3_2)
  # print('-'*50)
  # print(res_metric_3_2)
  #---------------------------------------------------------------------------------------------------------------------------

  #判断是否停止
  agg_compute_seconds = (datetime.now()-start_time).seconds
  print(f"agg_compute_seconds = {agg_compute_seconds} seconds")
  if agg_compute_seconds>=algo_time_limit: #停止条件1 
    print(f"computed for {len(old_batches+batches)}/{len(batches_list)} batches")
    break

  #更新历史数据
  old_batches += batches
  if len(old_batches)>=len(batches_list): #停止条件2
    print(f"computed for ALL {len(old_batches)} batches")
    break

# COMMAND ----------

#original- 2.09 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot for the best batch

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

df_res['weight'] = 1
df_res.loc[df_res['Printing_area']=='522x328','weight'] = 0.5
df_res['weighted_pds'] = df_res['weight']*df_res['pds']

# COMMAND ----------

df_res.groupby(['batch_id']).agg({'weighted_pds':'max'}).reset_index()

# COMMAND ----------

metrics_3_3 = np.sum(df_res.groupby(['batch_id']).agg({'weighted_pds':'max'}).values)
# metrics_3_3 = np.sum(df_3_3_res.groupby(['sub_batch_id']).agg({'weighted_sku_pds':'max'}).values)
n_batch = df_res['batch_id'].nunique()
print(f'sum_pds = {metrics_3_3+7*n_batch}') #在只有一种sheet_size的情况下只看sum_pds

# COMMAND ----------

end_time = datetime.now()
print(start_time)
print(end_time)
print('running time =', (end_time-start_time).seconds, 'seconds')

# COMMAND ----------

# 0519 case - 269
# https://adb-8939684233531805.5.azuredatabricks.net/?o=8939684233531805#job/509730401455551/run/1

# COMMAND ----------

# 0319 case - 404
# https://adb-8939684233531805.5.azuredatabricks.net/?o=8939684233531805#job/389854053364790/run/1
