# Databricks notebook source
import re
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC - Data Cleaning

# COMMAND ----------

df = pd.read_csv('../input/HTL_input_0419.csv') ######

#去掉列名中的空格, 并转换为小写
remove_space_dict = {}
for col in df.columns:
  remove_space_dict[col] = col.replace(' ', '_').lower()
df.rename(columns=remove_space_dict, inplace=True)

#数据格式标准化
str_cols = ['color_group', 'dimension_group', 'header_variable_data|sku_value', 'item', 'job_number', 'rb']
int_cols = ['fix_orientation', 'group_sku', 'group_nato', 'sku_quantity', 'sku_seq', 're_qty']
double_cols = ['overall_label_length', 'overall_label_width']
for c in str_cols:
  df[c] = df[c].astype('str')
  df[c] = df[c].str.replace(' ', '_')
  df[c] = df[c].str.lower() 
for c in int_cols:
  df[c] = df[c].astype('int')    
for c in double_cols:
  df[c] = df[c].astype('double')  
  df[c] = round(df[c], 1)

#分组之前的最终结果
cols = ['color_group','dimension_group','fix_orientation','overall_label_width','overall_label_length','job_number','item',
        'sku_seq','header_variable_data|sku_value','group_nato','group_sku','re_qty']
df = df[cols].sort_values(cols) 
df = df.reset_index()
#增加分组id的列
df['dg_id'] = df['dimension_group']+'<+>rotate_'+df['fix_orientation'].astype('str')
df['cg_dg_id'] = df['color_group']+'<+>'+df['dg_id']
df['sku_id'] = df['item']+'<+>'+df['group_nato'].astype('str')+'<+>'+df['header_variable_data|sku_value']+'<+>'+df['group_sku'].astype('str')
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Main

# COMMAND ----------

from utils.one_color_one_dimension_solver import *
from utils.one_color_more_dimension_solver import *
from utils.metric import *
from utils.plot import *

# COMMAND ----------

#inputs
sheet_size_list = [[678,528], [582,482], [522,328]] #优先选择总面积小的sheet size

criteria = {'one_cg_one_dg':{'678<+>528':{'pds':200,'internal_date':999999},
                             '582<+>482':{'pds':200,'internal_date':999999},
                             '522<+>328':{'pds':200,'internal_date':999999},
                             }, 
            'one_cg_mul_dg':{'678<+>528':{'pds':200,'internal_date':5},
                             '582<+>482':{'pds':200,'internal_date':5},
                             '522<+>328':{'pds':200,'internal_date':5},
                             },
            'mul_cg_mul_dg':{'678<+>528':{'pds':80,'internal_date':3},
                             '582<+>482':{'pds':80,'internal_date':3},
                             '522<+>328':{'pds':100,'internal_date':3},
                             },
            }
print(criteria)

# COMMAND ----------

#初始化，保存最终的batching信息
batch_res = {}
batch_index = 1 #第一个batch的index

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. One Color One Dimension

# COMMAND ----------

#以cg_dg_id分组
cols_to_first = ['overall_label_width', 'overall_label_length', 'fix_orientation']
agg_dict = {'re_qty':'sum'}
for c in cols_to_first:
  agg_dict[c] = 'first'
df_1 = df.groupby(['cg_dg_id']).agg(agg_dict).reset_index()
display(df_1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1  Filter
# MAGIC 过滤无论如何不可能满足criteria的id

# COMMAND ----------

#计算每一组cg_dg_rotation的最优sheet_size以及最优layout
id_list_1 = sorted(df_1['cg_dg_id'].tolist()) #所有candidate ids
print(f'candidate_id_list = {id_list_1}')

id_list_1_1 = [] #所有有可能满足criteria的ids
for cg_dg_rotate_id in id_list_1:
  label_width = df_1.loc[df_1['cg_dg_id']==cg_dg_rotate_id, 'overall_label_width']
  label_length = df_1.loc[df_1['cg_dg_id']==cg_dg_rotate_id, 'overall_label_length']
  re_qty = df_1.loc[df_1['cg_dg_id']==cg_dg_rotate_id, 're_qty']
  fail_criteria = filter_id_by_criteria_one_color_one_dimension(label_width, label_length, re_qty, sheet_size_list, criteria['one_cg_one_dg']) ###### --->>> 
  if fail_criteria == False:
    id_list_1_1.append(cg_dg_rotate_id)
print(f'qualified_id_list = {id_list_1_1}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2  Layout
# MAGIC 选择最优的sheet_size，并决定在该sheet_size上的layout

# COMMAND ----------

df_1_2 = df_1[df_1['cg_dg_id'].isin(id_list_1_1)] #筛选需要进行此计算步骤的id

# COMMAND ----------

#计算，寻找最优sheet_size并作layout
for cg_dg_rotate_id in id_list_1_1:
  print(f'------ calculating for {cg_dg_rotate_id} ------')
  df_temp = df_1[df_1['cg_dg_id']==cg_dg_rotate_id]
  label_size = [df_temp['overall_label_width'].values[0], df_temp['overall_label_length'].values[0]]
  fix_orientation = int(df_temp['fix_orientation'].values[0])
  print(f'label_size={label_size}, fix_orientation={fix_orientation}')
  re_qty = df_temp['re_qty'].values[0]
  best_sheet_size, res = sheet_size_selection_one_color_one_dimension(sheet_size_list, label_size, re_qty, fix_orientation) ######
  #formulate results
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'sheet_width'] = best_sheet_size[0]
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'sheet_length'] = best_sheet_size[1]
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'cg_dg_layout'] = str(res)
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'ups'] = res['ups']
  df_1_2.loc[df_1_2['cg_dg_id']==cg_dg_rotate_id,'pds'] = res['pds'] 

# df_1_2['sheet_size'] = df_1_2['sheet_width'].astype('int').astype('str')+'<+>'+df_1_2['sheet_length'].astype('int').astype('str')
display(df_1_2)

# COMMAND ----------

#根据criteria再次确认结果
df_1_2['pds_check'] = df_1_2.apply(lambda x: check_criteria_one_color_one_dimension(x, criteria['one_cg_one_dg']), axis=1)
df_1_2['checkpoint_1.1'] = df_1_2['pds_check']
display(df_1_2)

# COMMAND ----------

#把df拆分为两部分，第一部在one_cg_ong_dg做sku分配，第二部分做one_cg_mul_dg
df_1_2 = df_1_2[df_1_2['checkpoint_1.1']==True]
display(df_1_2)
ids_1_2 = list(df_1_2['cg_dg_id'].unique())
ups_1_2_dict = dict(zip(df_1_2['cg_dg_id'], df_1_2['ups'].astype('int')))
print('ups: ', ups_1_2_dict)

# COMMAND ----------

#保存结果
for id in ids_1_2:
  batch_res[batch_index] = [id]
  batch_index += 1
print(batch_res)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.3 allocate SKU

# COMMAND ----------

df_1_3 = df[df['cg_dg_id'].isin(ids_1_2)]
cols = ['cg_dg_id','sku_id','re_qty']
df_1_3 = df_1_3[cols]
display(df_1_3.sort_values(['cg_dg_id','sku_id']))

# COMMAND ----------

for cg_dg_id in ids_1_2:
  print(f'------ sku allocation for {cg_dg_id} ------')
  res_1_3 = allocate_sku_one_color_one_dimension(dict(zip(df_1_3['sku_id'], df_1_3['re_qty'].astype('int'))), ups_1_2_dict[cg_dg_id]) ###### --->>>
  print(res_1_3)
  con_1 = df_1_3['cg_dg_id']==cg_dg_id
  for sku_id, sku_res_dict in res_1_3.items():
    con_2 = df_1_3['sku_id']==sku_id
    df_1_3.loc[con_1&con_2, 'sku_ups'] = sku_res_dict['ups']
    df_1_3.loc[con_1&con_2, 'sku_pds'] = sku_res_dict['pds']
display(df_1_3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4 Plot

# COMMAND ----------

display(df_1_2)

# COMMAND ----------

# 1.3.1 基于df_1_1画layout
for cg_dg_id in ids_1_2:
  print(f'------ plot ups for {cg_dg_id} ------')
  df_temp = df_1_2[df_1_2['cg_dg_id']==cg_dg_id]
  label_width = df_temp['overall_label_width'].values[0]
  label_length = df_temp['overall_label_length'].values[0]
  sheet_width = df_temp['sheet_width'].values[0]
  sheet_length = df_temp['sheet_length'].values[0]  
  layout_dict = eval(df_temp['cg_dg_layout'].values[0])
  plot_ups_one_color_one_dimension([label_width, label_length], [sheet_width, sheet_length], layout_dict) ###### --->>>

# COMMAND ----------

#基于df_1_2画sku allocation - to do

# COMMAND ----------

#1.5 ABC版的处理

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. One Color More Dimension

# COMMAND ----------

# MAGIC %md
# MAGIC 简化：同一个cg_id，不管下面有多少种dimensions，都group在一起

# COMMAND ----------

df_1['cg_id'] = df_1.apply(lambda x: x.cg_dg_id.split('<+>')[0], axis=1)
df_2 = df_1[~df_1['cg_dg_id'].isin(ids_1_2)]
df_2 = df_2[['cg_id','cg_dg_id']+cols_to_first+['re_qty']]
df_2['label_area'] = df_2['overall_label_width']*df_2['overall_label_length']
# display(df_2)

#过滤：只考虑有多行的cg_id
df_2_cg_count = df_2.groupby('cg_id')['cg_dg_id'].count().reset_index().sort_values('cg_dg_id', ascending=False)
multi_dg_cg = df_2_cg_count[df_2_cg_count['cg_dg_id']>1]['cg_id'].values
df_2 = df_2[df_2['cg_id'].isin(multi_dg_cg)].sort_values(['cg_id', 're_qty'], ascending=[True, False])
# print(multi_dg_cg)
# display(df_2_cg_count)
display(df_2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Filter

# COMMAND ----------

print(f"candidate_id_list = {df_2['cg_id'].values}")
id_list_2_1 = [] #所有有可能满足criteria的ids
for cg_id in multi_dg_cg:
  # label_width = df_1.loc[df_1['cg_dg_id']==cg_dg_rotate_id, 'overall_label_width']
  # label_length = df_1.loc[df_1['cg_dg_id']==cg_dg_rotate_id, 'overall_label_length']
  # re_qty = df_1.loc[df_1['cg_dg_id']==cg_dg_rotate_id, 're_qty']
  fail_criteria = filter_id_by_criteria_one_color_more_dimension(df_2, cg_id, sheet_size_list, criteria['one_cg_mul_dg']) ###### --->>> 
  if fail_criteria == False:
    id_list_2_1.append(cg__id)
print(f'qualified_id_list = {id_list_2_1}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2 Layout

# COMMAND ----------

df_2_1 = df_2[df_2['cg_id'].isin(id_list_2_1)]
cg_ids_2_1 = df_2_1['cg_id']
print(len(cg_ids_2_1))
if len(cg_ids_2_1)>0:
  display(df_2_1)
else:
  print('no candidate id to layout')

# COMMAND ----------

# 必须列对齐
# 难点：同一个color，不同几个dimensions，怎么layout

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3 Allocate SKU

# COMMAND ----------

#

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4 Plot

# COMMAND ----------

# plot_one_color(all_rects)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. More Corlor More Dimension

# COMMAND ----------

from utils.more_color_more_dimension_solver import *

# COMMAND ----------

# 决策变量：
# sheet_size ------ 简化：暂时先只考虑一个sheet size
# 哪些id要batch在一起

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

#筛选需要继续的数据
df_3 = df_1[(~df_1['cg_dg_id'].isin(ids_1_2))&((~df_1['cg_id'].isin(id_list_2_1)))]
print(f"unique cg_id = {len(df_3['cg_id'].unique())}")
cols = ['cg_id', 'cg_dg_id', 're_qty', 'overall_label_width', 'overall_label_length', 'fix_orientation']
df_3 = df_3[cols]
display(df_3)

# COMMAND ----------

#简化：暂时先只考虑一个sheet size
df_3['dg_id'] = df_3['cg_dg_id'].apply(lambda x: x.split('<+>')[1])
df_3['sheet_width'] = sheet_size_list[0][0]
df_3['sheet_length'] = sheet_size_list[0][1]
display(df_3)

# COMMAND ----------

# # df_3['label_area'] = df_3['overall_label_width']*df_3['overall_label_length']
# pds_thres = criteria['mul_cg_mul_dg']['pds']
# print(f'mul_cg_mul_dg pds_thres = {pds_thres}')
# df_3['ups_upper_lim'] = df_3['re_qty']/pds_thres #要满足pds_thres，ups不能超过此值
# df_3['ups_upper_lim'] = df_3['ups_upper_lim'].astype('int')
# # df_3['area_upper_lim'] = df_3['ups_upper_lim']*df_3['label_area']
# # sheet_area = sheet_size_list[0][0]*sheet_size_list[0][1]
# # df_3['n_rows'] = round(df_3['area_upper_lim']/sheet_area,2)
# # # df_3['sheet_area_2'] = sheet_size_list[1][0]*sheet_size_list[1][1]
# # df_3['area_pct_1'] = round(df_3['area_upper_lim']/sheet_area,2)
# # # df_3['area_pct_2'] = round(df_3['area_upper_lim']/df_3['sheet_area_2'],2)
# # df_3 = df_3.sort_values(['cg_id', 'area_upper_lim'])
# display(df_3)

# COMMAND ----------

cg_agg_cnt = df_3.groupby('cg_id')['cg_dg_id'].agg('count').reset_index()
cg_agg_cnt = dict(zip(cg_agg_cnt['cg_id'],cg_agg_cnt['cg_dg_id']))
print(cg_agg_cnt)

# COMMAND ----------

# #处理有多于一个cg_dg_id的cg_id - 简单假设取width, length最大值，不太通用
# df_3['dummy_label_width'] = df_3.apply(lambda x: np.max([x['overall_label_width'],x['overall_label_length']]), axis=1)
# df_3['dummy_label_length'] = df_3.apply(lambda x: np.min([x['overall_label_width'],x['overall_label_length']]), axis=1)
# for cg_id, cnt in cg_agg_cnt.items():
#   if cnt>1:
#     #如果能旋转，统一设置为width长边，length短边
#     print(cg_id
#     )
#     fix_orientation = list(df_3[df_3['cg_id']==cg_id]['fix_orientation'])[0]
#     if fix_orientation == 0:
#       df_3.loc[df_3['cg_id']==cg_id, 'overall_label_width'] = df_3['dummy_label_width']
#       df_3.loc[df_3['cg_id']==cg_id, 'overall_label_length'] = df_3['dummy_label_length']

# agg_dict = {'re_qty':'sum','ups_upper_lim':'sum'}
# agg_dict['overall_label_width']='max'
# agg_dict['overall_label_length']='max'
# for col in [c for c in df_3.columns if c not in list(agg_dict.keys())+['cg_id','cg_dg_id']]:
#   agg_dict[col] = 'first'
# df_3_agg = df_3.groupby('cg_id').agg(agg_dict).reset_index()
# display(df_3_agg)

# COMMAND ----------


# df_3_agg['min_width'] = df_3_agg.apply(lambda x: get_min_width_for_one_color_one_dimension_for_one_sheetsize(x.overall_label_width, 
#                                                                                                       x.overall_label_length, 
#                                                                                                       x.sheet_width, 
#                                                                                                       x.sheet_length, 
#                                                                                                       x.ups_upper_lim, 
#                                                                                                       x.fix_orientation)['min_width'], axis=1)
# df_3_agg['is_rotated'] = df_3_agg.apply(lambda x: get_min_width_for_one_color_one_dimension_for_one_sheetsize(x.overall_label_width, 
#                                                                                                       x.overall_label_length, 
#                                                                                                       x.sheet_width, 
#                                                                                                       x.sheet_length, 
#                                                                                                       x.ups_upper_lim, 
#                                                                                                       x.fix_orientation)['is_rotated'], axis=1)
# df_3_agg['n_rows'] = df_3_agg.apply(lambda x: get_min_width_for_one_color_one_dimension_for_one_sheetsize(x.overall_label_width, 
#                                                                                                       x.overall_label_length, 
#                                                                                                       x.sheet_width, 
#                                                                                                       x.sheet_length, 
#                                                                                                       x.ups_upper_lim, 
#                                                                                                       x.fix_orientation)['n_rows'], axis=1)
# df_3_agg['n_cols'] = df_3_agg.apply(lambda x: get_min_width_for_one_color_one_dimension_for_one_sheetsize(x.overall_label_width, 
#                                                                                                       x.overall_label_length, 
#                                                                                                       x.sheet_width, 
#                                                                                                       x.sheet_length, 
#                                                                                                       x.ups_upper_lim, 
#                                                                                                       x.fix_orientation)['n_cols'], axis=1)
# df_3_agg['ups'] = df_3_agg['n_rows']*df_3_agg['n_cols']                                                                                              
# df_3_agg = df_3_agg.sort_values('min_width').reset_index()     
# display(df_3_agg)

# COMMAND ----------

# print(dict(zip(df_3_agg['cg_id'], df_3_agg['ups'])))

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
    # print(groups_list) 
    numgroups = numgroups + 1  

    ## 检查分组是否满足不同color不操作多少个dg的限制条件

    ## 将分组按要求格式输出
    batch = {}
    for i in range(len(groups_list)):
      b_key = 'b'+str(i)
      batch[b_key] = list(groups_list[i].keys()) 
    print(batch)
        
    ## 下一环节的确定sheetsize和各dg的ups

# COMMAND ----------

# # min_width_dict = dict(zip(df_3_agg['cg_id'], df_3_agg['min_width']))
# min_width_dict = {'cg_22': 663, 'cg_24': 616, 'cg_26': 420, 'cg_28': 224, 'cg_27': 224, 'cg_30': 213, 'cg_29': 168, 'cg_23': 29}
# sheet_index_dict = one_dim_packer_for_more_color_more_dimension_for_one_sheetsize(min_width_dict, sheet_size=sheet_size_list[0], ink_seperator=30)
# print(sheet_index_dict)

# COMMAND ----------

# df_agg = df_3.groupby(['cg_id'])['area_pct_1'].agg('sum').reset_index().sort_values('area_pct_1', ascending=False)
# display(df_agg)

# COMMAND ----------

# init_dict = dict(zip(df_agg['cg_id'], df_agg['area_pct_1']))
# init_dict

# COMMAND ----------

# temp_batch = {}
# candidate_ids = df_agg['cg_id'].values
# index = 0
# while len(candidate_ids) > np.sum([len(v) for v in temp_batch.values()]):
#   assigned_ids = []  
#   for v in temp_batch.values():
#     assigned_ids += v
#   non_assign_ids = [i for i in candidate_ids if i not in assigned_ids]
#   sum_area_pct = 0
#   for i in non_assign_ids:
#     sum_area_pct += 

# COMMAND ----------

# len(temp_batch.values())

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 layout

# COMMAND ----------

display(df_3)

# COMMAND ----------

#sample batch 输入
batch = {'b0':['dg_084','dg_086'],
        'b1':['dg_087','dg_088'],
        'b2':['dg_091','dg_093'],
        'b3':['dg_094','dg_095','dg_098','dg_099']}

ink_seperator_width = 30

# COMMAND ----------

batch_revert = {}
for k,v in batch.items():
  for i in v:
    batch_revert[i] = k
print(batch_revert)

# COMMAND ----------

df_3['batch_id'] = df_3['dg_id'].apply(lambda x: batch_revert[x])
display(df_3)

# COMMAND ----------

# MAGIC %md
# MAGIC 假设：每一个dg_id都排整列，同一列上不同dg_id不混排  
# MAGIC 决策：每一个dg是否旋转，每一个dg排几列  

# COMMAND ----------

import matplotlib.pyplot as plt
from utils.more_color_more_dimension_solver import get_all_dg_combinations_with_orientation, iterate_to_solve_min_total_sheet_area

# COMMAND ----------

for batch_id in batch.keys():
  print()
  print(f'iterate to find best dg_rotate_comb and best sheet_size for batch = {batch_id}')
  #准备数据
  df = df_3[df_3['batch_id']==batch_id]
  df = df.sort_values(['cg_id', 'dg_id'])
  display(df)
  cg_id = df['cg_id'].values.tolist() #cg相同的必须相邻
  dg_id = df['dg_id'].values.tolist()
  fix_orientation = df['fix_orientation'].values.tolist()
  label_width = df['overall_label_width'].values.tolist()
  label_length = df['overall_label_length'].values.tolist()
  re_qty = df['re_qty'].values.tolist()
  print(cg_id, dg_id, fix_orientation, label_width, label_length, re_qty)

  #穷举该batch在一个sheet中排列的所有组合，考虑rotation  
  #每一个comb都包含batch中的所有dg，区别在于dg是否rotate
  comb_names, comb_res_w, comb_res_h = get_all_dg_combinations_with_orientation(dg_id,fix_orientation,label_width,label_length)
  print(f'len(comb_names) = {len(comb_names)}')
  print(f'comb_names = {comb_names}')
  print(f'comb_res_w = {comb_res_w}')
  print(f'comb_res_h = {comb_res_h}')  #check w和h的对应关系

  #遍历所有comb和sheet_size，选择总耗材面积最小的comb+sheet_size的组合
  best_comb, best_sheet, res, min_tot_area = iterate_to_solve_min_total_sheet_area(sheet_size_list,comb_names,comb_res_w,comb_res_h,dg_id,cg_id,re_qty,ink_seperator_width)
  print(f'****** best_comb={best_comb}, best_sheet={best_sheet}, best_res={res}, min_tot_area={min_tot_area}')

  #plot上面的最优解
  scale = 50
  comb_index = comb_names.index(best_comb)
  label_w_list = comb_res_w[comb_index].split('<+>')
  label_w_list = [float(w) for w in label_w_list]
  label_h_list = comb_res_h[comb_index].split('<+>')
  label_h_list = [float(h) for h in label_h_list]
  ups_list = res['ups']
  n_cols = res['n_cols']
  plt.figure(figsize=(best_sheet[0]/scale, best_sheet[1]/scale))
  plot_full_height_for_each_dg_with_ink_seperator(best_sheet, ink_seperator_width, dg_id, cg_id, label_w_list, label_h_list, n_cols, ups_list)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3 allocate sku

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.4 plot

# COMMAND ----------


