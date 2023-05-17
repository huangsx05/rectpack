# Databricks notebook source
import re
import yaml
import numpy as np
import pandas as pd

# COMMAND ----------

#self-written libraries, 按照字母顺序
from utils.plot import plot_ups_ocod, plot_full_height_for_each_dg_with_ink_seperator
from model.ocod_solver import filter_id_by_criteria_ocod, sheet_size_selection_ocod, check_criteria_ocod, allocate_sku_ocod
from model.ocmd_solver import filter_id_by_criteria_ocmd
from model.mcmd_solver import iterate_to_solve_min_total_sheet_area
from utils.load_data import load_and_clean_data
from utils.tools import get_all_dg_combinations_with_orientation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Input

# COMMAND ----------

dbutils.widgets.text("production_line", "panyu_htl", "production_line")
production_line = dbutils.widgets.get("production_line")
print(f'production_line = {production_line}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

config_file = f'../config/config_{production_line}.yaml'
print(f'config_file = {config_file}')

with open(config_file) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

for k,v, in config.items():
  print(f'{k}: {v}')

# COMMAND ----------

# MAGIC %md
# MAGIC # Main

# COMMAND ----------

#outputs
#初始化，保存最终的batching信息
batch_res = {}
batch_index = 1 #第一个batch的index

# COMMAND ----------

#通过config拿到输入，按照字母顺序排列
criteria = config['criteria']
input_file = config['input_file']
ink_seperator_width = config['ink_seperator_width']
OCOD_filter = config['OCOD_filter']
OCMD_filter = config['OCMD_filter']
sheet_size_list_ocod = config['sheet_size_list_ocod'] #优先选择总面积小的sheet size
sheet_size_list_ocmd = config['sheet_size_list_ocmd']
sheet_size_list_mcmd = config['sheet_size_list_mcmd']

# COMMAND ----------

#intput data
df = load_and_clean_data(input_file)
display(df)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. One Color One Dimension (OCOD)

# COMMAND ----------

sheet_size_list = sheet_size_list_ocod

# COMMAND ----------

#以cg_dg_id分组，其实应该以dg_id分组就可以，也就是说dg_id是cg_id的下一级
cols_to_first = ['cg_id', 'dimension_group', 'fix_orientation','overall_label_width', 'overall_label_length']
agg_dict = {'re_qty':'sum'}
for c in cols_to_first:
  agg_dict[c] = 'first'
df_1 = df.groupby(['cg_dg_id']).agg(agg_dict).reset_index()
display(df_1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1  OCOD Filter
# MAGIC 过滤无论如何不可能满足criteria的id  
# MAGIC 比较鸡肋，因为在这个阶段即使遍历情况也不多

# COMMAND ----------

id_list_1 = sorted(df_1['cg_dg_id'].tolist()) #所有candidate ids
print(f'candidate_id_list = {id_list_1}')

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
  print(f'qualified_id_list = {id_list_1_1}')
  #输出：id_list_1_1（有可能满足criteria，进入下一步的id列表）
else:
  id_list_1_1 = id_list_1.copy()

print(f'OCOD_filter = {OCOD_filter}')
print(f'qualified_id_list = {id_list_1_1}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2  OCOD Layout
# MAGIC 对于每一个ocod id, 选择最优的sheet_size，并决定在该sheet_size上的layout

# COMMAND ----------

df_1_2 = df_1[df_1['cg_dg_id'].isin(id_list_1_1)] #根据1.1的结果筛选需要进行此计算步骤的id

#寻找最优sheet_size并计算ups和pds
for cg_dg_rotate_id in id_list_1_1:
  # print(f'------ calculating for {cg_dg_rotate_id} ------')
  df_temp = df_1_2[df_1_2['cg_dg_id']==cg_dg_rotate_id]
  label_size = [df_temp['overall_label_width'].values[0], df_temp['overall_label_length'].values[0]]
  fix_orientation = int(df_temp['fix_orientation'].values[0])
  # print(f'label_size={label_size}, fix_orientation={fix_orientation}')
  re_qty = df_temp['re_qty'].values[0]
  best_sheet_size, res = sheet_size_selection_ocod(sheet_size_list, label_size, re_qty, fix_orientation) ###--->>>遍历sheet_size
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

#把df拆分为两部分，第一部在ocod做sku分配，第二部分做ocmd
df_1_2 = df_1_2[df_1_2['checkpoint_1.1']==True]
print('results for qualified ocod ids')
display(df_1_2)

# COMMAND ----------

#保存结果
pass_cg_dg_ids_1_2 = list(df_1_2['cg_dg_id'].unique())
for cg_dg_id in pass_cg_dg_ids_1_2:
  batch_res[batch_index] = [cg_dg_id]
  batch_index += 1
print(batch_res)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.3 allocate SKU

# COMMAND ----------

if len(pass_cg_dg_ids_1_2)==0:
  pass
else:
  #get data for this step
  df_1_3 = df[df['cg_dg_id'].isin(pass_cg_dg_ids_1_2)]
  cols = ['cg_dg_id','cg_id', 'dimension_group', 'fix_orientation','overall_label_width', 'overall_label_length','sku_id','re_qty']
  df_1_3 = df_1_3[cols]

  #allocate sku
  ups_1_2_dict = dict(zip(df_1_2['cg_dg_id'], df_1_2['ups'].astype('int')))
  # print('ups: ', ups_1_2_dict)
  for cg_dg_id in pass_cg_dg_ids_1_2:
    # print(f'------ sku allocation for {cg_dg_id} ------')
    res_1_3 = allocate_sku_ocod(dict(zip(df_1_3['sku_id'], df_1_3['re_qty'].astype('int'))), ups_1_2_dict[cg_dg_id]) ###### --->>>
    # print(res_1_3)
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

# 1.3.1 基于df_1_1画layout
for cg_dg_id in pass_cg_dg_ids_1_2:
  df_temp = df_1_2[df_1_2['cg_dg_id']==cg_dg_id]
  label_width = df_temp['overall_label_width'].values[0]
  label_length = df_temp['overall_label_length'].values[0]
  sheet_width = df_temp['sheet_width'].values[0]
  sheet_length = df_temp['sheet_length'].values[0]  
  layout_dict = eval(df_temp['cg_dg_layout'].values[0])
  print(f'------ plot ups for {cg_dg_id} ------')  
  print(layout_dict)  
  plot_ups_ocod([label_width, label_length], [sheet_width, sheet_length], layout_dict) ###### --->>>

# COMMAND ----------

#基于df_1_2画sku allocation - to do

# COMMAND ----------

#1.5 ABC版的处理 - TO DO

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. One Color More Dimension (OCMD)

# COMMAND ----------

sheet_size_list = sheet_size_list_ocmd

# COMMAND ----------

df_2 = df_1[~df_1['cg_dg_id'].isin(pass_cg_dg_ids_1_2)]
df_2 = df_2[['cg_dg_id']+cols_to_first+['re_qty']]

#过滤：只考虑有多行的cg_id，因为单行的已经在OCOD考虑
df_2_cg_count = df_2.groupby('cg_id')['cg_dg_id'].count().reset_index().sort_values('cg_dg_id', ascending=False)
multi_dg_cg = df_2_cg_count[df_2_cg_count['cg_dg_id']>1]['cg_id'].values
df_2 = df_2[df_2['cg_id'].isin(multi_dg_cg)].sort_values(['cg_id', 're_qty'], ascending=[True, False])
display(df_2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1 Filter

# COMMAND ----------

print(f"candidate_id_list = {df_2['cg_id'].values}")
id_list_2_1 = [] #所有有可能满足criteria的ids

if OCMD_filter:
  df_2['label_area'] = df_2['overall_label_width']*df_2['overall_label_length']
  for cg_id in multi_dg_cg:
    fail_criteria = filter_id_by_criteria_ocmd(df_2, cg_id, sheet_size_list, criteria['one_cg_mul_dg']) ###### --->>> 
    if fail_criteria == False:
      id_list_2_1.append(cg_id)
else:
  id_list_2_1 = df_2['cg_id'].values

print(f'OCMD_filter = {OCMD_filter}')
print(f'qualified_id_list = {id_list_2_1}')

df_2_1 = df_2[df_2['cg_id'].isin(id_list_2_1)]
if len(df_2_1)>0:
  display(df_2_1)
else:
  print('no candidate id to layout')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2 Layout

# COMMAND ----------

# MAGIC %md
# MAGIC 假设：  
# MAGIC 1. 同一个cg_id，不管下面有多少种dimensions，都group在一起  
# MAGIC 2. 每个DG排整列，同一列上不同DG不混排

# COMMAND ----------

cg_id_list_2_2 = sorted(df_2_1['cg_id'].unique())
print(f'candidate cd_ids = {cg_id_list_2_2}')

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

  #获取cg rotation的combinations
  batch_name = 'batch_ccmd<+>'+cg_id
  print(f'$$$$$$ calculating batch = {batch_name} $$$$$$')  
  dg_id = df_i['dimension_group'].values.tolist()
  fix_orientation = df_i['fix_orientation'].values.tolist()
  label_width = df_i['overall_label_width'].values.tolist()
  label_length = df_i['overall_label_length'].values.tolist()
  # print(dg_id,fix_orientation,label_width,label_length)
  comb_names, comb_res_w, comb_res_h = get_all_dg_combinations_with_orientation(dg_id,fix_orientation,label_width,label_length)
  # print(f'len(comb_names) = {len(comb_names)}')
  # print(f'comb_names = {comb_names}')
  # print(f'comb_res_w = {comb_res_w}')
  # print(f'comb_res_h = {comb_res_h}')  #check w和h的对应关系

  #遍历所有comb和sheet_size，选择总耗材面积最小的comb+sheet_size的组合
  cg_id = [cg_id]
  re_qty = fix_orientation = df_i['re_qty'].values.tolist()
  # print(f'sheet_size_list,comb_names,comb_res_w,comb_res_h,dg_id,cg_id,re_qty,ink_seperator_width = ')
  # print(sheet_size_list,comb_names,comb_res_w,comb_res_h,dg_id,cg_id,re_qty,ink_seperator_width)
  best_comb, best_sheet, res, min_tot_area = iterate_to_solve_min_total_sheet_area(sheet_size_list,comb_names,comb_res_w,comb_res_h,dg_id,cg_id,re_qty,ink_seperator_width,
                                                                                    check_criteria=True,criteria_dict=criteria['one_cg_mul_dg']) ###
  print(f'****** best_comb={best_comb}, best_sheet={best_sheet}, best_res={res}, min_tot_area={min_tot_area}')  

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

print('-'*50)
print(f'fail_cg_ids_2_2 = {fail_cg_ids_2_2}')
print(f'pass_cg_ids_2_2 = {pass_cg_ids_2_2}')
print(f'qualified_res_2_2 = {qualified_res_2_2}')

# COMMAND ----------

#保存结果
batch_res.update(qualified_res_2_2)
print(batch_res)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3 Allocate SKU

# COMMAND ----------

if len(pass_cg_ids_2_2)==0:
  pass
else:
  #get data for this step
  df_2_3 = df[df['cg_id'].isin(pass_cg_ids_2_2)]
  cols = ['cg_dg_id','cg_id', 'dimension_group', 'fix_orientation','overall_label_width', 'overall_label_length','sku_id','re_qty']
  df_2_3 = df_2_3[cols]
  # display(df_2_3)

  #allocate sku
  print('ups: ', ups_2_2_dict)
  for dg_id, ups in ups_2_2_dict.items():
    print(f'------ sku allocation for {dg_id} ------')
    df_2_3_i = df_2_3[df_2_3['dimension_group']==dg_id]
    res_2_3 = allocate_sku_ocod(dict(zip(df_2_3_i['sku_id'], df_2_3_i['re_qty'].astype('int'))), ups_2_2_dict[dg_id]) ###### --->>>
    print(res_2_3)
    con_1 = df_2_3['dimension_group']==dg_id
    for sku_id, sku_res_dict in res_2_3.items():
      con_2 = df_2_3['sku_id']==sku_id
      df_2_3.loc[con_1&con_2, 'sku_ups'] = sku_res_dict['ups']
      df_2_3.loc[con_1&con_2, 'sku_pds'] = sku_res_dict['pds']
  display(df_2_3)  

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4 Plot

# COMMAND ----------

# to do by frontend

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. More Corlor More Dimension

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
df_3 = df_1[(~df_1['cg_dg_id'].isin(pass_cg_dg_ids_1_2))&((~df_1['cg_id'].isin(pass_cg_ids_2_2)))]
print(f"unique cg_id = {len(df_3['cg_id'].unique())}")
cols = ['cg_id', 'dimension_group', 'cg_dg_id', 're_qty', 'overall_label_width', 'overall_label_length', 'fix_orientation']
df_3 = df_3[cols].rename(columns={'dimension_group':'dg_id'})
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

# MAGIC %md
# MAGIC #### 3.2 layout
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

batches = [
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091', 'dg_086', 'dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091'], 'b1': ['dg_086', 'dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091'], 'b1': ['dg_086', 'dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093'], 'b1': ['dg_091', 'dg_086'], 'b2': ['dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094'], 'b1': ['dg_093', 'dg_091'], 'b2': ['dg_086'], 'b3': ['dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094'], 'b1': ['dg_093', 'dg_091'], 'b2': ['dg_086'], 'b3': ['dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094'], 'b1': ['dg_093', 'dg_091'], 'b2': ['dg_086'], 'b3': ['dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095'], 'b1': ['dg_094', 'dg_093'], 'b2': ['dg_091'], 'b3': ['dg_086'], 'b4': ['dg_088']},
# {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095'], 'b1': ['dg_094', 'dg_093'], 'b2': ['dg_091'], 'b3': ['dg_086'], 'b4': ['dg_088']},
{'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095'], 'b1': ['dg_094', 'dg_093'], 'b2': ['dg_091'], 'b3': ['dg_086'], 'b4': ['dg_088']},
# {'b0':['dg_084','dg_086'],
#         'b1':['dg_087','dg_088'],
#         'b2':['dg_091','dg_093'],
#         'b3':['dg_094','dg_095','dg_098','dg_099']} #ppc solution
]

# COMMAND ----------

sheet_size_list = sheet_size_list_mcmd

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
                                                                                    check_criteria=False,criteria_dict=criteria['mul_cg_mul_dg']) ###
    print(f'****** best_comb={best_comb}, best_sheet={best_sheet}, best_res={res}, min_tot_area={min_tot_area}')  

    #结果添加至res_3_2字典
    max_pds = np.max(res['pds'])
    res_detail_3_2[batch_name][batch_id] = {'best_comb':best_comb, 'best_sheet':best_sheet, 'best_res':res, 'max_pds':max_pds, 'min_tot_area':min_tot_area}

  #计算当前batch的指标
  res_batch = res_detail_3_2[batch_name]
  # 'batch_5': {
  # 'b0': {'best_comb': 'dg_084_w<+>dg_086_h', 'best_sheet': [678, 528], 'best_res': {'n_rows': [7, 8], 'n_cols': [3, 14], 'ups': array([ 21, 112]), 'pds': [61.0, 91.0]}, 'max_pds': 91.0, 'min_tot_area': 32576544.0}, 
  # 'b1': {'best_comb': 'dg_087_w<+>dg_088_w', 'best_sheet': [678, 528], 'best_res': {'n_rows': [13, 8], 'n_cols': [1, 14], 'ups': array([ 13, 112]), 'pds': [34.0, 99.0]}, 'max_pds': 99.0, 'min_tot_area': 35440416.0}
  # }
  metric = 0
  for k,v in res_batch.items():
    metric += v['max_pds']
  print(f'****** sum_pds = {metric}')  
  res_metric_3_2[batch_name] = metric

print('-'*50)
print(res_detail_3_2)
print('-'*50)
print(res_metric_3_2)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot for the best batch

# COMMAND ----------

import matplotlib.pyplot as plt

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

print(best_batch)
print()
print(res)

# COMMAND ----------

df_3_3 = df[['dimension_group','sku_id','re_qty']]
df_3_3 = df_3_3.sort_values(list(df_3_3.columns))

#revert字典，将batch号加入df_3_3
batch_revert = {}
for k,v in batch.items():
  for i in v:
    batch_revert[i] = k
df_3_3 = df_3_3[df_3_3['dimension_group'].isin(batch_revert.keys())]
df_3_3['sub_batch_id'] = df_3_3['dimension_group'].apply(lambda x: batch_revert[x])

df_3_3 = df_3_3.sort_values(['sub_batch_id','dimension_group','sku_id','re_qty'])
display(df_3_3)

# COMMAND ----------

df_i_list = []

for sub_batch_id in best_batch.keys():
  #一个sub_batch的sku allocation
  df_i = df_3_3[df_3_3['sub_batch_id']==sub_batch_id]
  # display(df_i)
  # dg_id = best_batch[sub_batch_id] #不能用这个，顺序会不一样
  best_comb = res[sub_batch_id]['best_comb']
  dg_id = [i[:-2] for i in best_comb.split('<+>')]  
  dg_orient = [i[-1] for i in best_comb.split('<+>')]
  ups_list = list(res[sub_batch_id]['best_res']['ups'])
  # print()
  # print(f'dg_id = {dg_id}')
  # print(f'ups_list = {ups_list}')  
  # print()

  for sub_dg_index in range(len(dg_id)):
    sub_dg = dg_id[sub_dg_index]
    df_i_sub = df_i[df_i['dimension_group']==sub_dg]
    print(f'sub_dg = {sub_dg}')

    sku_qty_dict = dict(zip(df_i_sub['sku_id'],df_i_sub['re_qty']))
    n_ups = ups_list[sub_dg_index]
    res_dict = allocate_sku_ocod(sku_qty_dict, n_ups)
    # print(f'n_ups = {n_ups}')
    # print(f'res_dict = {res_dict}')

    for sku_id in res_dict.keys():
      df_i_sub.loc[df_i_sub['sku_id']==sku_id, 'sku_ups'] = res_dict[sku_id]['ups']
      df_i_sub.loc[df_i_sub['sku_id']==sku_id, 'sku_pds'] = res_dict[sku_id]['pds']  

    # print(f"sum_sku_ups = {np.sum(df_i_sub['sku_ups'])}")
    # print(f"max_sku_ups = {np.max(df_i_sub['sku_ups'])}")
    # display(df_i_sub)
    df_i_list.append(df_i_sub)

df_3_3_res = pd.concat(df_i_list).sort_values(['sub_batch_id','dimension_group','sku_id','re_qty'])
print(f"sum_sku_ups = {np.sum(df_3_3_res['sku_ups'])}")
print(f"max_sku_ups = {np.max(df_3_3_res['sku_ups'])}")
display(df_3_3_res)

# COMMAND ----------

display(df_3_3_res.groupby(['sub_batch_id','dimension_group']).agg({'sku_id':'count','re_qty':'sum','sku_ups':'sum','sku_pds':'max'}).reset_index())

# COMMAND ----------

metrics_3_3 = np.sum(df_3_3_res.groupby(['sub_batch_id']).agg({'sku_pds':'max'}).values)
print(f'sum_pds = {metrics_3_3}')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.4 plot

# COMMAND ----------

#to do
#no need for now

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

#初始化results data
df_res = df.copy()[['sku_id','cg_id','dimension_group','overall_label_width','overall_label_length','sku_quantity','re_qty']]
df_res.rename(columns={'dimension_group':'DG',
                      'sku_quantity':'Quantity',
                      're_qty':'Req_Qty',
                      'cg_id':'Color Group',
                      'overall_label_width':'OverLabel W',
                      'overall_label_length':'OverLabel L'},
                      inplace=True)
cols = ['batch_id','DG','orient','Printing_area','Quantity','Req_Qty','pds','ups','qty_sum','the_pds','overproduction','blank','Color Group','OverLabel W','OverLabel L','中离数']
for c in cols:
  if c not in df_res.columns:
    df_res[c] = None
df_res = df_res[cols]
display(df_res)

# COMMAND ----------


