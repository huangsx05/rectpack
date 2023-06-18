import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

# from utils.load_data import agg_to_get_dg_level_df
from utils.plot import plot_full_height_for_each_dg_with_ink_seperator
from model.shared_solver import get_batches_with_filter, iterate_to_find_best_batch, split_abc_ups

def runner_3_mcmd_seperator_sku_pds(params_dict, df, df_3):
  start_time = datetime.now()
  print(start_time)  

  # 当前notebook会用到的params，其他的会再调用函数中直接传入params_dict
  algo_time_limit = params_dict['algo_params']['algo_time_limit']
  ink_seperator_width = params_dict['user_params']['ink_seperator_width']
  n_abc = params_dict['user_params']['n_abc']
  n_abc = int(n_abc)  
  n_color_limit_list = [v['n_color_limit'] for v in params_dict['user_params']['sheets'].values()]
  n_color_limit = np.max(n_color_limit_list) #用于初筛batches
  sample_batch_num = params_dict['algo_params']['sample_batch_num']

  #准备sku level的dict
  dg_sku_qty_dict = {}
  for dg_name in df['dimension_group'].unique(): #在每一个dg内部分配做sku的ups分配
    df_i_sub = df[df['dimension_group']==dg_name]
    sku_qty_dict = dict(zip(df_i_sub['sku_id'],df_i_sub['re_qty']))
    dg_sku_qty_dict[dg_name] = sku_qty_dict
  # print(dg_sku_qty_dict) 

  #数据接口
  df_3.rename(columns={'dimension_group':'dg_id'},inplace=True)
  # display(df_3)
  cg_agg_cnt = df_3.groupby('cg_id')['cg_dg_id'].agg('count').reset_index()
  cg_agg_cnt = dict(zip(cg_agg_cnt['cg_id'],cg_agg_cnt['cg_dg_id']))
  # print("dg count for each color group:")
  # print(cg_agg_cnt)

  ###Batching
  # print(f"batch_generate_mode = {params_dict['algo_params']['batch_generate_mode']}")
  batches_list = get_batches_with_filter(df_3, params_dict, n_color_limit)  
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

  ###Layout
  old_batches = []
  n_count = 0 #已取样数
  n_current = 0 #已计算数

  best_metric = 1e12
  best_index = 0 #batch name
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
    n_current, best_metric, best_index, best_batch, best_res = iterate_to_find_best_batch(batches_dict, df_3,
                                                                                          n_current, n_count, 
                                                                                          best_metric, best_index, best_batch, best_res,
                                                                                          params_dict, dg_sku_qty_dict)
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
    
  print(best_index, best_res)
  return best_index, best_batch, best_res