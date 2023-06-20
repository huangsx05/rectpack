import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

# from utils.load_data import agg_to_get_dg_level_df
from utils.plot import plot_full_height_for_each_dg_with_ink_seperator
from utils.tools import allocate_sku, get_all_dg_combinations_with_orientation
from model.shared_solver import get_batches_with_filter, iterate_to_solve_min_total_sheet_area, split_abc_ups

def runner_3_mcmd_seperator_sku_pds(params_dict, df, df_3):
  start_time = datetime.now()
  print(start_time)  

  # 当前notebook会用到的params，其他的会再调用函数中直接传入params_dict
  add_pds_per_sheet = params_dict['user_params']['add_pds_per_sheet']  
  algo_time_limit = params_dict['algo_params']['algo_time_limit']
  ink_seperator_width = params_dict['business_params']['ink_seperator_width']
  n_abc = params_dict['user_params']['n_abc']
  n_abc = int(n_abc)  
  n_color_limit = params_dict['business_params']['n_color_limit']
  internal_days_limit = params_dict['business_params']['internal_days_limit']
  # n_color = int(n_color)
  # sample_batch = params_dict['algo_params']['sample_batch'] #true/false  
  sample_batch_num = params_dict['algo_params']['sample_batch_num'] #考虑做成动态调整,并考虑在时间允许的范围内loop 

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
  batches_list = get_batches_with_filter(df_3, params_dict, n_color_limit, internal_days_limit)  
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
    
  print(best_index, best_res)
  return best_index, best_batch, best_res