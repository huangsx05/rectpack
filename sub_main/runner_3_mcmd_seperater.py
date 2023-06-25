import numpy as np
import pandas as pd
import random
from datetime import timedelta, datetime
from joblib import Parallel, delayed
from model.shared_solver import get_batches_heuristics_max_area_first, get_batches_heuristics_min_area_first, get_batches_with_filter, get_batches_by_partitions, filter_batches_with_criteria, calculate_one_batch
from utils.tools import calculate_best_batch

def runner_3_mcmd_seperator_sku_pds(start_time, params_dict, df, df_3):
  #数据接口
  df_3.rename(columns={'dimension_group':'dg_id'},inplace=True)

  #get params
  algo_time_limit = int(params_dict['algo_params']['algo_time_limit'])
  ink_seperator_width = int(params_dict['user_params']['ink_seperator_width'])
  n_abc = int(params_dict['user_params']['n_abc'])
  n_color_limit = np.max([int(v['n_color_limit']) for v in params_dict['user_params']['sheets'].values()]) #仅用于初筛，非最终值
  internal_days_limit = int(params_dict['user_params']['internal_days'])
  n_jobs = int(params_dict['algo_params']['n_jobs'])
  n_sample_per_job = int(params_dict['algo_params']['n_sample_per_job'])
  sample_batch_num = int(params_dict['algo_params']['sample_batch_num'])
  sample_batch_num = max(sample_batch_num, n_jobs*n_sample_per_job)
  print(f"sample_batch_num = {sample_batch_num }, n_jobs = {n_jobs}")
  
  #dg_sku_qty_dict
  dg_sku_qty_dict = {}
  for dg_name in df['dimension_group'].unique(): #在每一个dg内部分配做sku的ups分配
    df_i_sub = df[df['dimension_group']==dg_name]
    sku_qty_dict = dict(zip(df_i_sub['sku_id'],df_i_sub['re_qty']))
    dg_sku_qty_dict[dg_name] = sku_qty_dict
  # print(dg_sku_qty_dict) 

  #heuristics batching
  #为了后面的加速和标准化，这里给出的batches_list应该符合以下要求：
  #1) sub_batch按照长度降序; #2) dg_id升序
  heuristics_batches_list = []
  run_maxA = params_dict['algo_params']['heuristics_batches_max_area_first']
  run_minA = params_dict['algo_params']['heuristics_batches_min_area_first']

  if run_maxA=="True":
    print()
    print(f"[{datetime.now()}]: start heuristics_batches_MAX_area_first")  
    batches_list_maxA = get_batches_heuristics_max_area_first(df_3, params_dict, n_color_limit, internal_days_limit)
    # batches_list_maxA = filter_batches_with_criteria(batches_list_maxA, df_3, n_color_limit, internal_days_limit) #maxA自带filter
    heuristics_batches_list += batches_list_maxA
    print(f"filtered heuristics maxA batches # = {len(batches_list_maxA)}:")
    print(f"heuristics_batches_list # = {len(heuristics_batches_list)}:")    
    # for h in batches_list_maxA:
    #   print(h)

  if run_minA=="True":
    print()
    print(f"[{datetime.now()}]: start heuristics_batches_MIN_area_first")  
    initial_batches_list_minA = get_batches_heuristics_min_area_first(df_3, params_dict)
    batches_list_minA = filter_batches_with_criteria(initial_batches_list_minA, df_3, n_color_limit, internal_days_limit) #maxA不带filter
    heuristics_batches_list += [b for b in batches_list_minA if b not in heuristics_batches_list]    
    print(f"filtered heuristics minA batches # = {len(batches_list_minA)}:")
    print(f"heuristics_batches_list # = {len(heuristics_batches_list)}:")     
    # for h in batches_list_minA:
    #   print(h)

  #heuristics results
  # print(heuristics_batches_list[0])   
  best_metric = 1e12
  best_index = 0 #batch name
  best_batch = []
  best_res = {}

  res_list = []
  pre_n_count = 0
  n_count = len(heuristics_batches_list)
  print(f"heuristics batch sample = {heuristics_batches_list[0]}")
  heuristics_res = Parallel(n_jobs=n_jobs)(delayed(calculate_one_batch)(batch_i, pre_n_count, heuristics_batches_list, 
                                                                        df_3, best_metric, params_dict, dg_sku_qty_dict)
                                           for batch_i in range(pre_n_count, n_count))
  res_list.append(heuristics_res)
  # print(f"heuristics result sample:")
  # print(res_list[0])
  assessed_metrics, best_index, best_batch, best_res, best_metric = calculate_best_batch(res_list)
  print()
  print(f"****** heuristics results ******")
  # print(f"assessed_metrics={assessed_metrics}")  
  # print(f"best_res = {best_res}")      
  # print(f"best_index = {best_index}")
  print(f"best_batch = {best_batch}")
  print(f"best_metric = {best_metric}")  


  # Iterative Batching ===============================================================================================
  run_all = params_dict['algo_params']['iterate_all_batches']    
  if run_all=="True":
    print()
    print(f"[{datetime.now()}]: start generating all possible batches") 
    n_dg_thres = int(params_dict['algo_params']['iterate_n_dg_threshold'])
    n_cg_thres = int(params_dict['algo_params']['iterate_n_cg_threshold'])         
    batch_generate_mode = params_dict['algo_params']['all_batch_generate_mode']
    print(f"batch_generate_mode = {batch_generate_mode}")
    nunique_dg = df_3['dg_id'].nunique()
    nunique_cg = df_3['cg_id'].nunique()
    if nunique_dg>n_dg_thres:
      print(f"fail n_dg threshold!")
      run_all="False"
    elif nunique_cg>n_cg_thres:
      print(f"fail n_cg threshold!")    
      run_all="False"
    else:
      lower_sub_batch_num = len(best_batch) - 1
      upper_sub_batch_num = len(best_batch) - 1
      if batch_generate_mode=="combinations":
        batches_list = get_batches_with_filter(df_3, params_dict, lower_sub_batch_num, upper_sub_batch_num, n_color_limit, internal_days_limit)
      elif batch_generate_mode=="partitions":      
        initial_all_batches_list = get_batches_by_partitions(df_3, params_dict, lower_sub_batch_num, upper_sub_batch_num, n_color_limit)
        batches_list = filter_batches_with_criteria(initial_all_batches_list, df_3, n_color_limit, internal_days_limit) 
      else:
        print(f"undefined batch_generate_mode!!!")
      # batches_list = [b for b in batches_list if len(b)>0]
      # batches_list = [b for b in batches_list if b not in heuristics_batches_list]
      # print(f"all generated batches # = {len(all_batches_list)}:")
      print(f"iteration batch sample = {batches_list[0]}")      
      print(f"batches_list # = {len(batches_list)}:")     
      # for h in all_batches_list:
      #   print(h)   


  # #sample batch 输入
  # batches_list = [
  # {'b0': ['dg_10', 'dg_11', 'dg_12', 'dg_13'], 'b1': ['dg_02'], 'b2': ['dg_01', 'dg_04', 'dg_09'], 'b3': ['dg_03', 'dg_05', 'dg_08'], 'b4': ['dg_06', 'dg_07']} #276 -273
  # # # {'b0': ['dg_087', 'dg_098', 'dg_099'], 'b1': ['dg_088', 'dg_091'], 'b2': ['dg_084', 'dg_093'], 'b3': ['dg_094', 'dg_095'], 'b4': ['dg_086']}, # - 404 - 415
  # # # {'b0': ['dg_095', 'dg_098', 'dg_099'], 'b1': ['dg_087', 'dg_094'], 'b2': ['dg_084', 'dg_093'], 'b3': ['dg_088', 'dg_091'], 'b4': ['dg_086']} # - 405
  # ]
  # ppc_batch = [
  # # {'b0':['dg_01','dg_02','dg_03','dg_04'],'b1':['dg_05','dg_06','dg_07','dg_08','dg_09'],'b2':['dg_10'],'b3':['dg_11'],'b4':['dg_12','dg_13'] } #ppc solution - 0519
  # {'b0':['dg_084','dg_086'],'b1':['dg_087','dg_088'],'b2':['dg_091','dg_093'],'b3':['dg_094','dg_095','dg_098','dg_099']} #ppc solution -416- 0419
  # ]
  # batches_list = ppc_batch+batches_list+heuristics_batches_list
  # batches_list = [{'b0': ['dg_086', 'dg_087', 'dg_088', 'dg_094'], 'b1': ['dg_095', 'dg_098', 'dg_099'], 'b2': ['dg_093'], 'b3': ['dg_091'], 'b4': ['dg_084']}]


  #iterative results
  if run_all=="True":
    n_iteration = 0
    n_batch_max = len(batches_list)
    random.seed(1013)
    old_batches = []
    pre_n_count = 0
    n_count = 0
    while True: #时限未到
      print()
      #取样
      batches_list = [b for b in batches_list if b not in old_batches]
      print(f'[{datetime.now()}]: n_iteration = {n_iteration}, after dropping old batches, len(batches) = {len(batches_list)}')
      sample_batch_num = np.min([sample_batch_num,len(batches_list)])
      # sample_batch_num = 1
      batches = random.sample(batches_list, sample_batch_num)
      # batches = filter_batches_with_criteria(batches, df_3, n_color_limit, internal_days_limit)
      # print(batches[0])
      # for b in batches:
      #   print(b)    

      #转化batches输入为字典(列表转换为字典)
      # batches_dict = {}
      # for i in range(len(batches)):
      #   batch_name = 'batch_'+str(n_count+i)
      #   batches_dict[batch_name] = batches[i]

      #遍历batches找最优batch
      n_iteration += 1    
      n_count += len(batches)  
      # ======== Option 1 --------------------------------------------------------------------------------------------------------------------------
      # n_current, best_metric, best_index, best_batch, best_res = iterate_to_find_best_batch(batches_dict, df_3,
      #                                                                                       n_current, n_count, 
      #                                                                                       best_metric, best_index, best_batch, best_res,
      #                                                                                       params_dict, dg_sku_qty_dict)

      # ======== Option 2 ===========================================================
      # add_pds_per_sheet = params_dict['user_params']['add_pds_per_sheet']
      # pre_n_count = n_count-len(batches)
      # for batch_i in range(pre_n_count, n_count):
      #   print(batch_i)
      #   best_metric, temp_res = calculate_one_batch(batch_i, batches_dict, df_3, 
      #                                               best_metric, params_dict, dg_sku_qty_dict)
      #   res_list.append(temp_res)

      # best_metric, temp_res = calculate_one_batch(0, pre_n_count, batches, df_3, best_metric, params_dict, dg_sku_qty_dict)
      # print(f"best_metric={best_metric}")
      # print(f"temp_res={temp_res}")
      # stop

      # ======== Option 3 Parallel Computation ===================================================
      pre_n_count = n_count-len(batches)
      temp_res_list = []
      print(f"best_metric = {best_metric}")
      print(f"best_batch = {best_batch}")    
      temp_res = Parallel(n_jobs=n_jobs)(delayed(calculate_one_batch)(batch_i, pre_n_count, batches, df_3, best_metric, params_dict, dg_sku_qty_dict) 
                                        for batch_i in range(pre_n_count, n_count))
      temp_res_list.append(temp_res)
      temp_assessed_metrics, temp_best_index, temp_best_batch, temp_best_res, temp_best_metric = calculate_best_batch(temp_res_list)
      if temp_best_metric<best_metric:
        # best_index = temp_best_index 
        best_batch = temp_best_batch 
        best_res = temp_best_res 
        best_metric = temp_best_metric
      # ======================================================================  

      #判断是否停止
      agg_compute_seconds = (datetime.now()-start_time).seconds
      print(f"agg_compute_seconds = {agg_compute_seconds}/{algo_time_limit} seconds")
      if agg_compute_seconds>=algo_time_limit: #停止条件1 
        print(f"computed for {len(old_batches+batches)}/{n_batch_max} batches")
        break

      #更新历史数据
      old_batches += batches
      if len(old_batches)>=n_batch_max: #停止条件2
        print(f"computed for ALL {len(old_batches)} batches")
        break
    print(f"[{datetime.now()}]: break iteration, start to prepare results")
  else:
    print(f"[{datetime.now()}]: skip iteration")

  ############################################################################################

  #整理结果，找出最优batch
  # print(f"assessed_metrics={assessed_metrics}")  
  print(f"******FINAL RESULTS******")
  # print(f"best_index={best_index}")
  print(f"best_res={best_res}")    
  print(f"best_batch={best_batch}")
  print(f"best_metric={best_metric}")  

  return best_index, best_batch, best_res