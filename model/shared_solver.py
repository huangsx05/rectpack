import numpy as np
from datetime import datetime
from utils.tools import get_all_dg_combinations_with_orientation
from model.ocmd_solver import get_ups_layout_for_ocmd_comb_on_one_sheetsize
from model.mcmd_solver import get_n_cols_for_dg_comb_on_one_sheetsize


# ------ for batching ------

def get_batches_with_filter(df_3, params_dict, n_color_limit):
  #get params
  # print(f'[{datetime.now()}] get_batches_by_sampling')    
  N = df_3['dg_id'].nunique() #dg数量
  dg_sorted_list = sorted(df_3['dg_id'].tolist())
  dg_cg_dict = dict(zip(df_3['dg_id'].tolist(), df_3['cg_id'].tolist()))
  n_grp_lower = int(np.ceil(df_3['cg_id'].nunique()/n_color_limit)) #按照颜色数量决定sub_batch数量下限

  # sample_batch = params_dict['algo_params']['sample_batch'] #true/false
  # sample_batch_num = params_dict['algo_params']['sample_batch_num'] #考虑做成动态调整,并考虑在时间允许的范围内loop
  batch_generate_mode = params_dict['algo_params']['batch_generate_mode']
  if batch_generate_mode=='lower_upper_bound': #依赖于参数输入
    upper_sub_batch_num = params_dict['algo_params']['upper_sub_batch_num'] #考虑做成动态调整
    lower_sub_batch_num = params_dict['algo_params']['lower_sub_batch_num'] #考虑做成动态调整
  elif batch_generate_mode=='heuristics_sub_batch_number': #有很大的提升空间
    upper_sub_batch_num = int(df_3['cg_id'].nunique()/1.6)
    lower_sub_batch_num = upper_sub_batch_num
  # print("sample_batch, sample_batch_num, upper_sub_batch_num, lower_sub_batch_num")
  # print(sample_batch, sample_batch_num, upper_sub_batch_num, lower_sub_batch_num)
  len_lower_limit = max(n_grp_lower,lower_sub_batch_num) #sub_batch数量满足下限
  M = min(upper_sub_batch_num,N)  #dg分组数量上限  

  #get batches
  print(f'[{datetime.now()}] generate all samples')    
  # print(f'n_grp_lower={n_grp_lower}')
  batches_list = [] #存储待进行计算的batches(过滤不合格的batches之后)
  v_set_list = []   #for去重
  # combination_list = [] #存储candidate batches(过滤batches之前)，存放的是index集合
  #generate all possible combinations
  for n in range(N**M): #所有可能的组合的个数为N**M
    combination = [[] for __ in range(M)] #初始化, M是sub_batch数量
    for i in range(N):
      combination[n // M**i % M].append(i) 
    combination = [c for c in combination if len(c)>0] #这里的c应该是排好序的
    combination = sorted(combination,key = lambda i:len(i),reverse=True) #combination按照sub_batch长度排序,利于快速筛除颜色过多的batch
    #过滤掉sub_batch数量过少的batches    
    if len(combination)>=len_lower_limit:
      #去重 
      v_set = set([str(c) for c in combination])  
      # v_set = set(combination)       
      if v_set not in v_set_list:
        v_set_list.append(v_set)      
        # combination_list.append(combination)
        #去掉颜色数大于limit的sub_batch   
      # print(f'[{datetime.now()}] filter out n_color> batches')  
      # for combination in combination_list: #一个combination对应一个batch
        #将index变成dg_id
        # batch = []
        batch_dict = {}
        for index in range(len(combination)): #遍历sub_batch
        # for c in combination: #c is index list
          # if len(c)>0:
          sub_batch = [dg_sorted_list[i] for i in combination[index]]
          # batch.append(sub_batch)
        # if len(batch)>=max(n_grp_lower,lower_sub_batch_num): #sub_batch数量满足下限
        # if len(batch)==M:    
        #去掉颜色数大于limit的sub_batch    
        # for sub_batch in batch:
          colors = [dg_cg_dict[s] for s in sub_batch]
          if len(set(colors))>n_color_limit:      
            break
          else:
            # batch_dict = {}
            # for i in range(len(batch)):
            b_key = 'b'+str(index)
            batch_dict[b_key] = sub_batch      
            # #去重 
            # v_set = set([str(i) for i in batch_dict.values()])  
            # if v_set not in v_set_list:
            #   v_set_list.append(v_set)
        if len(batch_dict)==len(combination): #没有sub_batch因为n_color>limit
          batches_list.append(batch_dict)

  print(f"n_dg={N}, sub_batch # limit={M}, all possible combination # = {N**M}")  
  print(f"filtered batches, # = {len(batches_list)}")    

  return batches_list


# ------ for UPS Layout ------

def get_best_sheetsize_for_one_dg_comb(dg_id,cg_id,label_w_list,label_h_list,re_qty,
                                       dg_sku_qty_dict, params_dict):
  """
  遍历所有sheet_size，依据目标最小的原则选择最优的sheet_size
  """
  mode = params_dict['algo_params']['layout_mode']
  sheet_size_list = params_dict['user_params']['sheet_size_list']
  criteria_dict = params_dict['user_params']['sheets']

  min_tot_area = 1e12 #这里min_tot_area只是一个代称，其实指的是metric，不一定是基于面积   
  best_sheet = 'dummy'
  best_res = {}

  #遍历sheet_size
  for sheet_size in sheet_size_list:
    # print(f'sheet_size={sheet_size}')
    #get sheet_weight
    sheet_name = str(int(sheet_size[0]))+'<+>'+str(int(sheet_size[1]))
    sheet_weight = criteria_dict[sheet_name]['weight']
    n_color_limit = criteria_dict[sheet_name]['n_color_limit']
    if len(set(cg_id))>n_color_limit: #这里可以优化代码效率，因为目前是算到color大于limit的sub_batch才会break, 前面的sub_batch还是被计算了
        print(f'ERROR: nunique_color > {n_color_limit}, skip this case for {sheet_name}')
        continue

    if mode=='one_dg_one_column': #for mcmd中离
      #这里的pds_list应该是基于sku颗粒度判断的结果
      can_layout, n_rows, n_cols, ups_list, pds_list = get_n_cols_for_dg_comb_on_one_sheetsize(dg_id,cg_id,label_w_list,label_h_list,re_qty,sheet_size,
                                                                                               dg_sku_qty_dict,params_dict) ###--->>>
      if can_layout==False:
        continue
    elif mode=='mul_dg_one_column': #for ocmd(同颜色，无需中离)
      # print(dg_id,label_w_list,label_h_list,re_qty,sheet_size)
      ups_list, pds_list, dg_layout_seq, ups_info_by_col = get_ups_layout_for_ocmd_comb_on_one_sheetsize(dg_id,label_w_list,label_h_list,re_qty,sheet_size,params_dict) ###--->>>
      print(1/0, 'to include sku allocation')      
    else:
      print(f'unrecognized layout_mode = {mode}')   
      stop_flag = 1/0       
    # print(f'dg_id={dg_id}, re_qty={re_qty}, ups_list={ups_list}')

    #判断当前结果是否更优
    metric = np.max(pds_list)
    tot_area = metric*sheet_weight
    # print(f'sheet_size={sheet_size}, metric={tot_area}')    
    if tot_area<min_tot_area:
      min_tot_area = tot_area
      best_sheet = sheet_size
      best_res['re_qty'] = re_qty      
      best_res['ups'] = ups_list
      best_res['pds'] = pds_list
      if mode=='one_dg_one_column': 
        best_res['n_rows'] = n_rows
        best_res['n_cols'] = n_cols    
      elif mode=='mul_dg_one_column': 
        best_res['dg_layout_seq'] = dg_layout_seq
        best_res['ups_info_by_col'] = ups_info_by_col              
  # print(f'best_sheet={best_sheet}, best_res={best_res}, min_tot_area={min_tot_area}')
  return best_sheet, best_res, min_tot_area


def iterate_to_solve_min_total_sheet_area(comb_names, comb_res_w, comb_res_h, dg_id, cg_id, re_qty, dg_sku_qty_dict, params_dict):
  """
  main
  遍历所有dg_rotation_comb和sheet_size，选择总面积最小的comb+sheet_size的组合
  该函数对于不同的mode完全相同，区别仅在于mode的输入
  """
  criteria_dict = params_dict['user_params']['sheets']
  # ink_seperator_width = params_dict['business_params']['ink_seperator_width']
  # mode = params_dict['algo_params']['layout_mode']
  # layout_tolerance = params_dict['algo_params']['layout_tolerance']

  #初始化
  best_comb = 'dummy'
  best_sheet = [999,999]
  min_tot_area = 1e12 #这里min_tot_area只是一个代称，其实指的是metric，不一定是基于面积   
  best_res = {}

  #遍历所有dg_rotation_comb
  for comb_index in range(len(comb_names)):
    #准备输入数据
    comb_name = comb_names[comb_index]
    # print(f'{comb_name} - iterate_to_solve_min_total_sheet_area for comb')
    label_w_list = comb_res_w[comb_index].split('<+>')
    label_w_list = [float(w) for w in label_w_list]
    label_h_list = comb_res_h[comb_index].split('<+>')
    label_h_list = [float(h) for h in label_h_list]

    #对当前comb，遍历所有sheet_size
    #这里tot_area只是一个代称，其实指的是metric，不一定是基于面积    
    sheet, res, tot_area = get_best_sheetsize_for_one_dg_comb(dg_id,cg_id,label_w_list,label_h_list,re_qty,
                                                              dg_sku_qty_dict, params_dict) ###--->>>

    #判断解是否符合criteria
    if params_dict['algo_params']['criteria_check']:
      sheet_size_name = str(int(sheet[0]))+'<+>'+str(int(sheet[1]))
      pds_lower_lim = criteria_dict[sheet_size_name]['pds']
      pds_value = np.max(res['pds'])
      if pds_value < pds_lower_lim: #fail criteria
        continue

    #如果是当前更优解，更新结果a
    if tot_area<min_tot_area:
      best_comb = comb_name
      best_sheet = sheet   
      min_tot_area = tot_area
      best_res = res    

  # print(f'best_comb={best_comb}, best_sheet={best_sheet}, best_res={best_res}, min_tot_area={min_tot_area}')
  return best_comb, best_sheet, best_res, min_tot_area#, best_sku_pds, df_sku_res


def iterate_to_find_best_batch(batches_dict, df_3,
                               n_current, n_count, 
                               best_metric, best_index, best_batch, best_res,
                               params_dict, dg_sku_qty_dict
                               ):
  # 当前函数会用到的params
  add_pds_per_sheet = params_dict['user_params']['add_pds_per_sheet']

  #sample batches_dict
  #{'batch_0': {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091', 'dg_086', 'dg_088']}, 
  # 'batch_1': {'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091'], 'b1': ['dg_086', 'dg_088']}}
  for i in range(len(batches_dict)):
    break_flag = 0 #用于控制结果不可能更优时退出当前batch
    #获得batch
    batch_name = 'batch_'+str(n_current)
    res_batch = {}
    batch = batches_dict[batch_name] #{'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091', 'dg_086', 'dg_088']}
    print(f'{n_current}/{n_count} - {batch}')
    n_current += 1

    #获得dg和sub_batch_id的对应关系
    batch_revert = {}
    for k,v in batch.items():
      for i in v:
        batch_revert[i] = k #dg和batch的对应关系
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

      #穷举该sub_batch所有rotation可能性的组合
      comb_names, comb_res_w, comb_res_h = get_all_dg_combinations_with_orientation(dg_id,fix_orientation,label_width,label_length)

      #遍历所有comb和sheet_size，选择对于该sub_batch最优的sheet_size和rotation_comb
      #这里min_tot_area只是一个代称，其实指的是metric，不一定是基于面积
      best_comb, best_sheet, res, min_tot_area = iterate_to_solve_min_total_sheet_area(comb_names, comb_res_w, comb_res_h, dg_id, cg_id, re_qty, 
                                                                                      dg_sku_qty_dict, params_dict
                                                                                      ) ###--->>>
      max_pds = np.max(res['pds']) #这里是基于sku的max_pds    
      sheet_name = str(int(best_sheet[0]))+'<+>'+str(int(best_sheet[1]))
      sheet_weight = params_dict['user_params']['sheets'][sheet_name]['weight']
      temp_sub_batch_metric += max_pds*sheet_weight

      if temp_sub_batch_metric>best_metric: #虽然还没有计算完,但是结果已经不可能更好
        break_flag = 1
        print('temp_sub_batch_metric>best_metric')
        break

      res_batch[batch_id] = {'best_comb':best_comb, 'best_sheet':best_sheet, 'best_res':res, 'max_pds':max_pds, 'min_tot_area':min_tot_area}

    if break_flag == 1:
      continue

    #计算当前batch的指标, 更新最优指标
    metric = 0
    # temp_metric=0
    for k,v in res_batch.items():
      #考虑pds和sheet_weight
      metric += v['min_tot_area']
    #再考虑版数和pds之间的权衡
    add_metric = len(res_batch)*add_pds_per_sheet
    metric += add_metric

    if metric<best_metric:
      best_metric = metric
      best_index = batch_name
      best_batch = batch      
      best_res = res_batch

    print(f'metric for {batch_name} = {metric}; current best metric = {best_metric}, current best batch = {best_index}')

  return n_current, best_metric, best_index, best_batch, best_res


def calulcate_one_batch(batches_dict, df_3, batch_i,
                        best_metric,
                        params_dict, dg_sku_qty_dict):
  # 当前函数会用到的params
  add_pds_per_sheet = params_dict['user_params']['add_pds_per_sheet']

  break_flag = 0 #用于控制结果不可能更优时退出当前batch

  #获得batch
  batch_name = 'batch_'+str(batch_i)
  res_batch = {}
  batch = batches_dict[batch_name] #{'b0': ['dg_087', 'dg_099', 'dg_084', 'dg_098', 'dg_095', 'dg_094', 'dg_093', 'dg_091', 'dg_086', 'dg_088']}

  #获得dg和sub_batch_id的对应关系
  batch_revert = {}
  for k,v in batch.items():
    for i in v:
      batch_revert[i] = k #dg和batch的对应关系
  df_3['batch_id'] = df_3['dg_id'].apply(lambda x: batch_revert[x])

  #遍历sub_batch: 对每一个sub_batch，找到中离方案最优解
  temp_sub_batch_metric = 0 #用于不满足条件时尽早结束计算
  for batch_id in batch.keys(): #这里的batch_id是sub_batch_id
    df_i = df_3[df_3['batch_id']==batch_id].sort_values(['dg_id']) #按照dg_id排序 - 这个很重要，保证所有数据的对应性
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

    #穷举该sub_batch所有rotation可能性的组合
    comb_names, comb_res_w, comb_res_h = get_all_dg_combinations_with_orientation(dg_id,fix_orientation,label_width,label_length)

    #遍历所有comb和sheet_size，选择对于该sub_batch最优的sheet_size和rotation_comb
    #这里min_tot_area只是一个代称，其实指的是metric，不一定是基于面积
    best_comb, best_sheet, res, min_tot_area = iterate_to_solve_min_total_sheet_area(comb_names, comb_res_w, comb_res_h, dg_id, cg_id, re_qty, 
                                                                                    dg_sku_qty_dict, params_dict
                                                                                    ) ###--->>>
    max_pds = np.max(res['pds']) #这里是基于sku的max_pds    
    sheet_name = str(int(best_sheet[0]))+'<+>'+str(int(best_sheet[1]))
    sheet_weight = params_dict['user_params']['sheets'][sheet_name]['weight']
    temp_sub_batch_metric += max_pds*sheet_weight

    if temp_sub_batch_metric>best_metric: #虽然还没有计算完,但是结果已经不可能更好
      break_flag = 1
      print('temp_sub_batch_metric>best_metric')
      break

    res_batch[batch_id] = {'best_comb':best_comb, 'best_sheet':best_sheet, 'best_res':res, 'max_pds':max_pds, 'min_tot_area':min_tot_area}

  #计算当前batch的指标, 更新最优指标
  metric = 0
  # temp_metric=0
  for k,v in res_batch.items():
    #考虑pds和sheet_weight
    metric += v['min_tot_area']
  #再考虑版数和pds之间的权衡
  add_metric = len(res_batch)*add_pds_per_sheet
  metric += add_metric

  return {batch_name:{'res':res_batch, 'metric':metric}}


# ------ for SKU Allocation ------


def split_abc_ups(sub_id, sub_id_colname, df, ups_dict):
  """
  #step 2: 做ABC版的ups分割（每个版的ups应该相等）split ABC sheets
  """  
  print(f'------ abc splitting for {sub_id} ------')
  sets = ['Set A Ups','Set B Ups','Set C Ups','Set D Ups','Set E Ups','Set F Ups','Set G Ups','Set H Ups'] #预设版数
  df_temp = df[df[sub_id_colname]==sub_id].reset_index().drop(columns=['index'])  
  # n = 1 #当前的ups倍数
  cur_set_index = 0
  # sum_pds = 0
  for i in range(len(df_temp)): #对每一个sku
    cur_ups_thres = (cur_set_index+1)*ups_dict[sub_id]
    sku_id = df_temp.loc[i,'sku_id']
    set_name = sets[cur_set_index]
    # print(cur_set_index, set_name)      
    # print(df_temp['cum_sum_ups'].values.tolist())      
    # print(df_temp.loc[i,'cum_sum_ups'],cur_ups_thres)
    if df_temp.loc[i,'cum_sum_ups']<=cur_ups_thres: #无需换版
      df.loc[df['sku_id']==sku_id, set_name] = df['sku_ups']
    else: #换到下一个版，当前sku需要分配到两个不同的版
      # sum_pds += df_temp.loc[i,'sku_pds']
      if i==0:
        pre_sku_ups = cur_ups_thres
      else:
        pre_sku_ups = cur_ups_thres - df_temp.loc[i-1,'cum_sum_ups']          
      df.loc[df['sku_id']==sku_id, set_name] = pre_sku_ups #pre sheet
      next_sku_ups = df_temp.loc[i,'sku_ups'] - pre_sku_ups
      # n += 1
      cur_set_index += 1  
      set_name = sets[cur_set_index]
      # print(cur_set_index, set_name)   
      df.loc[df['sku_id']==sku_id, set_name] = next_sku_ups #next_sheet       
  # sum_pds += df_temp['sku_pds'].values[-1]

  for set_name in sets:
    if set_name in df.columns:
      df.fillna(0,inplace=True)
      df_temp = df[df[sub_id_colname]==sub_id]
      print(f'sum_ups for {set_name} = {np.sum(df_temp[set_name])}') #确认每个set的ups相等

  df = df.sort_values([sub_id_colname,'sku_pds'])    
  return df