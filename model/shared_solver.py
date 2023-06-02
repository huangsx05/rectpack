import numpy as np
import random
from model.ocmd_solver import get_ups_layout_for_ocmd_comb_on_one_sheetsize
from model.mcmd_solver import get_n_cols_for_dg_comb_on_one_sheetsize


# ------ for batching ------

def get_batches_by_sampling(df_3, params_dict, n_color_limit):
  #get params
  sample_batch = params_dict['algo_params']['sample_batch']
  sample_batch_num = params_dict['algo_params']['sample_batch_num'] #考虑做成动态调整
  upper_sub_batch_num = params_dict['algo_params']['upper_sub_batch_num'] #考虑做成动态调整
  lower_sub_batch_num = params_dict['algo_params']['lower_sub_batch_num'] #考虑做成动态调整
  print("sample_batch, sample_batch_num, upper_sub_batch_num, lower_sub_batch_num")
  print(sample_batch, sample_batch_num, upper_sub_batch_num, lower_sub_batch_num)

  #get batches
  N = df_3['dg_id'].nunique() #dg数量
  M = min(upper_sub_batch_num,N)  #dg分组数量上限  
  dg_sorted_list = sorted(df_3['dg_id'].tolist())
  dg_cg_dict = dict(zip(df_3['dg_id'].tolist(), df_3['cg_id'].tolist()))
  n_grp_lower = int(np.ceil(df_3['cg_id'].nunique()/n_color_limit)) #按照颜色数量决定sub_batch数量下限
  # print(f'n_grp_lower={n_grp_lower}')

  batches_list = [] #存储待进行计算的batches(过滤不合格的batches之后)
  v_set_list = []   #for去重
  combination_list = [] #存储candidate batches(过滤batches之前)，存放的是index集合
  #generate all possible combinations
  for n in range(N**M): #所有可能的组合的个数为N**M
    combination = [[] for __ in range(M)] #初始化
    for i in range(N):
      combination[n // M**i % M].append(i)
    combination_list.append(combination)
  print(f"all possible combination # = {len(combination_list)}")  

  #sampling - 这里怎么sample是一个可以改进的点，比如分层取样等
  if sample_batch:
    combination_list = random.sample(combination_list, sample_batch_num)
    print(f"after sampling combination # = {len(combination_list)}")

  #filter out unqualified batches
  for combination in combination_list: #一个combination对应一个batch
    #将index变成dg_id
    batch = []
    for c in combination: #c is index list
      if len(c)>0:
        sub_batch = [dg_sorted_list[i] for i in c]
        batch.append(sub_batch)
    if len(batch)>=max(n_grp_lower,lower_sub_batch_num): #sub_batch数量满足下限
    # if len(batch)==M:    
      #去掉颜色数大于limit的sub_batch    
      for sub_batch in batch:
        colors = [dg_cg_dict[s] for s in sub_batch]
        if len(set(colors))>n_color_limit:      
          break
        else:
          batch_dict = {}
          for i in range(len(batch)):
            b_key = 'b'+str(i)
            batch_dict[b_key] = batch[i]        
          #去重 
          v_set = set([str(i) for i in batch_dict.values()])  
          if v_set not in v_set_list:
            v_set_list.append(v_set)
            batches_list.append(batch_dict)
            # print(batch_dict)
  print(f"after filtering combination # = {len(batches_list)}")

  return batches_list


# ------ for UPS Layout ------

def get_best_sheetsize_for_one_dg_comb(dg_id,cg_id,label_w_list,label_h_list,re_qty,
                                       dg_sku_qty_dict, params_dict):
  """
  遍历所有sheet_size，依据目标最小的原则选择最优的sheet_size
  """
  mode = params_dict['algo_params']['layout_mode']
  sheet_size_list = params_dict['user_params']['sheet_size_list']
  criteria_dict = params_dict['business_params']['criteria']

  min_tot_area = 1e12 #这里min_tot_area只是一个代称，其实指的是metric，不一定是基于面积   
  best_sheet = 'dummy'
  best_res = {}

  #遍历sheet_size
  for sheet_size in sheet_size_list:
    # print(f'sheet_size={sheet_size}')
    #get sheet_weight
    sheet_name = str(int(sheet_size[0]))+'<+>'+str(int(sheet_size[1]))
    sheet_weight = criteria_dict[sheet_name]['weight']

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
  criteria_dict = params_dict['business_params']['criteria']
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