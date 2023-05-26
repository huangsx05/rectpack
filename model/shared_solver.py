import numpy as np
from model.ocmd_solver import get_ups_layout_for_ocmd_comb_on_one_sheetsize
from model.mcmd_solver import get_n_cols_for_dg_comb_on_one_sheetsize


# ------ for UPS Layout ------

def get_best_sheetsize_for_one_dg_comb(sheet_size_list, dg_id,cg_id,label_w_list,label_h_list,re_qty,ink_seperator_width,
                                       criteria_dict,
                                       mode='one_dg_one_column'):
  """
  遍历所有sheet_size，依据目标最小的原则选择最优的sheet_size
  """
  min_tot_area = 1e12
  best_sheet = sheet_size_list[0]
  best_res = {}
  for sheet_size in sheet_size_list:
    # print(f'--- sheet_size={sheet_size} ---')
    if mode=='one_dg_one_column': #for mcmd中离
      n_rows, n_cols, ups_list, pds_list = get_n_cols_for_dg_comb_on_one_sheetsize(dg_id,cg_id,label_w_list,label_h_list,re_qty,sheet_size,ink_seperator_width) ###--->>>
    elif mode=='mul_dg_one_column': #for ocmd(同颜色，无需中离)
      # print(dg_id,label_w_list,label_h_list,re_qty,sheet_size)
      ups_list, pds_list, dg_layout_seq, ups_info_by_col = get_ups_layout_for_ocmd_comb_on_one_sheetsize(dg_id,label_w_list,label_h_list,re_qty,sheet_size) ###--->>>
    else:
      print(f'unrecognized OCMD_layout_mode = {OCMD_layout_mode}')   
      stop_flag = 1/0      
    batch_the_pds = np.ceil(np.sum(re_qty)/np.sum(ups_list))  
    # print(f'dg_id={dg_id}, re_qty={re_qty}, ups_list={ups_list}, batch_the_pds={batch_the_pds}')

    #get sheet_weight
    sheet_name = str(int(sheet_size[0]))+'<+>'+str(int(sheet_size[1]))
    sheet_weight = criteria_dict[sheet_name]['weight']

    #判断当前结果是否更优
    tot_area = batch_the_pds*sheet_weight ######根据ppc要求，用batch_the_pds而不是max_pds,并且要考虑和sheet_size相对应的weight
    # tot_area = sheet_size[0]*sheet_size[1]*batch_the_pds ######根据ppc要求，用batch_the_pds而不是max_pds,并且要考虑和sheet_size相对应的weight    
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


def iterate_to_solve_min_total_sheet_area(sheet_size_list,comb_names,comb_res_w,comb_res_h,dg_id,cg_id,re_qty,ink_seperator_width,
                                          check_criteria=False,criteria_dict=None,
                                          mode='one_dg_one_column'):
  """
  main
  该函数对于不同的mode完全相同，区别仅在于mode的输入
  """
  #初始化
  best_comb = 'dummy'
  best_sheet = [999,999]
  min_tot_area = 999999999999
  best_res = {'re_qty': [0, 0], 'n_rows': [0, 0], 'n_cols': [0, 0], 'ups': [0,  0], 'pds': [0, 0]}

  #遍历所有comb和sheet_size，选择总面积最小的comb+sheet_size的组合
  for comb_index in range(len(comb_names)):
    comb_name = comb_names[comb_index]
    print(f'----- iterate_to_solve_min_total_sheet_area for comb = {comb_name} ------')
    label_w_list = comb_res_w[comb_index].split('<+>')
    label_w_list = [float(w) for w in label_w_list]
    label_h_list = comb_res_h[comb_index].split('<+>')
    label_h_list = [float(h) for h in label_h_list]

    #对当前comb，遍历所有sheet_size，选择总面积最小的sheet_size
    sheet, res, tot_area = get_best_sheetsize_for_one_dg_comb(sheet_size_list, dg_id,cg_id,label_w_list,label_h_list,re_qty,ink_seperator_width,
                                                              criteria_dict,
                                                              mode) ###--->>>

    #判断解是否符合criteria
    if check_criteria:
      sheet_size_name = str(int(sheet[0]))+'<+>'+str(int(sheet[1]))
      pds_lower_lim = criteria_dict[sheet_size_name]['pds']
      pds_value = np.max(res['pds'])
      if pds_value < pds_lower_lim: #fail criteria
        continue

    #如果是当前更优解，更新结果 --- 选sheet_size:基于pds*sheet_area
    if tot_area<min_tot_area:
      best_comb = comb_name
      best_sheet = sheet   
      min_tot_area = tot_area
      best_res = res
  # print()
  # print(f'best_comb={best_comb}, best_sheet={best_sheet}, best_res={best_res}, min_tot_area={min_tot_area}')
  return best_comb, best_sheet, best_res, min_tot_area


# ------ for SKU Allocation ------


def allocate_sku(sku_qty_dict, n_ups):
  """
  算法一：以小到大
  :param sku_qty_dict: {sku:qty}, 每一个sku的qty
  :param n_ups: int, 所有sku的ups总和
  """
  sku_list = list(sku_qty_dict.keys())
  qty_list = list(sku_qty_dict.values())
  cur_ups = [1]*len(sku_qty_dict)
  while np.sum(cur_ups)<n_ups:
    cur_pds = [a/b for a,b in zip(qty_list, cur_ups)]
    imax = cur_pds.index(max(cur_pds))
    cur_ups[imax] += 1
  
  res_dict = {}
  cur_pds = [a/b for a,b in zip(qty_list, cur_ups)]
  for i in range(len(sku_qty_dict)):
    res_dict[sku_list[i]] = {'qty':qty_list[i],
                              'ups':cur_ups[i], 
                              'pds':int(np.ceil(cur_pds[i]))}
  return res_dict


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