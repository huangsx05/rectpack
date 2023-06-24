import numpy as np


def combinations(dg_sorted_list, N, M):
  #generate all possible combinations
  for n in range(N**M): #所有可能的组合的个数为N**M
    combination = [[] for __ in range(M)] #初始化, M是sub_batch数量
    for i in range(N): #依次加入dg
      combination[n // M**i % M].append(dg_sorted_list[i]) #one batch
    combination = [c for c in combination if len(c)>0] #这里c里面的index应该是排好升序的
    return combination


def partitions(A):
  if not A:
    yield []
  else:
    a, *R = A
    for partition in partitions(R):
      yield partition + [[a]]
      for i, subset in enumerate(partition):
        yield partition[:i] + [subset + [a]] + partition[i+1:]
        

def add_sub_batch_id_to_df(df_3_3, batch):
  #revert字典，将batch号加入df_3_3
  batch_revert = {}
  for k,v in batch.items():
    for i in v:
      batch_revert[i] = k
  df_3_3 = df_3_3[df_3_3['dimension_group'].isin(batch_revert.keys())]
  df_3_3['sub_batch_id'] = df_3_3['dimension_group'].apply(lambda x: batch_revert[x])
  df_3_3 = df_3_3.sort_values(['sub_batch_id','dimension_group','sku_id','re_qty'])
  return df_3_3


def allocate_cols_based_on_qty(tot_n_cols, n_rows, qty_list):
  # tot_qty = np.sum(qty_list)
  # n_cols = [np.max([round(qty/tot_qty*tot_n_cols),1]) for qty in qty_list] #每个dg的初始列数, 并保证至少有一列
  # n_cols = [round(qty/tot_qty*tot_n_cols) for qty in qty_list] #每个dg的初始列数, 并保证至少有一列  
  # n_cols_float = [qty/tot_qty*tot_n_cols for qty in qty_list]
  # if tot_n_cols!=np.sum(n_cols):
  #   n_max_index = np.argmax(n_cols)
  #   # print(f'n_max_index={n_max_index}')
  #   updated_n_col = tot_n_cols - np.sum(n_cols[:n_max_index]) - np.sum(n_cols[n_max_index+1:])
  #   n_cols[n_max_index] = updated_n_col #调整最多列数的dg，保证列数总和等于tot_n_cols
  #   assert tot_n_cols == np.sum(n_cols)

  n_cols = [1]*len(qty_list)
  # print(tot_n_cols, np.sum(n_cols))
  if tot_n_cols<np.sum(n_cols): #调用函数已防止此情况发生，可以省略
    print('need to debug')
    stop = 1/0
  
  while np.sum(n_cols)<tot_n_cols:
    ups_list = list(np.multiply(n_cols, n_rows))
    pds_list = [np.ceil(a/b) for a, b in zip(qty_list, ups_list)]    
    n_max_index = np.argmax(pds_list)
    n_cols[n_max_index] += 1
  assert tot_n_cols == np.sum(n_cols)

  return n_cols


def get_all_dg_combinations_with_orientation(dg_id,fix_orientation,label_width,label_length):
  """
  穷举该batch在一个sheet中的所有排列组合，考虑rotation
  每一个comb都包含batch中的所有dg，区别在于dg是否rotate
  返回所有排列组合的结果
  :param:  
    dg_id = ['dg_091', 'dg_093']
    fix_orientation = [0, 0]
    label_width = [30.0, 28.0]
    label_length = [29.0, 41.0]
  :return:
    comb_names = ['dg_091_w<+>dg_093_w', 'dg_091_h<+>dg_093_w', 'dg_091_w<+>dg_093_h', 'dg_091_h<+>dg_093_h']
    comb_res_w = ['30.0<+>28.0', '29.0<+>28.0', '30.0<+>41.0', '29.0<+>41.0']
    comb_res_h = ['29.0<+>41.0', '30.0<+>41.0', '29.0<+>28.0', '30.0<+>28.0']
  """
  #给初始值：考虑第一个dg
  if fix_orientation[0]==1: 
    comb_names = [dg_id[0]+'_w']
    comb_res_w = [str(label_width[0])]
    comb_res_h = [str(label_length[0])]
  elif fix_orientation[0]==0: 
    comb_names = [dg_id[0]+'_w', dg_id[0]+'_h']
    comb_res_w = [str(label_width[0]), str(label_length[0])]
    comb_res_h = [str(label_length[0]), str(label_width[0])]    

  #继续添加之后的dg
  for i in range(1,len(dg_id)): #从1开始
    fix_orientation_i = fix_orientation[i]
    label_width_i = str(label_width[i])
    label_length_i = str(label_length[i])
    if fix_orientation_i==1: #不能旋转，组合数量不变
      for j in range(len(comb_names)):    
        comb_names[j] += ('<+>'+dg_id[i]+'_w')
      for k in range(len(comb_res_w)):    
        comb_res_w[k] += ('<+>'+label_width_i)
        comb_res_h[k] += ('<+>'+label_length_i)        
    elif fix_orientation_i==0: #可以旋转，组合数量翻倍
      comb_names_add = comb_names.copy()
      comb_res_w_add = comb_res_w.copy()
      comb_res_h_add = comb_res_h.copy()
      for j in range(len(comb_names)):    
        comb_names[j] += ('<+>'+dg_id[i]+'_w')
      for k in range(len(comb_res_w)):    
        comb_res_w[k] += ('<+>'+label_width_i)
        comb_res_h[k] += ('<+>'+label_length_i)      
      for j in range(len(comb_names_add)):    
        comb_names_add[j] += ('<+>'+dg_id[i]+'_h')
      for k in range(len(comb_res_w_add)):    
        comb_res_w_add[k] += ('<+>'+label_length_i)
        comb_res_h_add[k] += ('<+>'+label_width_i)
      comb_names = comb_names+comb_names_add
      comb_res_w = comb_res_w+comb_res_w_add
      comb_res_h = comb_res_h+comb_res_h_add      
  return comb_names, comb_res_w, comb_res_h


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


def get_max_sku_pds_for_each_dg(dg_id, ups_list, dg_sku_qty_dict,params_dict):
  n_abc = int(params_dict['user_params']['n_abc'])
  pds_list = [] #每一个dg的最大sku_pds
  for sub_dg_index in range(len(dg_id)): #在每一个dg内部分配做sku的ups分配
    sub_dg = dg_id[sub_dg_index] #dg_id
    #step 1: 按照n_abc*ups分配allocate sku
    sku_qty_dict = dg_sku_qty_dict[sub_dg]
    n_ups = ups_list[sub_dg_index]
    # print(f'sku_qty_dict = {sku_qty_dict}')    
    # print(f'n_ups = {n_ups}')       
    res_dict = allocate_sku(sku_qty_dict, n_ups*n_abc) ###--->>>
    # print(f'res_dict = {res_dict}')
    sku_pds_list = []
    for sku_id in res_dict.keys():
      sku_pds_list.append(res_dict[sku_id]['pds'])  
    pds_list.append(np.max(sku_pds_list))
  return pds_list


def calculate_best_batch(res_list):
  assessed_metrics = {}
  res = {}
  for i in range(len(res_list)): #一批并行计算的batch
    for j in range(len(res_list[i])): #batch
      # print(res_list[i][j])
      batch_name = list(res_list[i][j].keys())[0]
      metric = list(res_list[i][j].values())[0]['metric']
      result = list(res_list[i][j].values())[0]['res']
      assessed_metrics[batch_name] = metric
      res[batch_name] = result
      # print()
      # print(batch_name)
      # print(metric)
      # print(result)

  best_index = min(assessed_metrics, key=assessed_metrics.get)
  best_res = res[best_index]
  best_metric = np.min(list(assessed_metrics.values()))
  # print(f"check_min_metric = {np.min(list(assessed_metrics.values()))}")

  best_batch = {}
  for k,v in best_res.items():
    # print(k,v)
    if k!='metric':
      sub_dgs = v['best_comb'].split('<+>')
      sub_dgs = [i[:-2] for i in sub_dgs] 
      best_batch[k] = sub_dgs

  return assessed_metrics, best_index, best_batch, best_res, best_metric