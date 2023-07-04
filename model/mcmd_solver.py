import numpy as np
import pandas as pd
from utils.tools import allocate_cols_based_on_qty, get_max_sku_pds_for_each_dg
# from model.shared_solver import allocate_ups_dg_level

# ----------------------------
# ------ for ups layout ------
# ----------------------------


def iterate_to_get_best_n_cols_allocation(dg_id,label_w_list, n_cols_upper_lim, n_rows, re_qty, effective_sheet_width, sheet_size, 
                                                           dg_sku_qty_dict,params_dict, 
                                                           n_cols_search_lower=[], n_cols_search_upper=[]):
  """
  针对一个dg_comb和一个sheet_size
  遍历所有情况获得columns分配的最优解
  """
  tolerance = int(params_dict['algo_params']['layout_tolerance'])

  if tolerance==0:
    ups_list = np.multiply(n_cols_upper_lim, n_rows) #每个dg的ups
    best_pds_list = get_max_sku_pds_for_each_dg(dg_id, ups_list, dg_sku_qty_dict, params_dict) ###--->>>
    return n_cols_upper_lim, best_pds_list
  
  else:
    sheet_name = str(int(sheet_size[0]))+'<+>'+str(int(sheet_size[1]))
    sheet_weight = float(params_dict['user_params']['sheets'][sheet_name]['weight'])

    min_pds = 1e12 #优化目标
    n_cols = [0]*len(label_w_list)
    best_pds_list = [1e12]*len(label_w_list) #基于max_sku_pds

    # print('iterate_to_get_best_n_cols_allocation')
    if len(label_w_list)==1:
      for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
        cur_n_cols = [i]        
        label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
        if label_width_sum > effective_sheet_width: #无效解
          continue        
        ups_list = np.multiply(cur_n_cols, n_rows) #每个dg的ups
        pds_list = get_max_sku_pds_for_each_dg(dg_id,ups_list, dg_sku_qty_dict,params_dict) ###--->>>
        metric = np.max(pds_list)*sheet_weight     
        if metric<min_pds:
          min_pds = metric
          n_cols = cur_n_cols
          best_pds_list = pds_list

    elif len(label_w_list)==2:
      for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
        for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
          cur_n_cols = [i,j]        
          label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
          if label_width_sum > effective_sheet_width: #无效解
            continue        
          ups_list = np.multiply(cur_n_cols, n_rows) #每个dg的ups
          # max_sku_pds = allocate_ups_sku_level(df_i, n_abc, comb_name, ups_list)
          # pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]\
          pds_list = get_max_sku_pds_for_each_dg(dg_id,ups_list, dg_sku_qty_dict,params_dict) ###--->>>
          metric = np.max(pds_list)*sheet_weight     
          # print("i, j, pds_list, metric, min_pds", i, j, pds_list, metric, min_pds)
          if metric<min_pds:
            min_pds = metric
            n_cols = cur_n_cols
            best_pds_list = pds_list

    elif len(label_w_list)==3:
      for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
        for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
          for k in range(n_cols_search_lower[2],n_cols_search_upper[2]+1):        
            cur_n_cols = [i,j,k]        
            label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
            if label_width_sum > effective_sheet_width: #无效解
              continue        
            ups_list = np.multiply(cur_n_cols, n_rows)
            # max_sku_pds = allocate_ups_sku_level(df_i, n_abc, comb_name, ups_list)
            pds_list = get_max_sku_pds_for_each_dg(dg_id,ups_list, dg_sku_qty_dict,params_dict) ###--->>>
            metric = np.max(pds_list)*sheet_weight   
            if metric<min_pds:
              min_pds = metric
              n_cols = cur_n_cols
              best_pds_list = pds_list

    elif len(label_w_list)==4:
      for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
        for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
          for k in range(n_cols_search_lower[2],n_cols_search_upper[2]+1):      
            for l in range(n_cols_search_lower[3],n_cols_search_upper[3]+1): 
              cur_n_cols = [i,j,k,l]        
              label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
              if label_width_sum > effective_sheet_width: #无效解
                continue        
              ups_list = np.multiply(cur_n_cols, n_rows)
              # max_sku_pds = allocate_ups_sku_level(df_i, n_abc, comb_name, ups_list)
              pds_list = get_max_sku_pds_for_each_dg(dg_id,ups_list, dg_sku_qty_dict,params_dict) ###--->>>
              metric = np.max(pds_list)*sheet_weight   
              if metric<min_pds:
                min_pds = metric
                n_cols = cur_n_cols
                best_pds_list = pds_list

    elif len(label_w_list)==5:
      for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
        # print(i, 'be patient')
        for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
          for k in range(n_cols_search_lower[2],n_cols_search_upper[2]+1):      
            for l in range(n_cols_search_lower[3],n_cols_search_upper[3]+1): 
              for m in range(n_cols_search_lower[4],n_cols_search_upper[4]+1):            
                cur_n_cols = [i,j,k,l,m]        
                label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
                if label_width_sum > effective_sheet_width: #无效解
                  continue        
                ups_list = np.multiply(cur_n_cols, n_rows)
                # max_sku_pds = allocate_ups_sku_level(df_i, n_abc, comb_name, ups_list)
                pds_list = get_max_sku_pds_for_each_dg(dg_id,ups_list, dg_sku_qty_dict,params_dict) ###--->>>
                metric = np.max(pds_list)*sheet_weight   
                if metric<min_pds:
                  min_pds = metric
                  n_cols = cur_n_cols
                  best_pds_list = pds_list                            

    elif len(label_w_list)==6:
      for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
        # print(i, 'be patient')
        for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
          for k in range(n_cols_search_lower[2],n_cols_search_upper[2]+1):      
            for l in range(n_cols_search_lower[3],n_cols_search_upper[3]+1): 
              for m in range(n_cols_search_lower[4],n_cols_search_upper[4]+1):    
                for n in range(n_cols_search_lower[5],n_cols_search_upper[5]+1):                       
                  cur_n_cols = [i,j,k,l,m,n]        
                  label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
                  if label_width_sum > effective_sheet_width: #无效解
                    continue        
                  ups_list = np.multiply(cur_n_cols, n_rows)
                  # max_sku_pds = allocate_ups_sku_level(df_i, n_abc, comb_name, ups_list)
                  pds_list = get_max_sku_pds_for_each_dg(dg_id,ups_list, dg_sku_qty_dict,params_dict) ###--->>>
                  metric = np.max(pds_list)*sheet_weight   
                  if metric<min_pds:
                    min_pds = metric
                    n_cols = cur_n_cols
                    best_pds_list = pds_list                                  

    elif len(label_w_list)==7:
      for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
        # print(i, 'be patient')
        for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
          for k in range(n_cols_search_lower[2],n_cols_search_upper[2]+1):      
            for l in range(n_cols_search_lower[3],n_cols_search_upper[3]+1): 
              for m in range(n_cols_search_lower[4],n_cols_search_upper[4]+1):    
                for n in range(n_cols_search_lower[5],n_cols_search_upper[5]+1):                       
                  for o in range(n_cols_search_lower[6],n_cols_search_upper[6]+1):   
                    cur_n_cols = [i,j,k,l,m,n,o]        
                    label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
                    if label_width_sum > effective_sheet_width: #无效解
                      continue        
                    ups_list = np.multiply(cur_n_cols, n_rows)
                    # max_sku_pds = allocate_ups_sku_level(df_i, n_abc, comb_name, ups_list)
                    pds_list = get_max_sku_pds_for_each_dg(dg_id,ups_list, dg_sku_qty_dict,params_dict) ###--->>>
                    metric = np.max(pds_list)*sheet_weight   
                    if metric<min_pds:
                      min_pds = metric
                      n_cols = cur_n_cols
                      best_pds_list = pds_list      

    elif len(label_w_list)==8:
      for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
        # print(i, 'be patient')
        for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
          for k in range(n_cols_search_lower[2],n_cols_search_upper[2]+1):      
            for l in range(n_cols_search_lower[3],n_cols_search_upper[3]+1): 
              for m in range(n_cols_search_lower[4],n_cols_search_upper[4]+1):    
                for n in range(n_cols_search_lower[5],n_cols_search_upper[5]+1):                       
                  for o in range(n_cols_search_lower[6],n_cols_search_upper[6]+1):   
                    for p in range(n_cols_search_lower[7],n_cols_search_upper[7]+1):                     
                      cur_n_cols = [i,j,k,l,m,n,o,p]        
                      label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
                      if label_width_sum > effective_sheet_width: #无效解
                        continue        
                      ups_list = np.multiply(cur_n_cols, n_rows)
                      # max_sku_pds = allocate_ups_sku_level(df_i, n_abc, comb_name, ups_list)
                      pds_list = get_max_sku_pds_for_each_dg(dg_id,ups_list, dg_sku_qty_dict,params_dict) ###--->>>
                      metric = np.max(pds_list)*sheet_weight   
                      if metric<min_pds:
                        min_pds = metric
                        n_cols = cur_n_cols
                        best_pds_list = pds_list        

    elif len(label_w_list)==9:
      for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
        # print(i, 'be patient')
        for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
          for k in range(n_cols_search_lower[2],n_cols_search_upper[2]+1):      
            for l in range(n_cols_search_lower[3],n_cols_search_upper[3]+1): 
              for m in range(n_cols_search_lower[4],n_cols_search_upper[4]+1):    
                for n in range(n_cols_search_lower[5],n_cols_search_upper[5]+1):                       
                  for o in range(n_cols_search_lower[6],n_cols_search_upper[6]+1):   
                    for p in range(n_cols_search_lower[7],n_cols_search_upper[7]+1):              
                      for q in range(n_cols_search_lower[8],n_cols_search_upper[8]+1):                            
                        cur_n_cols = [i,j,k,l,m,n,o,p,q]        
                        label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
                        if label_width_sum > effective_sheet_width: #无效解
                          continue        
                        ups_list = np.multiply(cur_n_cols, n_rows)
                        # max_sku_pds = allocate_ups_sku_level(df_i, n_abc, comb_name, ups_list)
                        pds_list = get_max_sku_pds_for_each_dg(dg_id,ups_list, dg_sku_qty_dict,params_dict) ###--->>>
                        metric = np.max(pds_list)*sheet_weight   
                        if metric<min_pds:
                          min_pds = metric
                          n_cols = cur_n_cols
                          best_pds_list = pds_list             
    else:
      print('to add more codes to consider theis case')
      print(10/0)

    # print(f'iterate_to_get_best_n_cols_allocation ---> return n_cols = {n_cols}')
    return n_cols, best_pds_list


def get_n_cols_for_dg_comb_on_one_sheetsize(dg_id,cg_id,label_w_list,label_h_list,re_qty,sheet_size,
                                            dg_sku_qty_dict,params_dict):
  """
  用于中离
  一列只有一个dg
  返回每个dg布局多少列
  采用给予初始解和tolerance遍历的方法
  """
  can_layout = True
  ink_seperator_width = int(params_dict['user_params']['ink_seperator_width'])

  #基本信息
  n_dg = len(dg_id)
  n_cg = len(set(cg_id))
  n_ink_seperator = n_cg-1
  sheet_width = sheet_size[0]
  effective_sheet_width = sheet_width-n_ink_seperator*ink_seperator_width
  # print('dg_id, n_cg, n_dg, n_ink_seperator, sheet_width, ink_seperator_width, effective_sheet_width = ', dg_id, n_cg, n_dg, n_ink_seperator, sheet_width, ink_seperator_width, effective_sheet_width)

  #在sheet_size上做layout
  #限制条件
  n_rows = [int(sheet_size[1]/h) for h in label_h_list] #每一个dg的行数
  n_cols_upper_lim = [int(effective_sheet_width/w) for w in label_w_list] #每一个dg的最大列数
  #初始解：按照tot_cols_init预分配 
  tot_cols = np.min(n_cols_upper_lim) #初始总列数
  tot_cols = np.max([tot_cols, n_dg]) #保证每个dg至少有一列
  # print(f'n_cols heuristics initial solution = {tot_cols}>{np.min(n_cols_upper_lim)}')
  n_cols = allocate_cols_based_on_qty(tot_cols, n_rows, re_qty, dg_id, dg_sku_qty_dict, params_dict) ###--->>>
  print(f'***re_qty = {re_qty}')  
  print(f'***n_rows = {n_rows}')    
  print(f'***n_cols heuristics initial solution after allocation = {n_cols}')

  #处理edge case
  #处理25mm limit
  agg_width = np.multiply(n_cols, label_w_list)
  minWidthandLength = float(params_dict['user_params']['min_single_col_width'])
  print(f"dg_id = {dg_id}")  
  print(f"cg_id = {cg_id}")  
  print(f"agg_width after 25mm limit = {agg_width}")

  #处理中间CG的85mm limit
  if n_cg>2:
    agg_width = [max(i,minWidthandLength) for i in agg_width]
    minMiddleCGWidth = float(params_dict['user_params']['minMiddleCGWidth'])
    #当前每个cg的width
    cg_agg_width = {}
    for cg in cg_id:
      cg_agg_width[cg] = 0
    for i in range(len(dg_id)):
      cg = cg_id[i]
      cg_agg_width[cg] += agg_width[i]
    #处理85mm limit
    final_cg_width = {}
    non_assign_cg_width = {k:v for k,v in cg_agg_width.items()}
    for i in range(2):
      minValueKey = min(non_assign_cg_width, key=non_assign_cg_width.get)
      final_cg_width[minValueKey] = non_assign_cg_width[minValueKey]
      del non_assign_cg_width[minValueKey]
    for k,v in non_assign_cg_width.items():
      final_cg_width[k] = max(minMiddleCGWidth, non_assign_cg_width[k])
    print(f"agg_width after 85mm limit = {final_cg_width}")  
    label_width_sum = sum(list(final_cg_width.values()))
  else:
    label_width_sum = sum(agg_width)

  #check overall width
  if label_width_sum > effective_sheet_width:
    can_layout = False
    return can_layout, n_rows, n_cols, [], []
  
  #逐步增大列数直至超出effective_sheet_width, 得到最大的n_cols初始解
  temp_tot_cols = tot_cols
  while label_width_sum < effective_sheet_width:
    temp_tot_cols += 1
    temp_n_cols = allocate_cols_based_on_qty(temp_tot_cols, n_rows, re_qty, dg_id, dg_sku_qty_dict, params_dict) ###--->>>
    label_width_sum = sum(np.multiply(temp_n_cols, label_w_list))
    if label_width_sum <= effective_sheet_width: #有效解，更新结果
      tot_cols = temp_tot_cols
      n_cols = temp_n_cols
  print(f'n_cols heuristics final solution = {n_cols}')

  tolerance = int(params_dict['algo_params']['layout_tolerance'])
  if tolerance!=0:
    n_cols_search_upper = [np.min([int(n_cols[i]+tolerance),n_cols_upper_lim[i]]) for i in range(n_dg)] #不超过上限
    n_cols_search_lower = [int(np.max([i-tolerance,1])) for i in n_cols] #每个dg至少有1列
  else:
    n_cols_search_upper = []
    n_cols_search_lower = []   

  #遍历n_cols上下限获得columns分配的最优解
  n_cols_upper_lim = n_cols #初始解，不要被变量名误导，不是上限  
  n_cols, pds_list = iterate_to_get_best_n_cols_allocation(dg_id,label_w_list, n_cols_upper_lim, n_rows, re_qty, effective_sheet_width, sheet_size, 
                                                           dg_sku_qty_dict,params_dict, 
                                                           n_cols_search_lower, n_cols_search_upper) ###--->>>
  # print(f"n_cols search range = {n_cols_search_lower},{n_cols_search_upper}")  
  print(f"n_cols best solution = {n_cols}")
  ups_list = list(np.multiply(n_cols, n_rows))
  # pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
  # print(f'n_rows={n_rows}, n_cols={n_cols}, ups={ups_list}, pds={pds_list}')  
  # print(f'label_width_sum={label_width_sum}, effective_sheet_width={effective_sheet_width}')  

  return can_layout, n_rows, n_cols, ups_list, pds_list


# ----------------------------
# ------ for sku allocation ------
# ----------------------------


# def allocate_ups_sku_level(df_i, n_abc, comb_name, ups_list):
#   """
#   allocate sku within one dg
#   df_i: df for one dg_id
#   """
#   # #一个sub_batch的sku allocation
#   # df_i = df_3_3[df_3_3['sub_batch_id']==sub_batch_id]
#   # # display(df_i)
#   # # dg_id = best_batch[sub_batch_id] #不能用这个，顺序会不一样
#   # best_comb = res[sub_batch_id]['best_comb'] #每个dg的旋转方向
#   dg_id = [i[:-2] for i in comb_name.split('<+>')]  
#   dg_orient = [i[-1] for i in comb_name.split('<+>')]
#   # ups_list = list(res[sub_batch_id]['best_res']['ups'])
#   # print()
#   # print(f'dg_id = {dg_id}')
#   # print(f'ups_list = {ups_list}')  
#   # print()

#   for sub_dg_index in range(len(dg_id)): #在每一个dg内部分配做sku的ups分配
#     sub_dg = dg_id[sub_dg_index]
#     df_i_sub = df_i[df_i['dimension_group']==sub_dg]
#     # print(f'sub_dg = {sub_dg}')

#     #step 1: 按照n_abc*ups分配allocate sku
#     sku_qty_dict = dict(zip(df_i_sub['sku_id'],df_i_sub['re_qty']))
#     n_ups = ups_list[sub_dg_index]
#     # print(f'sku_qty_dict = {sku_qty_dict}')    
#     # print(f'n_ups = {n_ups}')       
#     res_dict = allocate_ups_dg_level(sku_qty_dict, n_ups*n_abc) ###--->>>
#     # print(f'res_dict = {res_dict}')

#     for sku_id in res_dict.keys():
#       df_i_sub.loc[df_i_sub['sku_id']==sku_id, 'sku_ups'] = res_dict[sku_id]['ups']
#       df_i_sub.loc[df_i_sub['sku_id']==sku_id, 'sku_pds'] = res_dict[sku_id]['pds']  

#     max_sku_ups = np.max(df_i_sub['sku_ups'])
#     # print(f"sum_sku_ups = {np.sum(df_i_sub['sku_ups'])}")
#     # print(f"max_sku_ups = {max_sku_ups}")
#     # # display(df_i_sub)

#     # df_i_sub = df_i_sub.sort_values(['dimension_group','sku_pds']).reset_index().drop(columns=['index']) #按照pds从小到大排序 - ppc要求
#     # df_i_sub['cum_sum_ups'] = df_i_sub.groupby(['dimension_group'])['sku_ups'].cumsum()   

#     # #step 2: 做ABC版的ups分割（每个版的ups应该相等）split ABC sheets
#     # df_i_sub = split_abc_ups(sub_id=sub_dg, sub_id_colname='dimension_group', df=df_i_sub, ups_dict={sub_dg:n_ups})

#     # #存放结果
#     # df_i_list.append(df_i_sub)
#     return max_sku_ups
  

# def allocate_ups_dg_level(sku_qty_dict, n_ups):
#   """
#   算法一：以小到大
#   :param sku_qty_dict: {sku:qty}, 每一个sku的qty
#   :param n_ups: int, 所有sku的ups总和
#   """
#   sku_list = list(sku_qty_dict.keys())
#   qty_list = list(sku_qty_dict.values())
#   cur_ups = [1]*len(sku_qty_dict)
#   while np.sum(cur_ups)<n_ups:
#     cur_pds = [a/b for a,b in zip(qty_list, cur_ups)]
#     imax = cur_pds.index(max(cur_pds))
#     cur_ups[imax] += 1
  
#   res_dict = {}
#   cur_pds = [a/b for a,b in zip(qty_list, cur_ups)]
#   for i in range(len(sku_qty_dict)):
#     res_dict[sku_list[i]] = {'qty':qty_list[i],
#                               'ups':cur_ups[i], 
#                               'pds':int(np.ceil(cur_pds[i]))}
#   return res_dict


# def split_abc_ups(sub_id, sub_id_colname, df, ups_dict):
#   """
#   #step 2: 做ABC版的ups分割（每个版的ups应该相等）split ABC sheets
#   """  
#   print(f'------ abc splitting for {sub_id} ------')
#   sets = ['Set A Ups','Set B Ups','Set C Ups','Set D Ups','Set E Ups','Set F Ups','Set G Ups','Set H Ups'] #预设版数
#   df_temp = df[df[sub_id_colname]==sub_id].reset_index().drop(columns=['index'])  
#   # n = 1 #当前的ups倍数
#   cur_set_index = 0
#   # sum_pds = 0
#   for i in range(len(df_temp)): #对每一个sku
#     cur_ups_thres = (cur_set_index+1)*ups_dict[sub_id]
#     sku_id = df_temp.loc[i,'sku_id']
#     set_name = sets[cur_set_index]
#     # print(cur_set_index, set_name)      
#     # print(df_temp['cum_sum_ups'].values.tolist())      
#     # print(df_temp.loc[i,'cum_sum_ups'],cur_ups_thres)
#     if df_temp.loc[i,'cum_sum_ups']<=cur_ups_thres: #无需换版
#       df.loc[df['sku_id']==sku_id, set_name] = df['sku_ups']
#     else: #换到下一个版，当前sku需要分配到两个不同的版
#       # sum_pds += df_temp.loc[i,'sku_pds']
#       if i==0:
#         pre_sku_ups = cur_ups_thres
#       else:
#         pre_sku_ups = cur_ups_thres - df_temp.loc[i-1,'cum_sum_ups']          
#       df.loc[df['sku_id']==sku_id, set_name] = pre_sku_ups #pre sheet
#       next_sku_ups = df_temp.loc[i,'sku_ups'] - pre_sku_ups
#       # n += 1
#       cur_set_index += 1  
#       set_name = sets[cur_set_index]
#       # print(cur_set_index, set_name)   
#       df.loc[df['sku_id']==sku_id, set_name] = next_sku_ups #next_sheet       
#   # sum_pds += df_temp['sku_pds'].values[-1]

#   for set_name in sets:
#     if set_name in df.columns:
#       df.fillna(0,inplace=True)
#       df_temp = df[df[sub_id_colname]==sub_id]
#       print(f'sum_ups for {set_name} = {np.sum(df_temp[set_name])}') #确认每个set的ups相等

#   df = df.sort_values([sub_id_colname,'sku_pds'])    
#   return df