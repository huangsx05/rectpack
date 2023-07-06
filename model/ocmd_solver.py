import numpy as np
from utils.tools import allocate_cols_based_on_qty

# -------------------------------------
# ------ 2. ONE COLOR MORE DIMENSION ------
# -------------------------------------


def get_batches_ocmd(df_3,internal_days_limit):
  dg_sorted_list = sorted(df_3['dg_id'].unique())  
  dg_cg_dict = dict(zip(df_3['dg_id'], df_3['cg_id']))
  dg_wds_dict = dict(zip(df_3['dg_id'], df_3['wds']))  #for internal dates    
  cg_list = sorted(df_3['cg_id'].unique())
  all_batches = []
  for cg in cg_list:
    #该CG下的DG
    dg_list = []
    for k,v in dg_cg_dict.items():
      if v==cg:
        dg_list.append(k)
    #获得DG batch的所有可能性
    batches_list = [] #存储待进行计算的batches(过滤不合格的batches之后)
    v_set_list = []   #for去重
    N = len(dg_list)
    M = len(dg_list)    
    for n in range(N**M): #所有可能的组合的个数为N**M
      combination = [[] for __ in range(M)] #初始化, M是sub_batch数量
      for i in range(N):
        combination[n // M**i % M].append(i) 
      combination = [c for c in combination if len(c)>0] #这里的c应该是排好序的
      #过滤条件1：去重 
      v_set = set([str(c) for c in combination])   
      if v_set not in v_set_list:
        v_set_list.append(v_set)      
        one_batch = []
        for index in range(len(combination)): #遍历sub_batch
          sub_batch = [dg_sorted_list[i] for i in combination[index]]
          wds_list = [dg_wds_dict[s] for s in sub_batch]
          #过滤条件3：internal dates 
          if np.max(wds_list)-np.min(wds_list)>internal_days_limit:
            break           
          else:
            one_batch.append(sub_batch)      
        if len(one_batch)==len(combination): #没有sub_batch被break
          batches_list.append(one_batch)
    #与当前其他cg组合
    if len(all_batches)==0:
      all_batches = batches_list
    else:
      temp_all_batches = []
      for b1 in all_batches:
        for b2 in batches_list:
          temp_all_batches.append(b1+b2)
      all_batches = temp_all_batches
  #加batch name
  final_batches = [] 
  for batch in all_batches: 
    batch_dict = {}
    for index in range(len(batch)): #遍历sub_batch
      b_key = 'b'+str(index)
      batch_dict[b_key] = batch[index]      
    final_batches.append(batch_dict)

  return final_batches  


# ------ 1.1 filter ------


def only_keep_cg_with_mul_dgs(df_2):
  cols = ['cg_id', 'dimension_group', 'fix_orientation','overall_label_width', 'overall_label_length']
  df_2 = df_2[['cg_dg_id']+cols+['re_qty']]

  #过滤：只考虑有多行的cg_id，因为单行的应该在OCOD考虑
  df_2_cg_count = df_2.groupby('cg_id')['cg_dg_id'].count().reset_index().sort_values('cg_dg_id', ascending=False)
  multi_dg_cg = df_2_cg_count[df_2_cg_count['cg_dg_id']>1]['cg_id'].values
  df_2 = df_2[df_2['cg_id'].isin(multi_dg_cg)].sort_values(['cg_id', 're_qty'], ascending=[True, False])
  return df_2


def filter_id_by_criteria_ocmd(df_2, cg_id, sheet_size_list, criteria_dict):
  """
  通过找pds的上限（ups的下限），判断是否有可能满足最小pds的要求
  """
  fail_criteria = True
  df_temp = df_2[df_2['cg_id']==cg_id]
  sum_qty = df_temp['re_qty'].sum()
  max_dimension = np.max([df_temp['overall_label_width'].max(), df_temp['overall_label_length'].max()])
  for sheet_size in sheet_size_list:
    n_rows_low_lim = int(sheet_size[1]/max_dimension)
    n_cols_low_lim = int(sheet_size[0]/max_dimension)
    min_ups = n_rows_low_lim*n_cols_low_lim
    max_pds = np.ceil(sum_qty/min_ups)
    sheet_size_name = str(int(sheet_size[0]))+'<+>'+str(int(sheet_size[1]))
    if max_pds>=criteria_dict[sheet_size_name]['pds']:
      fail_criteria = False
      break
  return fail_criteria

def get_coordinates_for_each_ups_on_one_sheetsize(sheet_size,dg_id,label_w_list,label_h_list,ups_list):
  #初始化
  max_n_cols = int(sheet_size[0]/np.min(label_w_list)) #最大可能的列数，预留存储空间
  used_col_w = [0]*max_n_cols #已占用的每一列的宽
  used_col_h = [0]*max_n_cols #已占用的每一列的高
  left_sheet_w = sheet_size[0] #sheet的剩余宽度
  left_col_h = [sheet_size[1]]*max_n_cols #已占用的每一列的剩余高度
  ups_info_by_col = [[]]*max_n_cols #每一个label_size在每一个col的每一个ups的左下角点坐标[[x,y,w,h,dg_id],[]]

  for index in range(len(label_w_list)): #对每一个dg
    dg_i = dg_id[index]
    label_w = label_w_list[index]
    label_h = label_h_list[index]
    n_ups = ups_list[index]

    #step 1: 如果已经有col, 先往每个col上方的剩余空间塞
    i = 0
    while n_ups>0:
      add_n_ups = np.min([int(left_col_h[i]/label_h),n_ups]) #该列剩余空间可放的ups数量
      if add_n_ups==0: #如果当前列放不下，尝试下一列
        i+=1
      else:
        #添加新增ups的左下角坐标
        if i==0:
          x = 0
        else:
          x = np.sum(used_col_w[:i])
        y = used_col_h[i]

        for n in range(add_n_ups): #每一个新增的ups
          x_n = x
          y_n = y+n*label_h
          ups_info = [x_n,y_n,label_w,label_h,dg_i]
          # print(ups_info_by_col)
          # print(ups_info)
          ups_info_by_col[i].append(ups_info)
          #更新layout信息
          used_col_h[i] += label_h
          left_col_h[i] -= label_h
      
      used_col_w[i] = np.max([used_col_w[i], label_w])
      n_ups = np.max([(n_ups-add_n_ups),0])

  return ups_info_by_col


def layout_ocmd_with_fixed_ups_on_one_sheetsize(sheet_size,dg_id,label_w_list,label_h_list,ups_list):
  """
  给定sheet_size, 每个dg的label_size，每个dg的ups
  返回是否能够layout, 以及layout结果
  为方便layout，label_w从大到小预先排序
  """
  # print(f'layout_ocmd_with_fixed_ups_on_one_sheetsize')
  #初始化
  max_n_cols = int(sheet_size[0]/np.min(label_w_list)) #最大可能的列数，预留存储空间
  used_col_w = [0]*max_n_cols #已占用的每一列的宽
  used_col_h = [0]*max_n_cols #已占用的每一列的高
  left_sheet_w = sheet_size[0] #sheet的剩余宽度
  left_col_h = [sheet_size[1]]*max_n_cols #已占用的每一列的剩余高度
  # ups_info_by_col = [[]]*max_n_cols #每一个label_size在每一个col的每一个ups的左下角点坐标[[x,y,w,h,dg_id],[]]

  can_layout = True
  for index in range(len(label_w_list)): #对每一个dg
    dg_i = dg_id[index]
    label_w = label_w_list[index]
    label_h = label_h_list[index]
    n_ups = ups_list[index]

    #如果已经有col, 先往每个col上方的剩余空间塞
    i = 0
    while n_ups>0:
      add_n_ups = np.min([int(left_col_h[i]/label_h),n_ups]) #该列剩余空间可放的ups数量
      if add_n_ups==0: #如果当前列放不下，尝试下一列
        i+=1
        if i >= max_n_cols: #max列数不足以放下ups
          can_layout = False
          break

      # #---------------------------------------------
      if add_n_ups>0:
        for n in range(add_n_ups): #每一个新增的ups
      #     #更新layout信息
          used_col_h[i] += label_h
          left_col_h[i] -= label_h
      # #-----------------------------------------------
      
      used_col_w[i] = np.max([used_col_w[i], label_w])
      n_ups = np.max([(n_ups-add_n_ups),0])

  return can_layout#, ups_info_by_col


def iterate_to_get_best_ups_list_allocation(sheet_size,dg_id,re_qty,label_w_list,label_h_list,
                                            n_cols_search_lower,n_ups_search_upper):
  """
  遍历所有情况获得ups分配的最优解
  """
  sheet_area = sheet_size[0]*sheet_size[1]
  label_area_list = np.multiply(label_w_list,label_h_list)
  
  # --- 在sheet_size上做layout ---
  # n_rows_upper_lim = [int(sheet_size[1]/h) for h in label_h_list] #每一个dg的行数
  # n_cols_upper_lim = [int(sheet_size[0]/w) for w in label_w_list] #每一个dg的最大列数
  # n_ups_upper_lim = np.multiply(n_rows_upper_lim, n_cols_upper_lim)

  if len(label_w_list)==1:
    # return n_ups_upper_lim, [np.ceil(re_qty[0]/n_ups_upper_lim[0])], 
    print(f'This should be OCOD case, not OCMD case')
    stop_flag = 1/0

  elif len(label_w_list)==2:
    min_pds = 1e6
    best_ups_list = [0]*len(label_w_list)
    best_pds_list = [0]*len(label_w_list) 
    # best_ups_layout = None

    #找最优ups的解
    for i in range(n_cols_search_lower[0],n_ups_search_upper[0]+1):
      # print(f'iterating {i} --- iterate_to_get_best_ups_list_allocation ---')
      for j in range(n_cols_search_lower[1],n_ups_search_upper[1]+1): ###多一层循环需要改这
        ups_list = [i,j] ###多一层循环需要改这        
        #根据总面积筛除不合理组合
        if np.sum(np.multiply(ups_list, label_area_list))>sheet_area:
          continue
        #尝试在sheet_size上layout ups，若可layout，判断是否更优解
        can_layout = layout_ocmd_with_fixed_ups_on_one_sheetsize(sheet_size,dg_id,label_w_list,label_h_list,ups_list) ###--->>>
        if can_layout==False: #无效解
          # print(f'ups_list={ups_list}, can_layout={can_layout}')          
          continue        
        #update最优解 --- 选ups layout: 基于pds
        pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
        if np.max(pds_list)<min_pds:
          min_pds = np.max(pds_list)
          best_ups_list = ups_list
          best_pds_list = pds_list
          # best_ups_layout = ups_info_by_col

    #获得最优解的layout - 每个ups的坐标
    print(f'best_ups_list={best_ups_list}')
    best_ups_layout = get_coordinates_for_each_ups_on_one_sheetsize(sheet_size,dg_id,label_w_list,label_h_list,best_ups_list)

  return best_ups_list, best_pds_list, best_ups_layout


def get_ups_layout_for_ocmd_comb_on_one_sheetsize(dg_id,label_w_list,label_h_list,re_qty,sheet_size,layout_tolerance):
  """
  #对一个sheet_size遍历不同dg ups的组合情况，找到最优ups组合
  用于ocmd, 一列中不同dg混排
  返回每个dg布局多少列，多少行，多少ups
  采用遍历法
  """
  # dg_id重排序，以利于先layout大标签再layout小标签
  dg_index_list = list(range(len(dg_id))) #存储原始排序，为了在得到结果之后按照原dg_id排序返回
  # print('before sorting: ',label_w_list,label_h_list,re_qty,dg_id,dg_index_list)
  zipped = zip(label_w_list,label_h_list,re_qty,dg_id,dg_index_list)
  zipped_sorted = sorted(zipped,reverse=True)
  label_w_list,label_h_list,re_qty,dg_id,dg_index_list = zip(*zipped_sorted)
  # print('after sorting: ',label_w_list,label_h_list,re_qty,dg_id,dg_index_list)  

  # --- 在sheet_size上做layout ----------------------------------------------
  effective_sheet_width = sheet_size[0] #one color，不需要预留ink seperator
  n_rows = [int(sheet_size[1]/h) for h in label_h_list] #每一个dg的最大行数
  n_cols_upper_lim = [int(effective_sheet_width/w) for w in label_w_list] #每一个dg的最大列数

  #初始解：按照初始tot_cols预分配 
  tot_cols = np.min(n_cols_upper_lim) #初始总列数
  n_cols = allocate_cols_based_on_qty(tot_cols, re_qty)
  # print(f'n_cols heuristics initial solution = {n_cols}')
  n_ups = np.multiply(n_rows, n_cols)
  # print(f'n_ups heuristics initial solution = {n_ups}')  

  #逐步增大列数直至超出effective_sheet_width
  label_width_sum = sum(np.multiply(n_cols, label_w_list))
  temp_tot_cols = tot_cols
  while label_width_sum < effective_sheet_width:
    temp_tot_cols += 1
    temp_n_cols = allocate_cols_based_on_qty(temp_tot_cols, re_qty)
    label_width_sum = sum(np.multiply(temp_n_cols, label_w_list))
    if label_width_sum <= effective_sheet_width: #有效解，更新结果
      tot_cols = temp_tot_cols
      n_cols = temp_n_cols
  # print(f'n_cols heuristics final solution = {n_cols}')
  n_ups = np.multiply(n_rows, n_cols)
  # print(f'n_ups heuristics final solution = {n_ups}')  

  tolerance = layout_tolerance
  # n_cols_upper_lim = n_cols
  n_cols_search_upper = [int(i+tolerance) for i in n_cols]
  n_cols_search_lower = [int(np.max([i-tolerance,0])) for i in n_cols] #因为考虑互扣，所以允许下限为0
  n_ups_search_upper = np.multiply(n_rows, n_cols_search_upper)
  n_ups_search_lower = np.multiply(n_rows, n_cols_search_lower)  
  n_ups_search_upper = [int(np.max([i,1])) for i in n_ups_search_upper]
  n_cols_search_lower = [int(np.max([i,1])) for i in n_cols_search_lower]  
  print(f'n_ups heuristics search range = {n_ups_search_lower} to {n_ups_search_upper}')  
  #---------------------------------------------------------------------------

  #遍历所有情况获得ups分配的最优解
  ups_list, pds_list, ups_info_by_col = iterate_to_get_best_ups_list_allocation(sheet_size,dg_id,re_qty,label_w_list,label_h_list,
                                                                                n_cols_search_lower,n_ups_search_upper) #每一个dg分配多少个ups
  # print(f'final ups_list = {ups_list}')

 # dg_id重排序，以和输入时的dg_id排序一致
  dg_layout_seq = list(range(len(dg_id))) #存储dg的layout排序，利于之后画图 
  # print('before output sorting: ',dg_index_list,dg_id,dg_layout_seq,ups_list,pds_list)  
  zipped_output = zip(dg_index_list,dg_id,dg_layout_seq,ups_list,pds_list)
  zipped_output_sorted = sorted(zipped_output,reverse=False)
  dg_index_list,dg_id,dg_layout_seq,ups_list,pds_list = zip(*zipped_output_sorted)
  # print('after output sorting: ',dg_index_list,dg_id,dg_layout_seq,ups_list,pds_list)  

  return list(ups_list), list(pds_list), list(dg_layout_seq), ups_info_by_col