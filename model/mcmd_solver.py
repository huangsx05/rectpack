import numpy as np
import pandas as pd
# from utils.one_dimension_packer_solver import *
from utils.tools import allocate_cols_based_on_qty


# def get_min_width_for_one_color_one_dimension_for_one_sheetsize(label_width, label_length, sheet_width, sheet_length, ups_upper_lim, fix_orientation):
#   """
#   根据需要的ups计算一个label size在sheet中布局所需的最小宽度（布局sheet全高）- 中离布局中一个cell的宽度
#   实际的ups要少于ups_upper_lim
#   one label == one color one dimension
#   TO IMPROVE: fix_orientation = 0的时候，暂未考虑横竖混排
#   :return:
#     mid_width
#     is_rotated: True/False
#   """
#   label_size = [label_width, label_length]
#   sheet_size = [sheet_width, sheet_length]
#   #before orientation
#   n_rows_1 = int(sheet_size[1]/label_size[1])
#   n_cols_1 = max(int(ups_upper_lim/n_rows_1),1) #注意不用np.ceil, 因为实际ups不能超过ups_upper_lim
#   min_width_1 = n_cols_1*label_size[0]
#   while min_width_1>sheet_size[0]:
#     n_cols_1 -= 1
#     min_width_1 = n_cols_1*label_size[0]
#   if fix_orientation==1:  
#     return {'n_rows':n_rows_1, 'n_cols':n_cols_1, 'min_width':min_width_1, 'is_rotated':0}
#   #after orientation
#   n_rows_2 = int(sheet_size[1]/label_size[0])
#   n_cols_2 = max(int(ups_upper_lim/n_rows_2),1)
#   min_width_2 = n_cols_2*label_size[1]
#   while min_width_2>sheet_size[0]:
#     n_cols_2 -= 1
#     min_width_2 = n_cols_2*label_size[1]  
#   if (ups_upper_lim-n_rows_1*n_cols_1)<=(ups_upper_lim-n_rows_2*n_cols_2): #取更接近ups_upper_lim的布局
#     return {'n_rows':n_rows_1, 'n_cols':n_cols_1, 'min_width':min_width_1, 'is_rotated':0}
#   else:
#     return {'n_rows':n_rows_2, 'n_cols':n_cols_2, 'min_width':min_width_2, 'is_rotated':1}    


# def one_dim_packer_for_more_color_more_dimension_for_one_sheetsize(min_width_dict, sheet_size, ink_seperator=30):
#   """
#   中离布局情景
#   根据每个label_size布局所需的最小宽度，以及sheet_width, 计算需要的最少sheet数量，以及每个sheet里面放哪些labels.
#   相当于一维装箱问题：装n个items，最少需要多少个箱子
#   """
#   # #计算每个label_size所需要的最小布局宽度
#   # min_width_dict = {}
#   # for label_size in label_dict.keys():
#   #   ups = label_dict[label_size]['ups']
#   #   fix_orientation = label_dict[label_size]['fix_orientation']    
#   #   min_width, is_rotated = get_min_width_for_one_color_one_dimension_for_one_sheetsize(label_size, sheet_size, ups, fix_orientation) ######
#   #   min_width_dict[label_size] = {'min_width':min_width, 'is_rotated':is_rotated}
  
#   #计算每个sheet中label size的分配 --- 一维装箱问题，但是还需要考虑中离导致sheet可用宽度减少
#   sheet_width=sheet_size[0]
#   sheet_index_dict = binningBestFit_with_inkSeperator(min_width_dict, sheet_width, ink_seperator)
#   return sheet_index_dict


def iterate_to_get_best_n_cols_allocation(label_w_list, n_cols_upper_lim, n_rows, re_qty, effective_sheet_width):
  """
  遍历所有情况获得columns分配的最优解
  """
  # print('iterate_to_get_best_n_cols_allocation')
  if len(label_w_list)==1:
    return n_cols_upper_lim[0]

  elif len(label_w_list)==2:
    min_pds = 1e6
    n_cols = [0]*len(label_w_list)
    for i in range(1,n_cols_upper_lim[0]+1):
      for j in range(1,n_cols_upper_lim[1]+1):
        cur_n_cols = [i,j]        
        label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
        if label_width_sum > effective_sheet_width: #无效解
          continue        
        ups_list = np.multiply(cur_n_cols, n_rows)
        pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
        if np.max(pds_list)<min_pds:
          min_pds = np.max(pds_list)
          n_cols = cur_n_cols

  elif len(label_w_list)==3:
    min_pds = 1e6
    n_cols = [0]*len(label_w_list)
    for i in range(1,n_cols_upper_lim[0]+1):
      for j in range(1,n_cols_upper_lim[1]+1):
        for k in range(1,n_cols_upper_lim[2]+1):        
          cur_n_cols = [i,j,k]        
          label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
          if label_width_sum > effective_sheet_width: #无效解
            continue        
          ups_list = np.multiply(cur_n_cols, n_rows)
          pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
          if np.max(pds_list)<min_pds:
            min_pds = np.max(pds_list)
            n_cols = cur_n_cols 

  elif len(label_w_list)==4:
    min_pds = 1e6
    n_cols = [0]*len(label_w_list)
    for i in range(1,n_cols_upper_lim[0]+1):
      for j in range(1,n_cols_upper_lim[1]+1):
        for k in range(1,n_cols_upper_lim[2]+1):      
          for l in range(1,n_cols_upper_lim[3]+1): 
            cur_n_cols = [i,j,k,l]        
            label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
            if label_width_sum > effective_sheet_width: #无效解
              continue        
            ups_list = np.multiply(cur_n_cols, n_rows)
            pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
            if np.max(pds_list)<min_pds:
              min_pds = np.max(pds_list)
              n_cols = cur_n_cols  

  elif len(label_w_list)==5:
    min_pds = 1e6
    n_cols = [0]*len(label_w_list)
    for i in range(1,n_cols_upper_lim[0]+1):
      print(i, 'be patient')
      for j in range(1,n_cols_upper_lim[1]+1):
        for k in range(1,n_cols_upper_lim[2]+1):      
          for l in range(1,n_cols_upper_lim[3]+1): 
            for m in range(1,n_cols_upper_lim[4]+1):            
              cur_n_cols = [i,j,k,l,m]        
              label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
              if label_width_sum > effective_sheet_width: #无效解
                continue        
              ups_list = np.multiply(cur_n_cols, n_rows)
              pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
              if np.max(pds_list)<min_pds:
                min_pds = np.max(pds_list)
                n_cols = cur_n_cols                                  

  elif len(label_w_list)==6:
    # print('iterate_to_get_best_n_cols_allocation - case = 6 dg')
    min_pds = 1e6
    n_cols = [0]*len(label_w_list)
    for i in range(1,n_cols_upper_lim[0]+1):
      print(i, 'be patient')
      for j in range(1,n_cols_upper_lim[1]+1):
        for k in range(1,n_cols_upper_lim[2]+1):      
          for l in range(1,n_cols_upper_lim[3]+1): 
            for m in range(1,n_cols_upper_lim[4]+1):    
              for n in range(1,n_cols_upper_lim[5]+1):                       
                cur_n_cols = [i,j,k,l,m,n]        
                label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
                if label_width_sum > effective_sheet_width: #无效解
                  continue        
                ups_list = np.multiply(cur_n_cols, n_rows)
                pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
                if np.max(pds_list)<min_pds:
                  min_pds = np.max(pds_list)
                  n_cols = cur_n_cols  
                n_cols = cur_n_cols                                  

  elif len(label_w_list)==7:
    # print('iterate_to_get_best_n_cols_allocation - case = 6 dg')
    min_pds = 1e6
    n_cols = [0]*len(label_w_list)
    for i in range(1,n_cols_upper_lim[0]+1):
      print(i, 'be patient')
      for j in range(1,n_cols_upper_lim[1]+1):
        for k in range(1,n_cols_upper_lim[2]+1):      
          for l in range(1,n_cols_upper_lim[3]+1): 
            for m in range(1,n_cols_upper_lim[4]+1):    
              for n in range(1,n_cols_upper_lim[5]+1):                       
                for o in range(1,n_cols_upper_lim[6]+1):   
                  cur_n_cols = [i,j,k,l,m,n,o]        
                  label_width_sum = sum(np.multiply(cur_n_cols, label_w_list))
                  if label_width_sum > effective_sheet_width: #无效解
                    continue        
                  ups_list = np.multiply(cur_n_cols, n_rows)
                  pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
                  if np.max(pds_list)<min_pds:
                    min_pds = np.max(pds_list)
                    n_cols = cur_n_cols  
  else:
    print('to add more codes to consider theis case')
    print(10/0)

  # print(f'iterate_to_get_best_n_cols_allocation ---> return n_cols = {n_cols}')
  return n_cols


def get_n_cols_for_dg_comb_on_one_sheetsize(dg_id,cg_id,label_w_list,label_h_list,re_qty,sheet_size,ink_seperator_width):
  """
  用于中离
  一列只有一个dg
  返回每个dg布局多少列
  """
  #基本信息
  n_dg = len(dg_id)
  n_cg = len(set(cg_id))
  n_ink_seperator = n_cg-1
  sheet_width = sheet_size[0]
  effective_sheet_width = sheet_width - n_ink_seperator*ink_seperator_width
  # print('n_cg, n_dg, n_ink_seperator, sheet_width, effective_sheet_width = ', n_cg, n_dg, n_ink_seperator, sheet_width, effective_sheet_width)

  # --- 在sheet_size上做layout ---
  n_rows = [int(sheet_size[1]/h) for h in label_h_list] #每一个dg的行数
  n_cols_upper_lim = [int(effective_sheet_width/w) for w in label_w_list] #每一个dg的最大列数

  # #初始解：按照tot_cols_init预分配 
  # tot_cols = np.min(n_cols_upper_lim) #初始总列数
  # n_cols = allocate_cols_based_on_qty(tot_cols, re_qty)
  # # print('n_dg, n_rows, n_cols_upper_lim, tot_cols, tot_qty, n_cols = ', n_dg, n_rows, n_cols_upper_lim, tot_cols, n_cols)

  # #逐步增大列数直至超出effective_sheet_width
  # label_width_sum = sum(np.multiply(n_cols, label_w_list))
  # temp_tot_cols = tot_cols
  # while label_width_sum < effective_sheet_width:
  #   temp_tot_cols += 1
  #   temp_n_cols = allocate_cols_based_on_qty(temp_tot_cols, re_qty)
  #   label_width_sum = sum(np.multiply(temp_n_cols, label_w_list))
  #   if label_width_sum <= effective_sheet_width: #有效解，更新结果
  #     tot_cols = temp_tot_cols
  #     n_cols = temp_n_cols

  #遍历所有情况获得columns分配的最优解
  n_cols = iterate_to_get_best_n_cols_allocation(label_w_list, n_cols_upper_lim, n_rows, re_qty, effective_sheet_width) #暂只考虑n_dg<=6的情况，若超出则需要在该方法中加情况
  # print(n_cols, n_rows)
  ups_list = list(np.multiply(n_cols, n_rows))
  pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
  # print(f'n_rows={n_rows}, n_cols={n_cols}, ups={ups_list}, pds={pds_list}')  
  # print(f'label_width_sum={label_width_sum}, effective_sheet_width={effective_sheet_width}')  

  return n_rows, n_cols, ups_list, pds_list


def get_best_sheetsize_for_one_dg_comb(sheet_size_list, dg_id,cg_id,label_w_list,label_h_list,re_qty,ink_seperator_width,mode='one_dg_one_column'):
  """
  使用get_n_cols_for_dg_comb_on_one_sheetsize遍历所有sheet_size，依据总面积最小的原则选择最优的sheet_size
  """
  min_tot_area = 1e12
  best_sheet = sheet_size_list[0]
  best_res = {}
  for sheet_size in sheet_size_list:
    # print(f'--- sheet_size={sheet_size} ---')
    if mode=='one_dg_one_column':
      n_rows, n_cols, ups_list, pds_list = get_n_cols_for_dg_comb_on_one_sheetsize(dg_id,cg_id,label_w_list,label_h_list,re_qty,sheet_size,ink_seperator_width) ###--->>>
    elif mode=='mul_dg_one_column':
      # n_rows, n_cols, ups_list, pds_list = get_n_cols_for_dg_comb_on_one_sheetsize(dg_id,cg_id,label_w_list,label_h_list,re_qty,sheet_size,ink_seperator_width) ###--->>>
      stop_flag = 1/0
    else:
      print(f'unrecognized OCMD_layout_mode = {OCMD_layout_mode}')   
      stop_flag = 1/0      
    tot_area = sheet_size[0]*sheet_size[1]*np.max(pds_list)
    if tot_area<min_tot_area:
      min_tot_area = tot_area
      best_sheet = sheet_size
      best_res['re_qty'] = re_qty      
      best_res['n_rows'] = n_rows
      best_res['n_cols'] = n_cols
      best_res['ups'] = ups_list
      best_res['pds'] = pds_list
  # print(f'best_sheet={best_sheet}, best_res={best_res}, min_tot_area={min_tot_area}')
  return best_sheet, best_res, min_tot_area


def iterate_to_solve_min_total_sheet_area(sheet_size_list,comb_names,comb_res_w,comb_res_h,dg_id,cg_id,re_qty,ink_seperator_width,check_criteria=False,criteria_dict=None,mode='one_dg_one_column'):
  """
  main
  """
  #初始化
  best_comb = 'dummy'
  best_sheet = [999,999]
  min_tot_area = 999999999999
  best_res = {'re_qty': [0, 0], 'n_rows': [0, 0], 'n_cols': [0, 0], 'ups': [0,  0], 'pds': [0, 0]}

  #遍历所有comb和sheet_size，选择总面积最小的comb+sheet_size的组合
  print(f'total # of comb = {len(comb_names)}')  
  for comb_index in range(len(comb_names)):
    comb_name = comb_names[comb_index]  
    print(f'calculating comb {comb_index} = {comb_name}')
    label_w_list = comb_res_w[comb_index].split('<+>')
    label_w_list = [float(w) for w in label_w_list]
    label_h_list = comb_res_h[comb_index].split('<+>')
    label_h_list = [float(h) for h in label_h_list]

    #对当前comb，遍历所有sheet_size，选择总面积最小的sheet_size
    sheet, res, tot_area = get_best_sheetsize_for_one_dg_comb(sheet_size_list, dg_id,cg_id,label_w_list,label_h_list,re_qty,ink_seperator_width,mode) ###--->>>

    #判断解是否符合criteria
    if check_criteria:
      sheet_size_name = str(int(sheet[0]))+'<+>'+str(int(sheet[1]))
      pds_lower_lim = criteria_dict[sheet_size_name]['pds']
      pds_value = np.max(res['pds'])
      if pds_value < pds_lower_lim: #fail criteria
        continue

    #如果是当前更优解，更新结果
    if tot_area<min_tot_area:
      best_comb = comb_name
      best_sheet = sheet   
      min_tot_area = tot_area
      best_res = res
  # print()
  # print(f'best_comb={best_comb}, best_sheet={best_sheet}, best_res={best_res}, min_tot_area={min_tot_area}')
  return best_comb, best_sheet, best_res, min_tot_area