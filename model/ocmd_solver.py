# import re
import numpy as np
# import pandas as pd
# from rectpack import newPacker

# -------------------------------------
# ------ 2. ONE COLOR MORE DIMENSION ------
# -------------------------------------

# ------ 1.1 filter ------

def filter_id_by_criteria_ocmd(df_2, cg_id, sheet_size_list, criteria_dict):
  """
  通过找pds的上限（ups的下限），判断是否有可能满足最小pds的要求
  """
  fail_criteria = True
  df_temp = df_2[df_2['cg_id']==cg_id]
  sum_qty = df_temp['re_qty'].sum()
  max_label_area = df_temp['label_area'].max() #基于area判断是不对的
  1/0 #需要改正
  # print(f'{cg_id}, sum_qty={sum_qty}, max_label_area={max_label_area}')
  for sheet_size in sheet_size_list:
    sheet_area = int(sheet_size[0])*int(sheet_size[1])
    min_ups = int(sheet_area/max_label_area)
    max_pds = np.ceil(sum_qty/min_ups)
    sheet_size_name = str(int(sheet_size[0]))+'<+>'+str(int(sheet_size[1]))
    if max_pds>=criteria_dict[sheet_size_name]['pds']:
      fail_criteria = False
      break



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

    #step 1: 如果已经有col, 先往每个col上方的剩余空间塞
    # if index!=0:
    i = 0
    while n_ups>0:
    # for i in range(max_n_cols): #对每一列
      # print(i, max_n_cols)
      add_n_ups = np.min([int(left_col_h[i]/label_h),n_ups]) #该列剩余空间可放的ups数量
      if add_n_ups==0: #如果当前列放不下，尝试下一列
        i+=1
        if i >= max_n_cols: #max列数不足以放下ups
          can_layout = False
          break

      # #---------------------------------------------
      # # 可提升空间：这部分可以先省略，找到最优解之后再计算
      if add_n_ups>0:
      #   #添加新增ups的左下角坐标
      #   if i==0:
      #     x = 0
      #   else:
      #     x = np.sum(used_col_w[:i])
      #   y = used_col_h[i]

        for n in range(add_n_ups): #每一个新增的ups
      #     x_n = x
      #     y_n = y+n*label_h
      #     ups_info = [x_n,y_n,label_w,label_h,dg_i]
      #     # print(ups_info_by_col)
      #     # print(ups_info)
      #     ups_info_by_col[i].append(ups_info)
      #     #更新layout信息
          used_col_h[i] += label_h
          left_col_h[i] -= label_h
      # #-----------------------------------------------
      
      used_col_w[i] = np.max([used_col_w[i], label_w])
      n_ups = np.max([(n_ups-add_n_ups),0])
      # if n_ups==0:
      #   break

    # #step 2: 开新的full height columns
    # max_n_rows = int(sheet_size[1]/label_h)
    # full_n_cols = int(n_ups/max_n_rows)
    # left_n_ups = n_ups-full_n_cols*max_n_rows

    # if left_n_ups>0:
    #   used_col_w += [label_w]*(full_n_cols+1) #剩余ups多占用一个col，所以+1
    #   used_col_h += [max_n_rows*label_h]*full_n_cols #full columns
    #   used_col_h.append(left_n_ups*label_h) #剩余ups所在的column
    #   #判断是否超出sheet_size
    #   left_sheet_w = sheet_size[0] - np.sum(used_col_w)
    #   if left_sheet_w<0:
    #     return False, []         
    # else:
    #   used_col_w += [label_w]*full_n_cols
    #   used_col_h += [max_n_rows*label_h]*full_n_cols  
    #   #判断是否超出sheet_size
    #   left_sheet_w = sheet_size[0] - np.sum(used_col_w)
    #   if left_sheet_w<0:
    #     return False, []            

    # #如果尚未超出sheet_size，更新当前结果
    # left_col_h = [sheet_size[0]-h for h in used_col_h] 
    # #添加整列的ups的info  
    # x_pre = np.sum(used_col_w)
    # col_cnt_pre = len(used_col_w)    
    # for n_c in range(full_n_cols):   
    #   x_n = x_pre+n_c*label_w
    #   y_pre = 0
    #   for n_r in range(max_n_rows):           
    #     y_n = y_pre+n_r*label_h
    #     ups_info = [x_n,y_n,label_w,label_h,dg_id]
    #     ups_info_by_col[col_cnt_pre+n_r].append(ups_info)  

    # if left_n_ups>0:              
    #   #添加剩余的ups的info
    #   x_pre_update = np.sum(used_col_w)
    #   y_pre_update = 0
    #   for n_u in range(left_n_ups):
    #     x_n = x_pre_update
    #     y_n = y_pre_update+n_u*label_h
    #     ups_info = [x_n,y_n,label_w,label_h,dg_id]
    #     ups_info_by_col[col_cnt_pre+full_n_cols].append(ups_info) 

  return can_layout#, ups_info_by_col


def iterate_to_get_best_ups_list_allocation(sheet_size,dg_id,re_qty,label_w_list,label_h_list):
  """
  遍历所有情况获得ups分配的最优解
  """
  sheet_area = sheet_size[0]*sheet_size[1]
  label_area_list = np.multiply(label_w_list,label_h_list)
  
  # --- 在sheet_size上做layout ---
  n_rows_upper_lim = [int(sheet_size[1]/h) for h in label_h_list] #每一个dg的行数
  n_cols_upper_lim = [int(sheet_size[0]/w) for w in label_w_list] #每一个dg的最大列数
  n_ups_upper_lim = np.multiply(n_rows_upper_lim, n_cols_upper_lim)

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
    for i in range(1,n_ups_upper_lim[0]+1):
      # print(f'iterating {i} --- iterate_to_get_best_ups_list_allocation ---')
      for j in range(1,n_ups_upper_lim[1]+1): ###多一层循环需要改这
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


def get_ups_layout_for_ocmd_comb_on_one_sheetsize(dg_id,label_w_list,label_h_list,re_qty,sheet_size):
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

  #初始解：按照tot_cols_init预分配 
  # tot_ups_init = np.min(np.multiply(n_rows_upper_lim, n_cols_upper_lim)) #初始总列数
  # n_cols = allocate_cols_based_on_qty(tot_cols, re_qty)
  # print('n_dg, n_rows, n_cols_upper_lim, tot_cols, tot_qty, n_cols = ', n_dg, n_rows, n_cols_upper_lim, tot_cols, n_cols)

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

  #遍历所有情况获得ups分配的最优解
  ups_list, pds_list, ups_info_by_col = iterate_to_get_best_ups_list_allocation(sheet_size,dg_id,re_qty,label_w_list,label_h_list) #每一个dg分配多少个ups
  # print(n_cols, n_rows)
  # ups_list = list(np.multiply(n_cols, n_rows))
  # pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
  # print(f'n_rows={n_rows}, n_cols={n_cols}, ups={ups_list}, pds={pds_list}')  
  # print(f'label_width_sum={label_width_sum}, effective_sheet_width={effective_sheet_width}')  

 # dg_id重排序，以和输入时的dg_id排序一致
  dg_layout_seq = list(range(len(dg_id))) #存储dg的layout排序，利于之后画图 
  # print('before output sorting: ',dg_index_list,dg_id,dg_layout_seq,ups_list,pds_list)  
  zipped_output = zip(dg_index_list,dg_id,dg_layout_seq,ups_list,pds_list)
  zipped_output_sorted = sorted(zipped_output,reverse=False)
  dg_index_list,dg_id,dg_layout_seq,ups_list,pds_list = zip(*zipped_output_sorted)
  # print('after output sorting: ',dg_index_list,dg_id,dg_layout_seq,ups_list,pds_list)  

  return list(ups_list), list(pds_list), list(dg_layout_seq), ups_info_by_col