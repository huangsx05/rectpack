# import re
import numpy as np
import pandas as pd
# from rectpack import newPacker

# -------------------------------------
# ------ 1. ONE COLOR ONE DIMENSION ------
# -------------------------------------

# ------ 1.1 filter ------

def filter_id_by_criteria_ocod(label_width, label_length, re_qty, sheet_size_list, criteria_dict):
  fail_criteria = True
  for sheet_size in sheet_size_list:
    temp_ups_1 = int(sheet_size[0]/label_width)*int(sheet_size[1]/label_length)
    temp_ups_2 = int(sheet_size[1]/label_length)*int(sheet_size[0]/label_width)    
    temp_ups = np.min([temp_ups_1, temp_ups_2])
    max_pds = np.ceil(re_qty/temp_ups)
    sheet_size_name = str(int(sheet_size[0]))+'<+>'+str(int(sheet_size[1]))
    if int(max_pds)>=int(criteria_dict[sheet_size_name]['pds']):
      fail_criteria = False
      break
  return fail_criteria

# ------ 1.2 layout ------

def sheet_size_selection_ocod(sheet_size_list, label_size, re_qty, fix_orientation):
  """
  step 1: sheet size selection的主方法
  遍历sheet_size_list，选择pds*sheet_area最小的sheet_size
  for OCOD case，一个batch只有一个DG，所以ups = sum(ups), 无需update sheet size selection的criteria
  """
  res={}
  #计算ups
  for sheet_size in sheet_size_list:
    print(f'calculating sheet_size {sheet_size}, re_qty={re_qty}')
    temp_res, temp_is_rotated = layout_one_color_one_dimension_in_one_sheet(label_size, sheet_size, fix_orientation) ###--->>>
    temp_ups = temp_res['n_rows']*temp_res['n_cols']+temp_res['n_rows_rotated']*temp_res['n_cols_rotated']
    temp_pds = np.ceil(re_qty/temp_ups)
    print(f"res = {res}")
    print(f"temp_res = {temp_res}")
    res[str(sheet_size)] = temp_res
    res[str(sheet_size)]['is_rotated'] = temp_is_rotated    
    res[str(sheet_size)]['ups'] = temp_ups
    res[str(sheet_size)]['pds'] = temp_pds
    res[str(sheet_size)]['tol_area'] = temp_pds*sheet_size[0]*sheet_size[1]
  
  #选择sheet_size
  # print(res)
  best_sheet_size = sheet_size_list[0]
  cur_tol_area = res[str(best_sheet_size)]['tol_area']
  for sheet_size in sheet_size_list[1:]:
    if res[str(sheet_size)]['tol_area']<cur_tol_area:
      cur_tol_area = res[str(sheet_size)]['tol_area']
      best_sheet_size = sheet_size
  print(f'best_sheet_size, res[str(best_sheet_size) = {best_sheet_size}, {res[str(best_sheet_size)]}')
  return best_sheet_size, res[str(best_sheet_size)]


def layout_one_color_one_dimension_in_one_sheet(label_size, sheet_size, fix_orientation):
  """
  step 2: 在一个sheet上layout的主方法
  :param label_size: [label_width, label_length], label_width>=label_length
  :param sheet_size: [sheet_width, sheet_length], sheet_width>=sheet_length, sheet必须横放
  :param fix_orientation: if label can be rotated
  :return res: res = {'n_rows':0, 'n_cols':0, 'n_rows_rotated':0, 'n_cols_rotated':0}
  """
  if fix_orientation==1:
    res, is_rotated = layout_fix_orientation_one_dimension_in_one_sheet(label_size, sheet_size) ###--->>>
  else:
    res, is_rotated = layout_rotatable_one_dimension_in_one_sheet(label_size, sheet_size) ###--->>>
  return res, is_rotated


def layout_fix_orientation_one_dimension_in_one_sheet(label_size, sheet_size):
  """
  for fix_orientation == 1
  :param label_size: [label_width, label_length]
  :param sheet_size: [sheet_width, sheet_length], sheet_width>=sheet_length, sheet必须横放
  """
  print(f'fix_orientation==1, layout_fix_orientation_one_dimension_in_one_sheet')  
  n_rows = int(sheet_size[1]/label_size[1])  
  n_cols = int(sheet_size[0]/label_size[0])
  is_rotated = False
  return {'n_rows':n_rows, 'n_cols':n_cols, 'n_rows_rotated':0, 'n_cols_rotated':0}, is_rotated


def layout_rotatable_one_dimension_in_one_sheet(label_size, sheet_size):
  """
  for fix_orientation == 0，遍历旋转和不旋转两种情况  
  :param label_size: [label_width, label_length], label_width>=label_length
  :param sheet_size: [sheet_width, sheet_length], sheet_width>=sheet_length, sheet必须横放
  """
  print(f'fix_orientation==0, layout_rotatable_one_dimension_in_one_sheet')      
  #情况1：旋转前
  res, n_ups = layout_rotatable_one_dimension_in_one_sheet_single_step(label_size, sheet_size)

  #情况2：旋转后
  label_size_2 = [label_size[1],label_size[0]]
  res_temp, n_ups_2 = layout_rotatable_one_dimension_in_one_sheet_single_step(label_size_2, sheet_size)
  res_2 = {'n_rows': res_temp['n_rows_rotated'], 'n_cols': res_temp['n_cols_rotated'], 
          'n_rows_rotated': res_temp['n_rows'], 'n_cols_rotated': res_temp['n_cols']}  

  #输出结果
  print(f'label_size = {label_size}, rotate condition 1 = {res}')
  print(f'label_size = {label_size_2}, rotate condition 2 = {res_2}')
  if n_ups >= n_ups_2:
    is_rotated = False
    return res, is_rotated
  else:
    is_rotated = True
    return res_2, is_rotated
    

def layout_rotatable_one_dimension_in_one_sheet_single_step(label_size, sheet_size):
  """
  :param label_size: [label_width, label_length], label_width>=label_length
  :param sheet_size: [sheet_width, sheet_length], sheet_width>=sheet_length, sheet必须横放
  """
  #情况1：旋转前
  n_rows = int(sheet_size[1]/label_size[1])  
  n_cols = int(sheet_size[0]/label_size[0])
  res = {'n_rows':n_rows, 'n_cols':n_cols, 'n_rows_rotated':0, 'n_cols_rotated':0}
  #判断是否能增加列
  if label_size[0]>label_size[1]:
    margin_width = sheet_size[0] - n_cols*label_size[0]
    n_add_cols = int(margin_width/label_size[1])
    n_add_rows = 0
    if n_add_cols>0:
      n_add_rows = int(sheet_size[1]/label_size[0])
      res['n_rows_rotated'] = n_add_rows
      res['n_cols_rotated'] = n_add_cols  
  elif label_size[0]<label_size[1]:
    margin_length = sheet_size[1] - n_rows*label_size[1]
    n_add_rows = int(margin_length/label_size[0])
    n_add_cols = 0
    if n_add_rows>0:
      n_add_cols = int(sheet_size[0]/label_size[1])
      res['n_rows_rotated'] = n_add_rows
      res['n_cols_rotated'] = n_add_cols  
  n_ups = n_cols*n_rows + n_add_cols*n_add_rows
  return res, n_ups


def check_criteria_ocod(x, criteria_dict):
  sheet_size_name = str(int(x.sheet_width))+'<+>'+str(int(x.sheet_length))
  pds_thres = criteria_dict[sheet_size_name]['pds']
  return x.pds>=pds_thres


# ------ 1.3 allocate sku ------

# def allocate_sku_ocod(sku_qty_dict, n_ups):
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


# ---------------------
# ------ 排版方法 ------
# ---------------------


# def pack_rectangles_in_one_bin(rect_size_list, bin_size, qty_list=[0], rotation=True):
#   """
#   :param rec_size_list: [[x1,y1], [x2,y2], ...], x<=y
#   :param bin_size: 版的尺寸 (x,y), x<=y
#   :param qty_list: [qty1, qty2, ...]   
#   :return ups: ups
#   :return all_rect: 所有矩形的坐标和方向
#   """
#   bins = [bin_size]
#   sum_qty = np.sum(qty_list)

#   #初始化ups
#   pre_n_x = int(max(bin_size)/max([max(i) for i in rect_size_list]))
#   pre_n_y = int(min(bin_size)/max([min(i) for i in rect_size_list]))
#   pre_ups = pre_n_x*pre_n_y-1

#   #尝试在初始ups的基础上增加ups
#   add_next_rec = True
#   while add_next_rec:
#     #根据qty的比例分配不同dimensions的ups
#     if len(rect_size_list)==1:
#       rectangles = [rect_size_list[0]]*pre_ups
#     else:
#       rectangles = []
#       ups_list = []
#       for i in range(len(rect_size_list)-1):
#         ups_list.append(int(round(pre_ups*qty_list[i]/(sum_qty)))) 
#       pre_ups_last = pre_ups - np.sum(ups_list)
#       ups_list.append(pre_ups_last) 
#       for i in range(len(rect_size_list)):    
#         rectangles += [rect_size_list[i]]*ups_list[i]
#     #开始pack
#     pack = newPacker(rotation=rotation)
#     for r in rectangles:
#         pack.add_rect(*r)
#     for b in bins:
#         pack.add_bin(*b)
#     pack.pack()
#     #本轮结果
#     all_rects = pack.rect_list()
#     cur_ups = len(all_rects)
#     #决定是否继续增加ups
#     print(f'ups = {pre_ups},{cur_ups}')    
#     if cur_ups<pre_ups:
#       add_next_rec = False
#     else:
#       pre_ups+=1

#   return cur_ups, all_rects

# # -------------------------------------
# # ------ 2. ONE COLOR MORE DIMENSION ------
# # -------------------------------------



# #------

# def allocate_sku_one_color_multi_dimension(dim_sku_qty_dict, dim_ups_dict, sheet_size, color_dot_size=[0,0]):
#   """
#   :param dim_sku_qty_dict: {[label_width, label_length]:{sku:qty}} 
#   :param dim_ups_dict: {[label_width, label_length]:n_ups}
#   :param sheet_size: [sheet_width, sheet_length]
#   :param color_dot_size: [dot_width, dot_length]
#   """
#   #预排
#   #变量名与excel表对应
#   dim_list = list(dim_sku_qty_dict.keys())
#   dim_dg_qty = [np.sum(list(sku_dict.values())) for sku_dict in list(dim_sku_qty_dict.values())] #每个dg的qty
#   dim_skus = [len(sku_dict) for sku_dict in list(dim_sku_qty_dict.values())] #以每个dg对应的sku数量作为该dg的ups初始值
  
#   dim_l_rows = [int(sheet_size[0]/dim[1]) for dim in dim_list]
#   dim_w_cols = [1]*len(dim_list)
#   for i in range(len(dim_skus)): #保证初始ups>=最少需要的ups
#     while dim_l_rows[i]*dim_w_cols[i]<dim_skus[i]:
#       dim_w_cols[i]+=1

#   #初始ups分配
#   dim_film_area_width_aft = cal_film_area_width_aft(sheet_size, dim_list, dim_w_cols) 
#   dim_dg_ups = [a*b for a,b in zip(dim_l_rows, dim_w_cols)]
#   dim_dg_pds = [np.ceil(a/b) for a,b in zip(dim_qty_list, dim_dg_ups)]  

#   #循环计算，不断增加column直到剩余宽度<0
#   while dim_film_area_width_aft[-1]>=np.min([dim[0] for dim in dim_list]):
#     imax = dim_dg_ups.index(max(dim_dg_ups))
#     dim_w_cols[imax] += 1  
#     dim_film_area_width_aft = cal_film_area_width_aft(sheet_size, dim_list, dim_w_cols) 
#     if dim_film_area_width_aft[-1]>=0:
#       dim_dg_ups = [a*b for a,b in zip(dim_l_rows, dim_w_cols)]
#       dim_dg_pds = [np.ceil(a/b) for a,b in zip(dim_qty_list, dim_dg_ups)]  

#   #若有color dot, 需要预留位置
#   if color_dot_size[0]>0:
#     reserve_color_dot = True
#     for i in range(len(dim_list)):
#       if (dim_w_cols[i]>1) and (dim_film_area_length_aft[i]>=color_dot_size[1]): #这里考虑color dot不可转置
#         reserve_color_dot = False
#         break
#     print(f'reserve_color_dot = {reserve_color_dot}')
#     #假如剩余空间不足以放color dot，需要减少ups
#     if reserve_color_dot == True:
#       dim_dg_ups_temp = [ups-1 for ups in dim_dg_ups]
#       imin = dim_dg_ups_temp.index(min(dim_dg_ups_temp))
#       dim_dg_ups[imin] -= 1 #为color dot留出位置
#       dim_dg_pds = [np.ceil(a/b) for a,b in zip(dim_qty_list, dim_dg_ups)]
  
#   # ------ 至此，为每一个DG分配好了UPS ------
#   # ------ 以下为在每一个DG内部进行sku的ups分配 ------
#   return res_dict

# # ---------------------
# # ------ 辅助方法 ------
# # ---------------------

# def cal_film_area_width_aft(sheet_size, dim_list, dim_w_cols):
#   dim_film_area_width_aft = [sheet_size[0]-dim[0][0]*dim_w_cols[0]]
#   for i in range(1,len(dim_list)):
#     width_reduction = dim[i][0]*dim_w_cols[i]
#     width_aft = dim_film_area_width_aft[i-1]-width_reduction
#     dim_film_area_width_aft.append(width_aft)
#   return dim_film_area_width_aft


# def cal_film_area_length_aft(sheet_size, dim_list, dim_l_rows):
#   dim_film_area_length_aft = [sheet_size[1]-dim[0][1]*dim_l_rows[0]]
#   for i in range(1,len(dim_list)):
#     length_reduction = dim[i][1]*dim_l_rows[i]
#     length_aft = dim_film_area_length_aft[i-1]-length_reduction
#     dim_film_area_length_aft.append(length_aft)
#   return dim_film_area_length_aft