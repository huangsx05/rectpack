from ocmd_solver import *
from mcmd_solver import *

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
      n_rows, n_cols, ups_list, pds_list = get_n_cols_for_dg_comb_on_one_sheetsize(dg_id,cg_id,label_w_list,label_h_list,re_qty,sheet_size,ink_seperator_width) ###--->>>
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
  for comb_index in range(len(comb_names)):
    comb_name = comb_names[comb_index]
    # print(f'----- comb = {comb_name} ------')
    label_w_list = comb_res_w[comb_index].split('<+>')
    label_w_list = [float(w) for w in label_w_list]
    label_h_list = comb_res_h[comb_index].split('<+>')
    label_h_list = [float(h) for h in label_h_list]

    #对当前comb，遍历所有sheet_size，选择总面积最小的sheet_size
    sheet, res, tot_area = get_best_sheetsize_for_one_dg_comb(sheet_size_list, dg_id,cg_id,label_w_list,label_h_list,re_qty,ink_seperator_width) ###--->>>

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