import numpy as np

def allocate_cols_based_on_qty(tot_n_cols, qty_list):
  tot_qty = np.sum(qty_list)
  n_cols = [np.max([round(qty/tot_qty*tot_n_cols),1]) for qty in qty_list] #每个dg的初始列数, 并保证至少有一列
  n_max_index = np.argmax(n_cols)
  # print(f'n_max_index={n_max_index}')
  updated_n_col = tot_n_cols - np.sum(n_cols[:n_max_index]) - np.sum(n_cols[n_max_index+1:])
  n_cols[n_max_index] = updated_n_col #调整最多列数的dg，保证列数总和等于tot_n_cols
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