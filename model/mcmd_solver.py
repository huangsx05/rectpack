import numpy as np
import pandas as pd
from utils.tools import allocate_cols_based_on_qty


def iterate_to_get_best_n_cols_allocation(label_w_list, n_cols_search_lower, n_cols_search_upper, n_cols_upper_lim, 
                                          n_rows, re_qty, effective_sheet_width):
  """
  遍历所有情况获得columns分配的最优解
  """
  # print('iterate_to_get_best_n_cols_allocation')
  if len(label_w_list)==1:
    return n_cols_upper_lim[0]

  elif len(label_w_list)==2:
    min_pds = 1e6
    n_cols = [0]*len(label_w_list)
    for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
      for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
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
    for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
      for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
        for k in range(n_cols_search_lower[2],n_cols_search_upper[2]+1):        
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
    for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
      for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
        for k in range(n_cols_search_lower[2],n_cols_search_upper[2]+1):      
          for l in range(n_cols_search_lower[3],n_cols_search_upper[3]+1): 
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
    for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
      print(i, 'be patient')
      for j in range(n_cols_search_lower[1],n_cols_search_upper[1]+1):
        for k in range(n_cols_search_lower[2],n_cols_search_upper[2]+1):      
          for l in range(n_cols_search_lower[3],n_cols_search_upper[3]+1): 
            for m in range(n_cols_search_lower[4],n_cols_search_upper[4]+1):            
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
    for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
      print(i, 'be patient')
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
                pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
                if np.max(pds_list)<min_pds:
                  min_pds = np.max(pds_list)
                  n_cols = cur_n_cols  
                n_cols = cur_n_cols                                  

  elif len(label_w_list)==7:
    # print('iterate_to_get_best_n_cols_allocation - case = 6 dg')
    min_pds = 1e6
    n_cols = [0]*len(label_w_list)
    for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
      print(i, 'be patient')
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
                  pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
                  if np.max(pds_list)<min_pds:
                    min_pds = np.max(pds_list)
                    n_cols = cur_n_cols  

  elif len(label_w_list)==8:
    # print('iterate_to_get_best_n_cols_allocation - case = 6 dg')
    min_pds = 1e6
    n_cols = [0]*len(label_w_list)
    for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
      print(i, 'be patient')
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
                    pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
                    if np.max(pds_list)<min_pds:
                      min_pds = np.max(pds_list)
                      n_cols = cur_n_cols      

  elif len(label_w_list)==9:
    # print('iterate_to_get_best_n_cols_allocation - case = 6 dg')
    min_pds = 1e6
    n_cols = [0]*len(label_w_list)
    for i in range(n_cols_search_lower[0],n_cols_search_upper[0]+1):
      print(i, 'be patient')
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
  采用遍历的方法
  """
  #基本信息
  n_dg = len(dg_id)
  n_cg = len(set(cg_id))
  n_ink_seperator = n_cg-1
  sheet_width = sheet_size[0]
  effective_sheet_width = sheet_width - n_ink_seperator*ink_seperator_width
  # print('dg_id, n_cg, n_dg, n_ink_seperator, sheet_width, ink_seperator_width, effective_sheet_width = ', dg_id, n_cg, n_dg, n_ink_seperator, sheet_width, ink_seperator_width, effective_sheet_width)

  # --- 在sheet_size上做layout ----------------------------------------------
  n_rows = [int(sheet_size[1]/h) for h in label_h_list] #每一个dg的行数
  n_cols_upper_lim = [int(effective_sheet_width/w) for w in label_w_list] #每一个dg的最大列数

  #初始解：按照tot_cols_init预分配 
  tot_cols = np.min(n_cols_upper_lim) #初始总列数
  n_cols = allocate_cols_based_on_qty(tot_cols, re_qty)
  print(f'n_cols heuristics initial solution = {n_cols}')

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
  print(f'n_cols heuristics final solution = {n_cols}')

  tolerance = 2
  n_cols_upper_lim = n_cols
  n_cols_search_upper = [int(i+tolerance) for i in n_cols_upper_lim]
  n_cols_search_lower = [int(np.max([i-tolerance,1])) for i in n_cols_upper_lim]
  #---------------------------------------------------------------------------

  #遍历所有情况获得columns分配的最优解
  n_cols = iterate_to_get_best_n_cols_allocation(label_w_list, n_cols_search_lower, n_cols_search_upper, n_cols_upper_lim,
                                                 n_rows, re_qty, effective_sheet_width) #暂只考虑n_dg<=6的情况，若超出则需要在该方法中加情况
  # print(n_cols, n_rows)
  ups_list = list(np.multiply(n_cols, n_rows))
  pds_list = [np.ceil(a/b) for a, b in zip(re_qty, ups_list)]
  # print(f'n_rows={n_rows}, n_cols={n_cols}, ups={ups_list}, pds={pds_list}')  
  # print(f'label_width_sum={label_width_sum}, effective_sheet_width={effective_sheet_width}')  

  return n_rows, n_cols, ups_list, pds_list