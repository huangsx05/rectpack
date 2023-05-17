# import re
import numpy as np
# import pandas as pd
# from rectpack import newPacker

# -------------------------------------
# ------ 2. ONE COLOR MORE DIMENSION ------
# -------------------------------------

# ------ 1.1 filter ------

def filter_id_by_criteria_ocmd(df_2, cg_id, sheet_size_list, criteria_dict):
  fail_criteria = True
  df_temp = df_2[df_2['cg_id']==cg_id]
  sum_qty = df_temp['re_qty'].sum()
  max_label_area = df_temp['label_area'].max()
  # print(f'{cg_id}, sum_qty={sum_qty}, max_label_area={max_label_area}')
  for sheet_size in sheet_size_list:
    sheet_area = int(sheet_size[0])*int(sheet_size[1])
    min_ups = int(sheet_area/max_label_area)
    max_pds = np.ceil(sum_qty/min_ups)
    sheet_size_name = str(int(sheet_size[0]))+'<+>'+str(int(sheet_size[1]))
    if max_pds>=criteria_dict[sheet_size_name]['pds']:
      fail_criteria = False
      break