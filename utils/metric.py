import numpy as np

# def cal_pds(qty, u):
#   N = len(qty)
#   n_slide = [np.ceil(qty[i]/u[i]) for i in range(N)]
#   n_pds = np.max(n_slide)
#   return n_pds

# def cal_overproduction(qty, u):
#   N = len(qty)
#   n_pds = cal_pds(qty, u)
#   n_produce = [u[i]*n_pds-qty[i] for i in range(N)]
#   result = np.sum(n_produce)/np.sum(qty)
#   return result

# def cal_scrap(ups_list, label_size_list, sheet_size):
#   sheet_area = sheet_size[0]*sheet_size[1]
#   total_label_area = 0
#   for i in range(len(ups_list)):
#     total_label_area += ups_list[i]*label_size_list[i][0]*label_size_list[i][1]
#   scrap = (sheet_area-total_label_area)/sheet_area
#   return round(scrap,3)