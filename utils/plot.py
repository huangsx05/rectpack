import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_rectangle(x,y,w,h,color='blue'):
  """
  plot a rectangle
  x,y: origin
  w,h: width and height of rectangle
  """
  x1, x2, x3, x4, x5 = x, x+w, x+w, x, x
  y1, y2, y3, y4, y5 = y, y, y+h, y+h, y  
  plt.plot([x1,x2,x3,x4,x5], [y1,y2,y3,y4,y5], color)

def plot_rectangle_array(x,y,w,h,n_rows,n_cols,color):
  """
  plot 矩形阵列
  """
  for i in range(n_cols): 
    x1 = x+w*i
    for j in range(n_rows):
      y1 = y+h*j
      plot_rectangle(x1,y1,w,h,color)


def plot_n_rectangle_full_sheet_height(x,y,w,h,w_sheet,h_sheet,n_rec,color):
  """
  plot 矩形阵列, 按列排，一列排满后再排下一列
  x,y: origin
  w,h: width and height of rectangle
  w_sheet,h_sheet: width and height of sheet
  n_rec: rectangle数量  
  """
  plot_rectangle(x,y,w,h,color) #plot第一个rec
  for i in range(1,int(n_rec)): 
    y = y+h
    # print(f'y={y}')
    # print(f'h={h}')
    # print(f'h_sheet={h_sheet}')
    if y+h>h_sheet:
      y = 0
      x = x+w
    plot_rectangle(x,y,w,h,color)
  # print(x,y+h)


def plot_ups_ocod(label_size, sheet_size, layout_dict, scale=60):
  """
  :param label_size: [label_width, label_length]
  :param sheet_size: [sheet_width, sheet_length], sheet_width>=sheet_length, sheet必须横放  
  :param layout_dict: {'n_rows': 12, 'n_cols': 6, 'n_rows_rotated': 0, 'n_cols_rotated': 0}
  """
  colors = list(mcolors.TABLEAU_COLORS.values())
  # print(f'colors = {colors}')

  label_size = [i/scale for i in label_size]
  sheet_size = [i/scale for i in sheet_size]
  # print(f'after scale, label_size={label_size}')           
  # plt.figure(figsize=(sheet_size[0], sheet_size[1]))

  #plot sheet
  x,y = 0,0 #原点
  w = sheet_size[0]
  h = sheet_size[1]  
  color = colors[0]
  plot_rectangle(x,y,w,h,color)

  #plot un-rotated labels
  color = colors[1]
  x,y = 0,0
  w = label_size[0]
  h = label_size[1] 
  plot_rectangle_array(x,y,w,h,layout_dict['n_rows'],layout_dict['n_cols'],color)

  #plot rotated labels
  if layout_dict['n_rows_rotated']>0:
    if label_size[0]>label_size[1]: #添加列
      x = layout_dict['n_cols']*label_size[0]
      y = 0
      print(f'plot rotated labels on the right, x,y={x},{y}')      
    else: #添加行
      x = 0
      y = layout_dict['n_rows']*label_size[1]
      # print(f'plot rotated labels on the top, x,y={x},{y}')            
    print(x,y,h,w,layout_dict['n_rows_rotated'],layout_dict['n_cols_rotated'],color)
    plot_rectangle_array(x,y,h,w,layout_dict['n_rows_rotated'],layout_dict['n_cols_rotated'],color) #注意在这里w和h互换
  
  # plt.show()


def plot_full_height_for_each_dg_with_ink_seperator(sheet_size, ink_seperator_width, dg_id, cg_id, label_w_list, label_h_list, n_cols, ups_list):
  colors = list(mcolors.TABLEAU_COLORS.values())
  #plot sheet
  plot_rectangle(x=0,y=0,w=sheet_size[0],h=sheet_size[1],color='black') 
  #plot第一个cg
  x = 0
  plot_color_index = 0
  plot_n_rectangle_full_sheet_height(x=x,y=0,w=label_w_list[0],h=label_h_list[0],w_sheet=sheet_size[0],h_sheet=sheet_size[1],n_rec=ups_list[0],color=colors[plot_color_index]) 
  #plot中间的cg
  if len(dg_id)>1:
    for i in range(1,len(dg_id)):
      pre_color = cg_id[i-1]
      cur_color = cg_id[i] 
      x += label_w_list[i-1]*n_cols[i-1]
      if pre_color!=cur_color: 
        plot_rectangle(x=x,y=0,w=ink_seperator_width,h=sheet_size[1],color='black') #ink seperator
        x += 30
        plot_color_index += 1 #换色
      plot_n_rectangle_full_sheet_height(x=x,y=0,w=label_w_list[i],h=label_h_list[i],w_sheet=sheet_size[0],h_sheet=sheet_size[1],n_rec=ups_list[i],color=colors[plot_color_index])  


# def plot_one_color(all_rects):
#     # colors = mcolors.TABLEAU_COLORS.values()
#     colors = list(mcolors.TABLEAU_COLORS.values())
#     print(f'colors = {colors}')
#     plt.figure(figsize=(10,10))

#     unique_rect = []
#     rect_colors = {}
#     for rect in all_rects:
#         b, x, y, w, h, rid = rect
#         rect_name = str(w)+'_'+str(h)
#         if rect_name not in unique_rect:
#           unique_rect.append(rect_name)
#           rect_colors[rect_name] = colors[len(unique_rect)-1]
#           print(f'color dict updated = {rect_colors}')

#         x1, x2, x3, x4, x5 = x, x+w, x+w, x, x
#         y1, y2, y3, y4, y5 = y, y, y+h, y+h, y
#         # color = colors[str(w)+'_'+str(h)]
#         color = rect_colors[rect_name]

#         plt.plot([x1,x2,x3,x4,x5], [y1,y2,y3,y4,y5], color) #draw a rect
    
#     plt.show()



# def plot_solution(all_rects, pal_812, pal_1012):
#     # Plot
#     plt.figure(figsize=(10,10))
#     # Loop all rect
#     for rect in all_rects:
#         b, x, y, w, h, rid = rect
#         x1, x2, x3, x4, x5 = x, x+w, x+w, x, x
#         y1, y2, y3, y4, y5 = y, y, y+h, y+h,y

#         # Pallet type
#         if [w, h] == pal_812:
#             color = '--k'
#         else:
#             color = '--r'

#         plt.plot([x1,x2,x3,x4,x5],[y1,y2,y3,y4,y5], color)
    
#     plt.show()