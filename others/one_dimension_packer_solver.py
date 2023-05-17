#reference: https://www.freesion.com/article/9911957387/

# def binningNextFit(min_width_dict, sheet_width):
#   """
# 	1） 下项适配(next fit)
# 	当处理一个物品的时候， 检查当前箱子容量是否足够；如果足够， 就将物品放入当前箱子， 如果不足， 就重新开辟一个新的箱子；
#   :param min_width_dict: min_width_dict[label_size] = {'min_width':min_width, 'is_rotated':is_rotated}
# 	:return: 
# 		min_width_dict: min_width_dict[label_size] = {'min_width':min_width, 'is_rotated':is_rotated, 'sheet_index':index}
#   """
#   N = len(min_width_dict)
#   sheet_width_list = [sheet_width]*N #最多需要N个箱子
# 	index = 0
# 	for i in range(N):
#     width = min_width_dict[min_width_dict.keys()[i]]['min_width']
#       if width <= sheet_width_list[index]:
#         sheet_width_list[index] -= width
#         min_width_dict[min_width_dict.keys()[i]]['sheet_index'] = index + 1
#       else: #开一个新箱子
#         index += 1
#         sheet_width_list[index] -= width
#         min_width_dict[min_width_dict.keys()[i]]['sheet_index'] = index + 1
#   return min_width_dict


# def binningFirstFit(min_width_dict, sheet_width):
# 	"""
# 	2） 首次适配(first fit)
# 	每次考虑将一个物品放入箱子中的时候，都从第一个箱子开始尝试放入物品；当所有的已打开的箱子的容量都不满足的时候， 重新开辟一个新的箱子；
# 	:param min_width_dict: min_width_dict[label_size] = {'min_width':min_width, 'is_rotated':is_rotated}
# 	"""
# 	N = len(min_width_dict)
#   sheet_width_list = [sheet_width]*N #最多需要N个箱子	
# 	index = 0
# 	for i in range(N):
# 		width = min_width_dict[min_width_dict.keys()[i]]['min_width']
# 		for j in range(N):
# 			if (sheet_width_list[j] >= width) {
# 				sheet_width_list[j] -= width
# 				min_width_dict[min_width_dict.keys()[i]]['sheet_index'] = index + 1
# 				break
#   return min_width_dict

# def binningBestFit(min_width_dict, sheet_width):
# 	"""
# 	3） 最佳适配(best fit)
# 	每次考虑将一个物品放入箱子的时候， 都考虑所有能够容纳物品的箱子中， 选取剩余容量最小的那个箱子；
# 	"""
# 	N = len(min_width_dict)
#   sheet_width_list = [sheet_width]*N #最多需要N个箱子	
# 	for i in range(N):
# 		min_index = 1 #剩余容量最小的箱子
# 		index = 0
# 		width = min_width_dict[min_width_dict.keys()[i]]['min_width']
# 		for j in range(N):
# 			tmp = sheet_width_list[j] - width #剩余容量
# 			if (tmp >= 0) & (tmp < min_index):
# 				min_index = tmp
# 				index = j
# 		sheet_width_list[index] -= width
# 		min_width_dict[min_width_dict.keys()[i]]['sheet_index'] = index + 1
#   return min_width_dict		


def binningBestFit_with_inkSeperator(min_width_dict, sheet_width, ink_seperator):
	"""
	3） 最佳适配(best fit)，并考虑多个颜色有中离，减小sheet的有效宽度
	每次考虑将一个物品放入箱子的时候， 都考虑所有能够容纳物品的箱子中， 选取剩余容量最小的那个箱子；
  :param min_width_dict: min_width_dict[cg_id] = min_width
	:return: 
		sheet_index_dict: min_width_dict[cg_id] = index
	"""
	N = len(min_width_dict)
	sheet_width_list = [sheet_width]*N #最多需要N个箱子	
	sheet_index_dict = {}
	for i in range(N):
		min_width_margin = sheet_width #剩余最小容量
		index = 0
		width = min_width_dict[list(min_width_dict.keys())[i]]
		for j in range(N):
			if sheet_width_list[j]==sheet_width:
				tmp = sheet_width_list[j] - width #剩余容量
			else:
				tmp = sheet_width_list[j] - width - ink_seperator #考虑中离
			if (tmp >= 0) & (tmp < min_width_margin):
				min_width_margin = tmp
				index = j
		sheet_width_list[index] = min_width_margin
		sheet_index_dict[list(min_width_dict.keys())[i]] = index + 1
	return sheet_index_dict		