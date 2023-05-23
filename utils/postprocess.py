import numpy as np


def prepare_dg_level_results(df_res, df_1_2, df_1_5, pass_cg_dg_ids_1_2):
  """
  df_res: 初始化的df_res
  df_1_2: dg level results
  df_1_5: sku level results
  pass_cg_dg_ids_1_2: qualified cg_dg_ids in df_1_2
  """
  #DG level results - 更新结果和PPC比较的结果df_res，同时也是Files_to_ESKO的file-1
  df_1_5_agg = df_1_5.groupby(['cg_id'])['sku_pds'].agg('max').reset_index()
  max_sku_pds_dict_1_5 = dict(zip(df_1_5_agg['cg_id'],df_1_5_agg['sku_pds']))
  print(f'max_sku_pds_dict_1_5 = {max_sku_pds_dict_1_5}')

  for cg_dg_id in pass_cg_dg_ids_1_2:
    #batch_id
    df_temp = df_1_2[df_1_2['cg_dg_id']==cg_dg_id]
    cg_id = df_temp['cg_id'].values[0]
    batch_id = 'ocod'+'<+>'+cg_id
    df_res.loc[df_res['Color Group']==cg_id,'batch_id'] = batch_id
    #orient
    df_res_temp = df_res[df_res['Color Group']==cg_id]
    label_width = df_temp['overall_label_width'].values[0]
    label_length = df_temp['overall_label_length'].values[0]
    hor_label_width = df_res_temp['OverLabel W'].values[0]
    hor_label_length = df_res_temp['OverLabel L'].values[0]  
    print([label_width,label_length],[hor_label_width,hor_label_length])
    if (label_width==hor_label_width) & (label_length==hor_label_length):
      df_res.loc[df_res['Color Group']==cg_id,'orient'] = 'horizon'
    elif (label_width==hor_label_length) & (label_length==hor_label_width):
      df_res.loc[df_res['Color Group']==cg_id,'orient'] = 'vertical'
    #pds
    df_res.loc[df_res['Color Group']==cg_id,'pds'] = int(max_sku_pds_dict_1_5[cg_id])
    #ups,the_pds 
    layout_dict = eval(df_temp['cg_dg_layout'].values[0])
    ups = int(layout_dict['ups'])
    qty_sum = df_res_temp['qty_sum'].values[0]
    df_res.loc[df_res['Color Group']==cg_id,'ups'] = ups
    df_res.loc[df_res['Color Group']==cg_id,'the_pds'] = np.ceil(qty_sum/ups)
    #Printing_area
    sheet_width = df_temp['sheet_width'].values[0]
    sheet_length = df_temp['sheet_length'].values[0] 
    df_res.loc[df_res['Color Group']==cg_id,'Printing_area'] = str(int(sheet_width))+'x'+str(int(sheet_length))
    #中离数
    df_res.loc[df_res['Color Group']==cg_id,'中离数'] = 0
    return df_res


def prepare_sku_level_results(res_file_3, df_1_5):
  """
  res_file_3: 初始化的res_file_3
  df_1_5: sku level results
  """
  #sku level results - 更新结果文件Files_to_ESKO的file-3
  df_temp = df_1_5.copy()
  df_temp['Job Number'] = df_1_5['sku_id'].apply(lambda x: x.split('<+>')[0])
  df_temp['Sku Seq'] = df_1_5['sku_id'].apply(lambda x: x.split('<+>')[1])
  df_temp['Total Prod'] = df_temp['sku_ups']*df_temp['sku_pds']
  df_temp['Over Prod Qty'] = df_temp['Total Prod'] - (df_temp['re_qty']-10) #这里是否需要减10？
  df_temp['Over Prod Per'] = round(df_temp['Over Prod Qty']/df_temp['Total Prod'],2)
  # display(df_temp)
  # display(res_file_3)
  for df_i in [df_temp, res_file_3]:
    df_i['Sku Seq'] = df_i['Sku Seq'].astype('int')
  res_file_3 = res_file_3.merge(df_temp, how='left', on=['Job Number','Sku Seq'])
  # display(res_file_3)
  sets = ['Set A Ups','Set B Ups','Set C Ups','Set D Ups','Set E Ups','Set F Ups','Set G Ups','Set H Ups']
  set_cols = [c for c in res_file_3.columns if c in sets]
  cols = ['Job Number', 'Sku Value', 'Order Qty', 'Sku Seq', 'Item', 'Total Prod','Over Prod Qty','Over Prod Per']+set_cols
  res_file_3 = res_file_3[cols].sort_values(['Job Number','Sku Seq'])
  # res_file_3.rename(columns={'sku_ups':'Set A Ups'}, inplace=True
  return res_file_3