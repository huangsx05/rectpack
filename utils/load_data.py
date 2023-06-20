import pandas as pd
import json


def load_user_params(input_params):
  user_params = {}
  user_params['request_type'] = input_params['requestType']
  user_params["batching_type"] = input_params['batchingType']
  user_params["add_pds_per_sheet"] = input_params['setUpPerBatch']
  user_params["n_abc"] = input_params['setOfPlates']
  user_params["min_single_col_width"] = input_params['minWidthandLength']
  user_params["ink_seperator_width"] = input_params['separatorWidth']
  user_params["internal_days"] = input_params['internalDays']

  #考虑：如果可以直接判断某个sheet_size没有用，可以在这里排除
  sheet_input_list = input_params["filmSize"].split('/')
  sheet_input_list = [i.split(',') for i in sheet_input_list]  
  sheet_size_list = [sorted([int(i[0]), int(i[1])],reverse=True) for i in sheet_input_list] #严格按照纸张从大到小排序
  user_params['sheet_size_list'] = sheet_size_list

  user_params['sheets'] = {}
  batching_type = user_params["batching_type"]
  for i in sheet_input_list:
    sheet_name = str(i[0])+'<+>'+str(i[1])
    sheet_weight = i[4]    
    if batching_type=='3_MCMD_Seperater':
      n_color_limit = i[2]
    elif batching_type=='4_MCMD_No_Seperater':
      n_color_limit = i[3]      
    user_params['sheets'][sheet_name] = {'n_color_limit':n_color_limit, 'weight':sheet_weight}

  return user_params


def load_config(config_file, params_dict={}):
  with open(config_file, "r", encoding="utf-8") as fh:
      config = json.load(fh)
  params_dict.update(config)
  return params_dict


# 写入YAML文件的方法
def write_config(config_file, params_dict):
    with open(config_file, encoding="utf-8", mode="w") as f:
        yaml.dump(params_dict, stream=f, allow_unicode=True)


def load_and_clean_data(df, input_file=None):
  """
  :param input_file: '../input/HTL_input_0419.csv'
  """
  if len(df)==0:
    df = pd.read_csv(input_file)

  cols = ['sku_id', 'color_group', 'dimension_group', 'fix_orientation', 'dg_id', 'cg_dg_id', 'job_number', 'item', 
          'overall_label_width', 'overall_label_length', 'sku_seq', 'rb', 'group_sku', 'group_nato',
          'sku_quantity', 're_qty', 'header_variable_data|sku_value', 'wds']
  
  #删除多于列
  drop_cols = []
  for col in df.columns:
    if col.lower() not in cols:
      drop_cols.append(col)
  df.drop(columns=drop_cols, inplace=True)

  #删除全部为空的行
  df = df.dropna(how='all')    

  #去掉列名中的空格, 并转换为小写
  remove_space_dict = {}
  for col in df.columns:
    remove_space_dict[col] = col.replace(' ', '_').lower()
  df.rename(columns=remove_space_dict, inplace=True)

  #临时性代码
  if 're_qty' not in df.columns:
    print(f're_qty is missing, use re_qty=sku_qty+10')
    df['re_qty'] = df['sku_quantity']+10
  if 'wds' not in df.columns:
    print(f'wds is missing, use wds=1')
    df['wds'] = 1    

  #数据格式标准化
  str_cols = ['color_group', 'dimension_group', 'header_variable_data|sku_value', 'item', 'job_number', 'rb']
  int_cols = ['fix_orientation', 'group_sku', 'group_nato', 'sku_quantity', 'sku_seq', 're_qty', 'wds']
  double_cols = ['overall_label_length', 'overall_label_width']
  for c in str_cols:
    df[c] = df[c].astype('str')
    df[c] = df[c].str.replace(' ', '_')
    df[c] = df[c].str.lower() 
  for c in int_cols:
    df[c] = df[c].astype('int')    
  for c in double_cols:
    df[c] = df[c].astype('double')  
    df[c] = round(df[c], 1)

  #增加分组id的列
  df = df.sort_values(['color_group','dimension_group','item','sku_seq']) 
  if 'index' in df.columns:
    df.drop(columns=['index'], inplace=True)
  df['dg_id'] = df['dimension_group']+'<+>rotate_'+df['fix_orientation'].astype('str')
  df['cg_dg_id'] = df['color_group']+'<+>'+df['dg_id']
  df['sku_id'] = df['job_number']+'<+>'+df['sku_seq'].astype('str') 

  #rename cols
  df = df[cols]
  df.rename(columns={'color_group':'cg_id'},inplace=True)

  return df


def agg_to_get_dg_level_df(sku_level_df):
  cols_to_first = ['cg_id', 'dimension_group', 'fix_orientation','overall_label_width', 'overall_label_length', 'wds']
  agg_dict = {'re_qty':'sum', 'wds':'min'}
  for c in cols_to_first:
    agg_dict[c] = 'first'
  dg_level_df = sku_level_df.groupby(['cg_dg_id']).agg(agg_dict).reset_index()
  return dg_level_df


def initialize_input_data(input_file, filter_Color_Group):
  """
  :return:
    df_raw: input data before data cleaning;
    df: input data after data cleaning;    
    df_1: aggregated input data at dg level.      
  """
  df_raw = pd.read_csv(input_file)
  if len(filter_Color_Group)>0:
    df_raw = df_raw[df_raw['Color_Group'].isin(filter_Color_Group)]  
  df = load_and_clean_data(df_raw)
  df_1 = agg_to_get_dg_level_df(df)
  return df_raw, df, df_1


def initialize_dg_level_results(df):
  #初始化和PPC结果比较的results data - 同时也是Files_to_ESKO的file-1
  res = df.copy()[['sku_id','cg_id','dimension_group','overall_label_width','overall_label_length','sku_quantity','re_qty']]
  res.rename(columns={'dimension_group':'DG',
                        'sku_quantity':'Quantity',
                        're_qty':'Req_Qty',
                        'cg_id':'Color Group',
                        'overall_label_width':'OverLabel W',
                        'overall_label_length':'OverLabel L'},
                        inplace=True)

  #aggregate by cg,dg
  agg_dict = {'Quantity':'sum', 'Req_Qty':'sum',}
  for c in [col for col in res.columns if col not in ['Color Group','DG']]:
    if c not in agg_dict.keys():
      agg_dict[c] = 'first' 
  df_res = res.groupby(['Color Group','DG']).agg(agg_dict).reset_index()

  #初始化所有列
  cols = ['batch_id','DG','orient','Printing_area','Quantity','Req_Qty','pds','ups','qty_sum','the_pds','overproduction','blank','Color Group','OverLabel W','OverLabel L','中离数']
  for c in cols:
    if c not in df_res.columns:
      df_res[c] = None
  df_res['qty_sum'] = df_res['Req_Qty']
  df_res = df_res[cols].sort_values(['Color Group','DG'])
  return df_res


def initialize_sku_level_results(df):
  #初始化结果文件Files_to_ESKO的file-3
  res_file_3 = df.copy()[['job_number','header_variable_data|sku_value','sku_quantity','sku_seq','item']]
  res_file_3.rename(columns={'job_number':'Job Number',
                            'header_variable_data|sku_value':'Sku Value',
                            'sku_quantity':'Order Qty',
                            'sku_seq':'Sku Seq',
                            'item':'Item',
                            },
                            inplace=True)
  res_file_3 = res_file_3.sort_values(['Job Number','Sku Seq'])
  return res_file_3