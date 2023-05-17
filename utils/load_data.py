import pandas as pd

def load_and_clean_data(input_file):
  """
  :param input_file: '../input/HTL_input_0419.csv'
  """
  df = pd.read_csv(input_file)

  #去掉列名中的空格, 并转换为小写
  remove_space_dict = {}
  for col in df.columns:
    remove_space_dict[col] = col.replace(' ', '_').lower()
  df.rename(columns=remove_space_dict, inplace=True)

  #数据格式标准化
  str_cols = ['color_group', 'dimension_group', 'header_variable_data|sku_value', 'item', 'job_number', 'rb']
  int_cols = ['fix_orientation', 'group_sku', 'group_nato', 'sku_quantity', 'sku_seq', 're_qty']
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

  df.rename(columns={'color_group':'cg_id'},inplace=True)
  cols = ['sku_id', 'cg_id', 'dimension_group', 'fix_orientation', 'dg_id', 'cg_dg_id', 'job_number', 'item', 
          'overall_label_width', 'overall_label_length', 'sku_seq', 'rb', 'group_sku', 'group_nato',
          'sku_quantity', 're_qty', 'header_variable_data|sku_value']

  return df[cols]