import json
from utils.load_data import initialize_input_data, load_config, write_config

def main():
  #inputs for main
  user_params_path = "../config/user_params.json"
  input_file = '../input/HTL_input_0519.csv' #'../input/HTL_input_0419.csv','../input/HTL_input_0519.csv',
  filter_Color_Group = [] #空代表不筛选，全部计算
  #filter_Color_Group = ['CG_22', 'CG_23', 'CG_24', 'CG_26', 'CG_27', 'CG_28', 'CG_29', 'CG_30'],
  config_path = f"../config/config.json"

  #get user inputs
  with open(user_params_path, "r", encoding="utf-8") as f:
    user_params = json.load(f)
  batching_type = user_params["batching_type"]
  n_abc = user_params["n_abc"]  
  n_abc = int(n_abc)  
  n_color = user_params["n_color"]
  n_color = int(n_color)
  request_type = user_params["request_type"]  
  sheet_size_list = user_params["sheet_size_list"]
  sheet_size_list = [sorted([int(i[0]), int(i[1])],reverse=True) for i in sheet_size_list] #严格按照纸张从大到小排序
  if batching_type!='4_MCMD_No_Seperater':
    add_pds_per_sheet = int((n_color+1)*3.5)
  else:
    add_pds_per_sheet = int((n_color+2)*3.5)

  #get and update configs
  params_dict = load_config(config_path)[batching_type]
  params_dict.update({'user_params':{'add_pds_per_sheet':add_pds_per_sheet,
                                     'batching_type':batching_type,
                                     'n_abc':n_abc,
                                     'n_color':n_color, 
                                     'request_type':request_type,
                                     'sheet_size_list':sheet_size_list,
                                     }})
  print(params_dict)

  #jobs input
  df_raw, df, df_1 = initialize_input_data(input_file, filter_Color_Group) #------ 数据清洗部分可以转移到GPM完成

  #main
  if batching_type=='1_OCOD':
    pass
  elif batching_type=='2_OCMD':
    pass
  elif batching_type=='3_MCMD_Seperater':  
    from sub_main.runner_3_mcmd_seperater import runner_3_mcmd_seperator_sku_pds
    runner_3_mcmd_seperator_sku_pds(params_dict, df, df_1)
  elif batching_type=='4_MCMD_No_Seperater':
    pass
  print("done")

if __name__ == "__main__":
  main()