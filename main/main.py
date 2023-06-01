# Databricks notebook source
from datetime import datetime
import yaml
from utils.load_data import load_config, write_config

# COMMAND ----------

start_time = datetime.now()
print(start_time)

# COMMAND ----------

# MAGIC %md
# MAGIC #### user inputs

# COMMAND ----------

dbutils.widgets.dropdown("Request Type", '1_Batching_Proposal', ['1_Batching_Proposal', '2_Direct-Layout'])
request_type = dbutils.widgets.get("Request Type")

dbutils.widgets.dropdown("Batching Type", '1_OCOD', ['1_OCOD', '2_OCMD', '3_MCMD_Seperater', '4_MCMD_No_Seperater'])
batching_type = dbutils.widgets.get("Batching Type")

dbutils.widgets.text("ABC Plates", "1", "ABC Plates")
n_abc = dbutils.widgets.get("ABC Plates")

dbutils.widgets.text("Films", "678x528, 582x482, 522x328", "Films")
sheet_size_list = dbutils.widgets.get("Films")

dbutils.widgets.text("Colors", '1', "Colors")
n_color = dbutils.widgets.get("Colors")

# COMMAND ----------

n_abc = int(n_abc)

sheet_size_list = sheet_size_list.split(', ') #678x528, 582x482, 522x328
sheet_size_list = [sorted([int(i.split('x')[0]), int(i.split('x')[1])],reverse=True) for i in sheet_size_list]

n_color = int(n_color)
if batching_type!='4_MCMD_No_Seperater':
  add_pds_per_sheet = int((n_color+1)*3.5)
else:
  add_pds_per_sheet = int((n_color+2)*3.5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### config

# COMMAND ----------

config_filename = 'config_panyu_htl.yaml'
config_file = f"../config/{config_filename}"
params_dict = load_config(config_file)[batching_type]

params_dict.update({'user_params':{
  'add_pds_per_sheet':add_pds_per_sheet,
  'batching_type':batching_type,
  'n_abc':n_abc,
  'n_color':n_color,  
  'request_type':request_type,
  'sheet_size_list':sheet_size_list,
  }})

for k,v in params_dict.items():
  print(f'{k}: {v}')

# COMMAND ----------

write_config('/tmp/'+config_filename, params_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main

# COMMAND ----------

if batching_type=='1_OCOD':
  pass
elif batching_type=='2_OCMD':
  pass
elif batching_type=='3_MCMD_Seperater':  
  notebook_path = './3_mcmd_seperator_dg_pds'
  dbutils.notebook.run(notebook_path, 36000, {'config_filename':config_filename})
elif batching_type=='4_MCMD_No_Seperater':
  pass

# COMMAND ----------

end_time = datetime.now()
print(start_time)
print(end_time)
print('running time =', (end_time-start_time).seconds, 'seconds')
