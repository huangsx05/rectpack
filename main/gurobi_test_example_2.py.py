# Databricks notebook source
# Gurobi cloud license

#for panyu_stage
# lic = '''
# CLOUDACCESSID=c3409978-274b-4364-a7a9-ec17bfbff229
# CLOUDKEY=MzJmMmQ5MTEtYmQ1Mi00ZG
# LICENSEID=948898
# CLOUDPOOL=948898-8Cores
# '''

#for htl_batching
lic = '''
CLOUDACCESSID=b25801de-4c67-4efb-a42b-7f566332e924
CLOUDKEY=NWFhMDNmYjgtYzQ1Ny00N2
LICENSEID=948898
CLOUDPOOL=948898-Test
'''

with open("/tmp/gurobi.lic", "wb") as file:
    file.write(lic.encode('ascii'))

# COMMAND ----------

import time
from gurobipy import *

# COMMAND ----------

import gurobipy as grb
from gurobipy import GRB
import numpy as np
 
# column就是pattern
def master_problem(column, vtype, demand_number_array):
    m = grb.Model()
    x = m.addMVar(shape=column.shape[1], lb=0, vtype=vtype)
    m.addConstr(lhs=column @ x >= demand_number_array)
    m.setObjective(x.sum(), GRB.MINIMIZE)
    m.optimize()
 
    if vtype == GRB.CONTINUOUS:
        return np.array(m.getAttr('Pi', m.getConstrs()))
    else:
        return m.objVal, np.array(m.getAttr('X'))
 
# 将主问题松弛成线性问题
def restricted_lp_master_problem(column,demand_number_array):
    return master_problem(column, GRB.CONTINUOUS,demand_number_array)
 
# 找到所有的cut pattern后，求解整数优化主问题。
def restricted_ip_master_problem(column,demand_number_array):
    return master_problem(column, GRB.INTEGER,demand_number_array)
 
# 子問題求解，寻找pattern
def knapsack_subproblem(kk,demand_width_array,demand_number_array,roll_width):
    m = grb.Model()
    x = m.addMVar(shape=kk.shape[0], lb=0, vtype=GRB.INTEGER)  # 整数规划
    m.addConstr(lhs=demand_width_array @ x <= roll_width)
    m.setObjective(1 - kk @ x, GRB.MINIMIZE)
    m.optimize()
 
    flag_new_column = m.objVal < 0   # 判别数
    if flag_new_column:
        new_column = m.getAttr('X')  # 子问题求解，找到新的列
    else:
        new_column = None
    return flag_new_column, new_column
 
def test(roll_width,demand_width_array,demand_number_array):
    # 初始定义最简单的pattern
    initial_cut_pattern = np.diag(np.floor(roll_width / demand_width_array))
    flag_new_cut_pattern = True
    new_cut_pattern = None
    cut_pattern = initial_cut_pattern  # 最近的pattern
    while flag_new_cut_pattern:
        if new_cut_pattern:
            cut_pattern = np.column_stack((cut_pattern, new_cut_pattern))    # 最新的pattern。按列合并
        kk = restricted_lp_master_problem(cut_pattern, demand_number_array)  # 将主问题松弛成线性问题
        flag_new_cut_pattern, new_cut_pattern = knapsack_subproblem(kk, demand_width_array, demand_number_array, roll_width)
    # 找到所有的cut pattern后，求解整数优化主问题。
    minimal_stock, optimal_number = restricted_ip_master_problem(cut_pattern, demand_number_array)
    print(f'demand_width_array: {demand_width_array}')
    print(f'demand_number_array: {demand_number_array}')
    print('result:')
    print(f'minimal_stock: {minimal_stock}')
 
# 生成测试数据
def generate_data(roll_width,customer_num):
    roll_width = np.array(roll_width)
    demand_width_array = np.random.randint(1, 3, size=(customer_num))
    demand_number_array = np.random.randint(50, 200, size=(customer_num))
    return roll_width, demand_width_array, demand_number_array
 
# print(generate_data(120,100))
roll_width, demand_width_array, demand_number_array = generate_data(roll_width=120,customer_num=2)
test(roll_width, demand_width_array, demand_number_array)

# COMMAND ----------


