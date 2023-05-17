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

def cg_solve(master_batch_length, sub_material_lengths, demands):
    """
    :param master_batch_length: 母料长度，int
    :param sub_material_lengths: 子料长度列表，list    
    :param demands: 子料需求计数列表，list        
    """
    try:
        sub_material_count = len(sub_material_lengths) #子料类型数量
        plan_list = [] #存放结果, 一个元素是一种母料layout类型
        # 声明限制主问题模型和定价子问题模型
        rmp = Model("rmp")
        sub = Model("sub")
        # 取消模型打印信息
        rmp.setParam("OutputFlag", 0)
        sub.setParam("OutputFlag", 0)

        # ------ 主问题：每种母料类型需要生产多少才能满足子料的生产数量需求-----------------------------------------------------
        # 初始化方案 - 使用和子料类型数量一样的母料layout类型数量，一个母料只layout一种子料
        for i in range(sub_material_count):
            plan_list.append(
                [master_batch_length // sub_material_lengths[i] if i == j else 0 for j in range(sub_material_count)])
        # plan_list = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1]]
        # print(plan_list)

        # 初始化RMP
        rmp_var = [] #变量：每一种母料layout的生产数量
        for i in range(len(plan_list)):
            # 1 是目标函数上的系数
            rmp_var.append(rmp.addVar(0, GRB.INFINITY, 1, GRB.CONTINUOUS, f"rmp_{i}"))
        
        rmp_con = [] #限制条件：每一种资料的生产数量必须大于需求
        for i in range(sub_material_count):
            rmp_con.append(
                rmp.addConstr(quicksum(rmp_var[j] * plan_list[j][i] for j in range(len(plan_list))) >= demands[i],
                              f"rmp_con_{i}"))
        # for j in range(len(plan_list)):
        #     rmp_con.append(                              
        #         rmp.addConstr(rmp_var[j] >= 60,
        #                       f"pds_rmp_con_{j}")) ######新增          

        rmp.setAttr("ModelSense", GRB.MINIMIZE)

        # ------ 子问题：在一个母料上layout子料的组合 ---------------------------------------------------------------------
        # 初始化sub
        sub_var = [] #变量：每一种子料在母料上layout的数量
        for i in range(sub_material_count):
            sub_var.append(sub.addVar(0, GRB.INFINITY, 0, GRB.INTEGER, f"sub_{i}"))
        #限制条件：子料的长度之和不能超过母料的长度
        sub.addConstr(
            quicksum(sub_var[i] * sub_material_lengths[i] for i in range(sub_material_count)) <= master_batch_length,
            "sub_con")

        # ------ 求解 ------
        # cg loop
        epoch = 1
        while True:
            print("-" * 20, f"Iteration {epoch}", "-" * 20)
            # 求解受限主问题
            rmp.optimize()
            print(f"RMP Obj: {rmp.ObjVal}")
            # 获取对偶值
            dual_values = rmp.getAttr("Pi", rmp.getConstrs())
            print(f"Dual Values: {dual_values}")
            # 根据对偶值给子问题定价
            sub.setObjective(1 - quicksum(sub_var[i] * dual_values[i] for i in range(sub_material_count)), GRB.MINIMIZE)
            # 求解子问题
            sub.optimize()
            print(f"Sub Obj: {sub.ObjVal}")
            # cg 结束判断
            # if sub.ObjVal > -1e-06:
            if sub.ObjVal > -0.51:              
                print("CG Over!")
                break
            # 加入新列
            new_plan = sub.getAttr("X", sub.getVars())
            plan_list.append([int(round(v)) for v in new_plan])
            print(f"New Col: {plan_list[-1]}")
            rmp_col = Column(new_plan, rmp_con)
            rmp.addVar(0, GRB.INFINITY, 1, GRB.CONTINUOUS, f"cg_{epoch}", rmp_col)
            # 迭代次数自增
            epoch += 1
        # 将RMP转化为MIP
        print("-" * 20, f"Solve MIP", "-" * 20)
        mip_var = rmp.getVars()
        for i in range(len(mip_var)):
            mip_var[i].setAttr("VType", GRB.INTEGER)
        rmp.optimize()
        print(f"MIP Obj: {rmp.ObjVal}")
        c = 1
        for i in range(len(mip_var)):
            if mip_var[i].x > 0.5:
                print(
                    f"Plan-{c}: {plan_list[i]} , len: {quicksum([sub_material_lengths[j] * plan_list[i][j] for j in range(sub_material_count)])} , cnt: {int(round(mip_var[i].x))}")
                c += 1
    except GurobiError as e:
        print(f"Error code {e.errno} : {e}")
    except AttributeError:
        print('Encountered an attribute error')

# COMMAND ----------

# MAGIC %%time
# MAGIC if __name__ == '__main__':
# MAGIC     # 母料长度
# MAGIC     master_batch_length = 678
# MAGIC     # 子料长度
# MAGIC     sub_material_lengths = [117,546,29,616,420,224,224,168,174,39]
# MAGIC     # 子料需求
# MAGIC     demands = [61,91,34,99,115,111,71,69,63,63]
# MAGIC     # 求解
# MAGIC     start_time = time.time()
# MAGIC     cg_solve(master_batch_length, sub_material_lengths, demands)
# MAGIC     print(f"Total Solve Time: {time.time() - start_time} s")

# COMMAND ----------

# %%time
# if __name__ == '__main__':
#     # 母料长度
#     master_batch_length = 678
#     # 子料长度
#     # sub_material_lengths = [663,29,616,420,224,224,168,213]
#     sub_material_lengths = [678,44,631,435,239,239,183,228]    
#     # 子料需求
#     demands = [91,34,99,115,111,71,69,63]
#     # demands = [86,34,91,114,111,67,67,63]    
#     # 求解
#     start_time = time.time()
#     cg_solve(master_batch_length, sub_material_lengths, demands)
#     print(f"Total Solve Time: {time.time() - start_time} s")

# COMMAND ----------


