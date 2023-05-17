# Databricks notebook source
from gurobipy import *
import numpy as np

# COMMAND ----------

# Gurobi cloud license
lic = '''
CLOUDACCESSID=c3409978-274b-4364-a7a9-ec17bfbff229
CLOUDKEY=MzJmMmQ5MTEtYmQ1Mi00ZG
LICENSEID=948898
CLOUDPOOL=948898-8Cores
'''

with open("/tmp/gurobi.lic", "wb") as file:
    file.write(lic.encode('ascii'))

# COMMAND ----------

m=4
s=[1,3,6,7]
B=8
a=[]
a1=[0,0,1,0]
a2=[0,0,0,1]
a3=[0,1,0,0]
a4=[1,0,0,0]
a.append(a1)
a.append(a2)
a.append(a3)
a.append(a4)
n=len(a)

# COMMAND ----------

master=Model("master")
x={}
for j in range(n):
    x[j]=master.addVar(vtype="B",name="x(%s)"%j)
 
orders={}
for i in range(m):
    orders[i]=master.addConstr(
        quicksum(a[j][i]*x[j] for j in range(n))>=1,"order(%s)"%i
    )
master.setObjective(quicksum(x[j] for j in range(n)), GRB.MINIMIZE)
master.update()
 
iter = 0
dual=[]
EPS = 1.e-6
while True:
    iter += 1
    print("*****relaxation for master problem*****")
    relax=master.relax()
    relax.optimize()
    pi=[c.pi for c in relax.getConstrs()]
    sub=Model("sub")
    y={}
    for i in range(m):
        y[i]=sub.addVar(vtype="B")
    sub.addConstr(quicksum(y[i]*s[i] for i in range(m))<=B)
    sub.setObjective(quicksum(pi[i]*y[i] for i in range(m)),GRB.MAXIMIZE)
    print("***** sub problem*****")
    sub.optimize()
    for i in range(m):
        print(y[i].x)
    if sub.ObjVal <1+EPS:
        break
    var=[]  
    for i in range(m):
        var.append(y[i].x)
    a.append(var)
    print(a)
    col=Column()
    for i in range(m):
        col.addTerms(a[n][i],orders[i])
    x[n]=master.addVar(obj=1,vtype="B",name="x(%s)"%n,column=col)
    master.update()
    master.write("MP" + str(iter) + ".lp")
    n+=1
    dual.append(pi)
print("*****integer problem*****")
master.update()
master.optimize()
print(a)
for j in range(n):
    print(x[j].x)

# COMMAND ----------

import time
from gurobipy import *


def cg_solve(master_batch_length, sub_material_lengths, demands):
    try:
        sub_material_count = len(sub_material_lengths)
        plan_list = []
        # 声明限制主问题模型和定价子问题模型
        rmp = Model("rmp")
        sub = Model("sub")
        # 取消模型打印信息
        rmp.setParam("OutputFlag", 0)
        sub.setParam("OutputFlag", 0)
        # 初始化方案
        for i in range(sub_material_count):
            plan_list.append(
                [master_batch_length // sub_material_lengths[i] if i == j else 0 for j in range(sub_material_count)])
        # 初始化RMP
        rmp_var = []
        for i in range(len(plan_list)):
            # 1 是目标函数上的系数
            rmp_var.append(rmp.addVar(0, GRB.INFINITY, 1, GRB.CONTINUOUS, f"rmp_{i}"))
        rmp_con = []
        for i in range(sub_material_count):
            rmp_con.append(
                rmp.addConstr(quicksum(rmp_var[j] * plan_list[j][i] for j in range(len(plan_list))) >= demands[i],
                              f"rmp_con_{i}"))
        rmp.setAttr("ModelSense", GRB.MINIMIZE)
        # 初始化sub
        sub_var = []
        for i in range(sub_material_count):
            sub_var.append(sub.addVar(0, GRB.INFINITY, 0, GRB.INTEGER, f"sub_{i}"))
        sub.addConstr(
            quicksum(sub_var[i] * sub_material_lengths[i] for i in range(sub_material_count)) <= master_batch_length,
            "sub_con")
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
            if sub.ObjVal > -1e-06:
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


if __name__ == '__main__':
    # 母料长度
    master_batch_length = 115
    # 子料长度
    sub_material_lengths = [25, 40, 50, 55, 70]
    # 子料需求
    demands = [50, 36, 24, 8, 30]
    # 求解
    start_time = time.time()
    cg_solve(master_batch_length, sub_material_lengths, demands)
    print(f"Total Solve Time: {time.time() - start_time} s")

# COMMAND ----------


