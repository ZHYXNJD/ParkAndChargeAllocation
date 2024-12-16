from gurobipy import *
import pandas as pd
import numpy as np

np.random.seed(100)

# 需求是50个需求
req_info = pd.read_csv("G:\\2023-纵向\\停车分配\\需求分布\\demand0607\\25-1.csv")
# 泊位供给情况
berths = 2
# 虚拟用户
virtual_req = pd.DataFrame(columns=['arrival_t', 'leave_t', 'activity_t', 'label'])
virtual_req['arrival_t'] = [360] * berths + [1320] * berths
virtual_req['leave_t'] = [360] * berths + [1320] * berths
virtual_req['activity_t'] = [0] * berths * 2
virtual_req['label'] = ['h'] * berths + ['t'] * berths
# 拼接真实用户和虚拟用户
demand_info = pd.concat([req_info[['arrival_t', 'leave_t', 'activity_t', 'label']], virtual_req], axis=0,
                        ignore_index=True)
demand_info = demand_info.sort_values(by='arrival_t').reset_index(drop=True)
arr = demand_info['arrival_t']
lea = demand_info['leave_t']

# 用户集合
u1 = demand_info.loc[demand_info['label'] == 'h'].index.tolist()
u3 = demand_info.loc[demand_info['label'] == 't'].index.tolist()
u2 = list(set(demand_info.index.tolist()) - set(u1) - set(u3))
u0 = u1 + u2 + u3
u23 = u2 + u3

# model
cco = Model("Chance-constraint optimization model")

# decision variable
h_ik = cco.addVars({(i, k) for i in range(berths) for k in u0}, vtype=GRB.BINARY, name='h_ik')
r_jk = cco.addVars({(j, k) for j in u0 for k in u0}, vtype=GRB.BINARY, name='r_jk')
# gurobi无法处理大于二次的高阶项相乘 因此引入额外的变量
# 该变量用于判断j和k是否分配到同一个停车场i
h_ijk = cco.addVars({(i, j, k) for i in range(berths) for j in u0 for k in u0}, vtype=GRB.BINARY, name='h_ijk')


# auxiliary variable
# 蒙特卡洛模拟 计算冲突概率
def conflict(a_k, d_j, delta):
    arrival_li = np.random.normal(loc=a_k, scale=0, size=1000)
    leave_li = np.random.normal(loc=d_j, scale=5, size=1000)
    return np.mean(arrival_li < leave_li - delta)


def conflict_1(a_k, a_j):
    arrival_k = np.random.normal(loc=a_k, scale=0, size=1000)
    arrival_j = np.random.normal(loc=a_j, scale=5, size=1000)
    return np.mean(arrival_k < arrival_j)


# Precompute conflict probability matrices
conflict_matrix = {(j, k): conflict(arr[k], lea[j], delta=5) for j in u0 for k in u0}
conflict_1_matrix = {(j, k): conflict_1(arr[k], arr[j]) for j in u0 for k in u0}

# update
cco.update()
# Objective function with precomputed conflict matrix
obj = quicksum((1 - conflict_matrix[(j, k)]) * r_jk[(j, k)] * h_ijk[(i, j, k)] for j in u0 for k in u2 for i in range(berths))
# obj = quicksum(h_ik[(i, k)] for i in range(berths) for k in u2)
cco.setObjective(obj, GRB.MAXIMIZE)

# constraints
cco.addConstrs(quicksum(h_ik[(i, k)] for i in range(berths)) <= 1 for k in u2)
cco.addConstrs(quicksum(h_ik[(i, k)] for i in range(berths)) == 1 for k in u1)
cco.addConstrs(quicksum(h_ik[(i, k)] for i in range(berths)) == 1 for k in u3)
cco.addConstrs(quicksum(h_ik[(i, k)] for k in u1) == 1 for i in range(berths))
cco.addConstrs(quicksum(h_ik[(i, k)] for k in u3) == 1 for i in range(berths))

cco.addConstrs(h_ijk[(i, j, k)] <= h_ik[(i, j)] for i in range(berths) for j in u0 for k in u0)
cco.addConstrs(h_ijk[(i, j, k)] <= h_ik[(i, k)] for i in range(berths) for j in u0 for k in u0)
cco.addConstrs(h_ijk[(i, j, k)] >= h_ik[(i, j)] + h_ik[(i, k)] - 1 for i in range(berths) for j in u0 for k in u0)

cco.addConstrs(r_jk[(j, k)] <= h_ijk[(i, j, k)] for j in u0 for k in u0 for i in range(berths))
cco.addConstrs(conflict_matrix[(j, k)] * r_jk[(j, k)] <= 0.1 for j in u0 for k in u0)
cco.addConstrs(conflict_1_matrix[(j, k)] * r_jk[(j, k)] <= 0.01 for j in u0 for k in u0)

cco.addConstrs(quicksum(r_jk[(j, k)] for j in u0) == 0 for k in u1)
cco.addConstrs(quicksum(r_jk[(j, k)] for k in u0) == 1 for j in u1)
cco.addConstrs(quicksum(r_jk[(j, k)] for j in u0) == 1 for k in u3)
cco.addConstrs(quicksum(r_jk[(j, k)] for k in u0) == 0 for j in u3)

cco.addConstrs(quicksum(r_jk[(j, k)] for j in u0) == quicksum(r_jk[(k, j)] for j in u0) for k in u2)
cco.addConstrs(quicksum(r_jk[(j, k)] for k in u0) <= 1 for j in u2)
cco.addConstrs(r_jk[(j, j)] == 0 for j in u0)

cco.optimize()
cco.computeIIS()
cco.write("model1.ilp")

# Output results
if cco.status == GRB.OPTIMAL:
    for i in range(berths):
        for k in u0:
            if h_ik[(i, k)].X > 0.5:  # Check assignment
                print(f"User {k} assigned to berth {i} with arrival at {arr[k]} and departure at {lea[k]}")
    print(f"Optimal Objective Value: {cco.objVal}")
else:
    print("No optimal solution found.")

# cco.computeIIS()
# cco.write("model1.ilp")

# Retrieve decision variable values
# Assuming model optimization was successful
allocation_results = pd.DataFrame([(i, k, h_ik[(i, k)].X) for i in range(berths) for k in u0],
                                  columns=['berth', 'user', 'allocated'])

# Filter out allocations where h_ik = 1 (indicating successful allocations)
allocated_users = allocation_results[allocation_results['allocated'] == 1]

# Check allocations by berth and by user
allocated_by_berth = allocated_users.groupby('berth').size()
allocated_by_user = allocated_users.groupby('user').size()

print("Allocated Users by Berth:")
print(allocated_by_berth)
print("\nAllocated Users by User:")
print(allocated_by_user)

# Check conflict probabilities
# Verifying that conflict probabilities for accepted allocations are within threshold
conflicts = []
for j, k in conflict_matrix.keys():
    if r_jk[(j, k)].X > 0.5:  # Check if this pair was assigned
        conflict_prob = conflict_matrix[(j, k)]
        if conflict_prob > 0.1:
            conflicts.append((j, k, conflict_prob))

print("\nConflicting Assignments (conflict probability > 0.1):")
for conflict in conflicts:
    print(f"User pair ({conflict[0]}, {conflict[1]}) - Conflict Probability: {conflict[2]}")

# Overall Utilization and Failure Rate Calculation
successful_allocations = sum(1 for k in u2 if any(h_ik[(i, k)].X > 0.5 for i in range(berths)))
total_requests = len(u2)
failure_rate = 1 - (successful_allocations / total_requests)

print(f"\nTotal Requests: {total_requests}")
print(f"Successful Allocations: {successful_allocations}")
print(f"Failure Rate: {failure_rate:.2%}")

# Checking the objective function value
objective_value = cco.ObjVal
print(f"\nObjective Value (Parking Probability Sum): {objective_value}")
total_acc = quicksum(h_ik[(i, k)] for k in u2 for i in range(berths)).getValue()
print(f"\nTotal acc: {total_acc}")
