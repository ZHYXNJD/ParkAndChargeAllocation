# Importing necessary libraries
import gurobipy as gp
from gurobipy import GRB

# Define a simple example for illustration
# Sets of users (I) and slots (J)
users = ["Driver1", "Driver2", "Owner1"]
slots = ["Slot1", "Slot2"]

# Parameters
# Requested parking start and end times for users
parking_times = {
    "Driver1": (8, 10),  # (start, end)
    "Driver2": (9, 12),
    "Owner1": (9, 18)
}

# Buffer times (pre-computed based on phi function or time constraints)
buffers = {
    ("Driver1", "Driver2"): 0,  # Time difference required to avoid conflict
    ("Driver2", "Owner1"): 0,
    ("Driver1", "Owner1"): 0,
    ("Driver2", "Driver1"): 0,
    ("Owner1", "Driver2"): 0,
    ("Owner1", "Driver1"): 0,
}

# Define a big M constant for relaxed constraints
M = 1000

# Create a model
model = gp.Model("Parking Slot Assignment")

# Decision variables
x = model.addVars(users, slots, vtype=GRB.BINARY, name="x")
z = model.addVars([(i, j) for i in users for j in users if i != j], vtype=GRB.BINARY, name="z")

# Objective: Maximize total revenue (simplified)
revenue = {
    "Driver1": 20,  # Revenue from Driver1
    "Driver2": 30,
    "Owner1": 0   # Owners do not generate revenue
}
model.setObjective(gp.quicksum(x[u, s] * revenue.get(u, 0) for u in users for s in slots), GRB.MAXIMIZE)

# Constraints
# Each user can be assigned to at most one slot
for u in users:
    model.addConstr(gp.quicksum(x[u, s] for s in slots) <= 1, name=f"OneSlotPerUser_{u}")

model.addConstr(gp.quicksum(x["Owner1",s] for s in slots) == 1)

# Temporal constraints to avoid conflicts using buffers
for (i, j) in buffers.keys():
    for s in slots:
        model.addConstr(
            x[i, s] + x[j, s] <= 1 + z[i, j] * buffers[i, j] + z[j, i] * buffers[j, i],
            name=f"ConflictAvoidance_{i}_{j}_{s}"
        )

# Solve the model
model.optimize()

# Collect results
results = {
    "Assignments": [],
    "Schedule": [],
    "Revenue": model.objVal
}
if model.status == GRB.OPTIMAL:
    for u in users:
        for s in slots:
            if x[u, s].x > 0.5:  # Binary variable, check if assigned
                results["Assignments"].append((u, s))
    for (i, j) in buffers.keys():
        if z[i, j].x > 0.5:
            results["Schedule"].append((i, "before", j))

print(results)
