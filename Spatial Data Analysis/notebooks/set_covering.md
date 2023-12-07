### 空间优化（二）：集合覆盖问题

在此模型中，设施可以为距设施给定覆盖距离 Dc 内的所有需求节点提供服务。 问题在于放置**最少**数量的设施，以确保所有需求节点都能得到服务。 我们假设设施没有容量限制。



```python
pip install -q pulp
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m14.3/14.3 MB[0m [31m81.7 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
from pulp import *
import numpy as np
import geopandas as gp
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
```


```python
#read a sample shapefile
georgia_shp = gp.read_file("https://raw.githubusercontent.com/Ziqi-Li/GEO4162C/main/data/georgia/G_utm.shp")
```


```python
georgia_shp.shape
```




    (172, 18)



创建一个需求和一个设施变量，表示每个需求和设施的索引。  
需求节点：所有县  
facility：我们可以在一些县建造设施  


```python
#create a demand and a facilities variable, indicating the indices of each demand and facility.
#demand node: all counties
#facility: we could build facilities in some counties

demand = np.arange(0,172,1)
facilities = np.arange(0,172,1)
```

计算距离矩阵d_ij(n×n)


```python
#Calculate a distance matrix d_ij (n by n)
coords = list(zip(georgia_shp.centroid.x,georgia_shp.centroid.y))
d = cdist(coords,coords)
```

阈值覆盖距离


```python
# Threshold coverage distance
Dc = 100000 #100km coverage, change this and re run the code.
```

创建一个变量，指示节点 i 是否可以被设施 j 覆盖。


```python
#Creata a variable (alpha in the lecture slide pg.28), indicating  whether a node i can be covered by facility j.
a = np.zeros(d.shape)
a[d <= Dc] = 1
a[d > Dc] = 0
```

声明设施变量 Xj


```python
# declare facilities variables Xj
X = LpVariable.dicts('X_%s',(facilities),cat='Binary')
```

创建一个最小化问题


```python
#Create an minimization problem
prob = LpProblem('Set_Covering', LpMinimize)
```

目标函数：我们要最小化放置设施的数量


```python
# Objective function: we want to minimize the number of placed facilities
prob += sum([X[j] for j in facilities])
```

该约束意味着每个需求节点 i 需要至少由设施服务


```python
# This constraint implies every demand node i needs to be served by at least facility
for i in demand:
    prob += sum(a[i][j]*X[j] for j in facilities) >= 1

```


```python
%%time
# Solve the above problem
prob.solve()

print("Status:", LpStatus[prob.status])
```

    Status: Optimal
    CPU times: user 22.5 ms, sys: 1.05 ms, total: 23.6 ms
    Wall time: 66.4 ms
    


```python
# The minimal number of facilities with the defiened coverage.
print("Objective: ",value(prob.objective))
```

    Objective:  8.0
    


```python
# Print the facility nodes.
rslt = []
for v in prob.variables():
    subV = v.name.split('_')

    if subV[0] == "X" and v.varValue == 1:
        rslt.append(int(subV[1]))
        print('Facility Node: ', subV[1])
```

    Facility Node:  102
    Facility Node:  120
    Facility Node:  145
    Facility Node:  150
    Facility Node:  30
    Facility Node:  38
    Facility Node:  9
    Facility Node:  97
    


```python
# Get the geomerty of the facility nodes.
fac_loc = georgia_shp.iloc[rslt,:]
```


```python
#Plot the faclities (stars) on top of the demand map.
fig, ax = plt.subplots(figsize=(5,5))

georgia_shp.centroid.plot(ax=ax)
fac_loc.centroid.plot(ax=ax,color="red",markersize=300,marker="*")
```




    <Axes: >




    
![png](set_covering_files/set_covering_26_1.png)
    



```python

```
