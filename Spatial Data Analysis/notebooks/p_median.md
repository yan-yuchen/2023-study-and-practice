### 空间优化（一）：p-中值问题

这个问题需要p设施的位置，同时最小化服务所有需求的总加权距离。
每个节点都有一个关联的权重，表示该节点的需求量。


**目标函数：** 最小化所有设施和需求节点的需求加权总和。

**决策变量：** 将设施放置在何处以及哪个设施位置为哪些需求节点提供服务

**限制：**
- 每个节点由 1 个设施提供服务
- 仅当某个位置存在设施时，节点才可以由该设施提供服务。
- 我们必须放置p设施
- 每个节点要么是一个设施，要么不是。


```python
pip install -q pulp
```


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
facility：设施将建在一些选定的县之上


```python
#create a demand and a facilities variable, indicating the indices of each demand and facility.
#demand node: all counties
#facility: Facilities will be built on top of some chosen counties

demand = np.arange(0,172,1)
facilities = np.arange(0,172,1)
```

计算距离矩阵d_ij(n×n)


```python
#Calculate a distance matrix d_ij (n by n)
coords = list(zip(georgia_shp.centroid.x,georgia_shp.centroid.y))

d = cdist(coords,coords)
```

每个县(hi)的需求是总人口


```python
#the demand for each county (h_i) is the total populatoion
h = georgia_shp.TotPop90.values
```

声明设施变量；生成的变量名称为：X_1,X_2,...


```python
# declare facilities variables;the resulting variable names are: X_1,X_2,...
X = LpVariable.dicts('X_%s',(facilities),cat='Binary')

# declare demand-facility pair variables; the resulting variable names are Y_0_1, Y_0_2,...
Y = LpVariable.dicts('Y_%s_%s', (demand,facilities),cat='Binary')
```

要放置的设施数量


```python
#Number of facilities to place
p = 3 #change this and re-run the code.

#Create a new problem
prob = LpProblem('P_Median', LpMinimize)
```

目标函数：最小化所有设施和需求节点的加权需求距离总和  
(h_i: i 处的需求；d_ij: i 和 j 之间的距离)  
 “for”循环用于迭代序列  


```python
# Objective function: Minimizing weighted demand-distance  summed over all facilities and demand nodes
# (h_i: demand at i; d_ij: distance between i and j)
# A "for" loop is used for iterating over a sequence

prob += sum(sum(h[i] * d[i][j] * Y[i][j] for j in facilities) for i in demand)
```

这个约束表明我们必须精确放置 p 个设施


```python
# This constraint indicates we must place exactly p facilities

prob += sum([X[j] for j in facilities]) == p
```

这一约束意味着需求节点 i 只能由一个设施提供服务


```python
# This constraint implies that a demand node i can only be serviced by one facility

for i in demand:
    prob += sum(Y[i][j] for j in facilities) == 1
```

这个约束意味着需求节点 i  
仅当 j 处有设施时才能由 j 处的设施提供服务  
它隐式地消除了 X[j] = 0 但 Y[i][j] = 1 时的情况  
（节点 i 由 j 提供服务，但 j 处没有设施）  


```python
# This constraint implies that that demand node i
# can be serviced by a facility at j only if there is a facility at j
# It implicitly removes situation when X[j] = 0 but Y[i][j] = 1
# (node i is served by j but there is no facility at j)

for i in demand:
    for j in facilities:
        prob +=  Y[i][j] <= X[j]
```


```python
%%time

# Solve the above problem
prob.solve()

print("Status:", LpStatus[prob.status])
```

    Status: Optimal
    CPU times: user 1.35 s, sys: 64 ms, total: 1.42 s
    Wall time: 11.5 s
    


```python
# The minimized total demand-distance. The unit is person * meter (total distance travelled)
print("Objective: ",value(prob.objective))
```

    Objective:  469538765110.4489
    


```python
# Print the facility node.
rslt=[]
for v in prob.variables():
    subV = v.name.split('_')

    if subV[0] == "X" and v.varValue == 1:
        rslt.append(int(subV[1]))
        print('Facility Node: ', subV[1])
```

    Facility Node:  126
    Facility Node:  30
    Facility Node:  82
    


```python
# Get the geomerty of the facility nodes.
fac_loc = georgia_shp.iloc[rslt,:]
```


```python
fac_loc
```





  <div id="df-c29c13e8-3948-4cd0-90df-2834efc57c7c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AREA</th>
      <th>PERIMETER</th>
      <th>G_UTM_</th>
      <th>G_UTM_ID</th>
      <th>AREANAME</th>
      <th>Latitude</th>
      <th>Longitud</th>
      <th>TotPop90</th>
      <th>PctRural</th>
      <th>PctBach</th>
      <th>PctEld</th>
      <th>PctFB</th>
      <th>PctPov</th>
      <th>PctBlack</th>
      <th>X</th>
      <th>Y</th>
      <th>AreaKey</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>126</th>
      <td>7.315030e+08</td>
      <td>117190.0</td>
      <td>130</td>
      <td>128</td>
      <td>GA, Crisp County</td>
      <td>31.92540</td>
      <td>-83.77159</td>
      <td>20011</td>
      <td>48.4</td>
      <td>10.0</td>
      <td>12.47</td>
      <td>0.30</td>
      <td>29.0</td>
      <td>40.66</td>
      <td>805648.4</td>
      <td>3537103</td>
      <td>13081</td>
      <td>POLYGON ((787012.250 3547615.750, 820243.312 3...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.385270e+09</td>
      <td>274218.0</td>
      <td>32</td>
      <td>31</td>
      <td>GA, Fulton County</td>
      <td>33.78940</td>
      <td>-84.46716</td>
      <td>648951</td>
      <td>4.2</td>
      <td>31.6</td>
      <td>9.63</td>
      <td>4.13</td>
      <td>18.4</td>
      <td>49.92</td>
      <td>733728.4</td>
      <td>3733248</td>
      <td>13121</td>
      <td>POLYGON ((752606.688 3785970.500, 752835.062 3...</td>
    </tr>
    <tr>
      <th>82</th>
      <td>9.179670e+08</td>
      <td>121744.0</td>
      <td>84</td>
      <td>84</td>
      <td>GA, Jenkins County</td>
      <td>32.78866</td>
      <td>-81.96042</td>
      <td>8247</td>
      <td>53.8</td>
      <td>7.7</td>
      <td>13.10</td>
      <td>0.21</td>
      <td>27.8</td>
      <td>41.51</td>
      <td>970465.7</td>
      <td>3640263</td>
      <td>13165</td>
      <td>POLYGON ((989566.750 3653155.750, 981378.062 3...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-c29c13e8-3948-4cd0-90df-2834efc57c7c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-c29c13e8-3948-4cd0-90df-2834efc57c7c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-c29c13e8-3948-4cd0-90df-2834efc57c7c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5c917aea-ea4f-4856-8450-3b69dd1894fd">
  <button class="colab-df-quickchart" onclick="quickchart('df-5c917aea-ea4f-4856-8450-3b69dd1894fd')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5c917aea-ea4f-4856-8450-3b69dd1894fd button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
#Plot the faclities (stars) on top of the demand map.
fig, ax = plt.subplots(figsize=(5,5))

georgia_shp.centroid.plot(ax=ax,markersize=georgia_shp.TotPop90/1000)#markersize is proportional to the population
fac_loc.centroid.plot(ax=ax,color="red",markersize=300,marker="*")
```




    <Axes: >




    
![png](p_median_files/p_median_29_1.png)
    

