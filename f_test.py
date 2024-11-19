#%%
import numpy as np
import pandas as pd
import ast
import re
#%%
def converter(s):
    s = s[1:-2]
    s = s.decode('utf-8')
    # Split the string into individual numbers and convert them to floats
    f = np.array([float(x) for x in s.split(',')])
    return f

path = "labopt_logs/lab_opt_2024_11/optimization__2024-11-17-13-14__dual_annealing__.txt"
msgs = ["logs"]
head_numbers = 8
#%%
with open(path, 'r', encoding='utf-8') as file:
    for current_line, line in enumerate(file, start=1):  # 从第1行开始计数
        msgs.append(line.strip())
        if current_line > head_numbers:
            break
        
for i in msgs:
    print(i)

method = msgs[2]
method = method[8::]

data_vec = np.loadtxt(path,skiprows = head_numbers,usecols=(2),converters = {2: converter},dtype = np.ndarray,max_rows = 5)
value = np.loadtxt(path,skiprows = head_numbers,usecols=(3))


    
