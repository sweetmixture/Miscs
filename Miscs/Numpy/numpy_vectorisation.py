import numpy as np
import pandas as pd
import time

# Create a large DataFrame
n = 10**6
df = pd.DataFrame({
    'a': np.random.rand(n),
    'b': np.random.rand(n)
})

# Method 1: Using for loop (slow)
start_loop = time.time()
result_loop = []
for i in range(n):
    result_loop.append(df['a'][i] + df['b'][i])
end_loop = time.time()

# Method 2: Using vectorized NumPy operation (fast)
start_vec = time.time()
result_vec = df['a'] + df['b']
end_vec = time.time()

# Store and display timings
timing_comparison = pd.DataFrame({
    'Method': ['For Loop', 'Vectorized'],
    'Time (seconds)': [end_loop - start_loop, end_vec - start_vec]
})

print(timing_comparison)
