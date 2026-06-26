import json

cells = []
def add_markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text]})

add_markdown("# GPU Accelerated Matrix Factorization\n\nImplementation of Early Bird Gets the Gradient (EBGTG) parallel SGD.")

add_code("!pip install cupy-cuda12x scipy numpy pandas")

add_code("""import numpy as np
import cupy as cp
import pandas as pd
import time
from scipy.sparse import csr_matrix
import urllib.request
import os""")

add_markdown("## 1. Data Preparation\nLoad `u.data` from MovieLens 100k, and structure it into a CSR matrix for fast item retrieval within the CUDA kernel.")
add_code("""# If u.data doesn't exist locally, download it
if not os.path.exists('u.data'):
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data'
    urllib.request.urlretrieve(url, 'u.data')

names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\\t', names=names)

# Convert to 0-based index
df['user_id'] -= 1
df['item_id'] -= 1

num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()
print(f"Users: {num_users}, Items: {num_items}")

# Split into 80% train, 20% test globally random
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Create CSR matrix using scipy so we can extract indptr, indices, data easily
train_csr = csr_matrix((train_df['rating'], (train_df['user_id'], train_df['item_id'])), shape=(num_users, num_items))

indptr = cp.asarray(train_csr.indptr)
indices = cp.asarray(train_csr.indices)
data = cp.asarray(train_csr.data, dtype=cp.float32)

test_users = cp.asarray(test_df['user_id'].values)
test_items = cp.asarray(test_df['item_id'].values)
test_ratings = cp.asarray(test_df['rating'].values, dtype=cp.float32)

global_mean = float(train_df['rating'].mean())
print(f"Global Mean Rating: {global_mean:.4f}")
""")

add_markdown("## 2. CuPy RawKernel Definition\nWe define the custom CUDA kernel implementing the `Early Bird Gets the Gradient` logic to resolve updating races on the feature weights.")
add_code("""sgd_kernel = cp.RawKernel(r'''
extern "C" __global__
void mf_sgd_ebgtg(const int* indptr, const int* indices, const float* data,
                  float* P, float* Q, float* QTrgt, 
                  float* userBias, float* itemBias, float* itemBiasTrgt,
                  bool* itemIsUpdated,
                  int num_users, int num_factors, int stride_offset,
                  float mu, float lr, float reg_p, float reg_q, float reg_u, float reg_i,
                  unsigned long long seed) {
    
    // Each thread gets a unique user
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_users) return;
    
    // User index striding (as per the paper to avoid over-fitting front users)
    int u = (tid + stride_offset) % num_users;
    
    int start_idx = indptr[u];
    int end_idx = indptr[u+1];
    int num_ratings = end_idx - start_idx;
    
    if (num_ratings == 0) return;
    
    // Pick a random rating for this user using LCG PRNG
    unsigned long long rnd = seed ^ (u * 19937ULL);
    rnd = rnd * 2862933555777941757ULL + 7046029254386353087ULL;
    int rand_offset = rnd % num_ratings;
    
    int data_idx = start_idx + rand_offset;
    int i = indices[data_idx];
    float true_rating = data[data_idx];
    
    // Cache heavily reused values into registers
    float ub = userBias[u];
    float ib = itemBias[i];
    
    float pred = mu + ub + ib;
    for(int f=0; f < num_factors; ++f) {
        pred += P[u * num_factors + f] * Q[i * num_factors + f];
    }
    
    float error = true_rating - pred;
    
    // EBGTG logic
    bool earlyBird = !itemIsUpdated[i];
    itemIsUpdated[i] = true;
    
    // Update features
    for(int f=0; f < num_factors; ++f) {
        float pOld = P[u * num_factors + f];
        float qOld = Q[i * num_factors + f];
        
        P[u * num_factors + f] = pOld + lr * (error * qOld - reg_p * pOld);
        
        if (earlyBird) {
            QTrgt[i * num_factors + f] = qOld + lr * (error * pOld - reg_q * qOld);
        }
    }
    
    userBias[u] = ub + lr * (error - reg_u * ub);
    if (earlyBird) {
        itemBiasTrgt[i] = ib + lr * (error - reg_i * ib);
    }
}
''', 'mf_sgd_ebgtg')""")

add_markdown("## 3. Training Loop")
add_code("""# Hyperparameters
num_factors = 50
learning_rate = 0.01
reg_p = 0.02
reg_q = 0.02
reg_u = 0.02
reg_i = 0.02
iterations = 1000

# Memory Initialization
cp.random.seed(42)
P = cp.random.normal(0, 1.0/num_factors, size=(num_users, num_factors), dtype=cp.float32)
Q = cp.random.normal(0, 1.0/num_factors, size=(num_items, num_factors), dtype=cp.float32)
QTrgt = Q.copy()

userBias = cp.zeros(num_users, dtype=cp.float32)
itemBias = cp.zeros(num_items, dtype=cp.float32)
itemBiasTrgt = itemBias.copy()

itemIsUpdated = cp.zeros(num_items, dtype=cp.bool_)

# Hardware Launch Config
threads_per_block = 256
blocks_per_grid = (num_users + threads_per_block - 1) // threads_per_block

def get_rmse(P, Q, userBias, itemBias, u_idx, i_idx, true_ratings, mu):
    p_u = P[u_idx]
    q_i = Q[i_idx]
    dot_prod = cp.sum(p_u * q_i, axis=1)
    preds = mu + userBias[u_idx] + itemBias[i_idx] + dot_prod
    mse = cp.mean((true_ratings - preds)**2)
    return float(cp.sqrt(mse))

print(f"Starting Training for {iterations} iterations on GPU...")
start_time = time.time()
stride_offset = 0

for it in range(iterations):
    # Wipe the updated items array
    itemIsUpdated.fill(False)
    
    # Kernel Launch
    seed = cp.random.randint(0, 100000000)
    sgd_kernel((blocks_per_grid,), (threads_per_block,), 
               (indptr, indices, data,
                P, Q, QTrgt,
                userBias, itemBias, itemBiasTrgt,
                itemIsUpdated,
                num_users, num_factors, stride_offset,
                global_mean, learning_rate, reg_p, reg_q, reg_u, reg_i,
                seed))
    
    cp.cuda.Stream.null.synchronize()
    
    # Swap pointers efficiently 
    Q, QTrgt = QTrgt, Q
    itemBias, itemBiasTrgt = itemBiasTrgt, itemBias
    
    # Stride index shift to improve fairness across blocks
    stride_offset = (stride_offset + 131) % num_users
    
    if (it + 1) % 100 == 0:
        test_rmse = get_rmse(P, Q, userBias, itemBias, test_users, test_items, test_ratings, global_mean)
        print(f"Iteration {it+1:04d}/{iterations} | Test RMSE: {test_rmse:.4f}")

end_time = time.time()
print(f"\\nTraining completed in {end_time - start_time:.2f} seconds.")
final_rmse = get_rmse(P, Q, userBias, itemBias, test_users, test_items, test_ratings, global_mean)
print(f"Final Test RMSE: {final_rmse:.4f}")
""")

nb = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(r'practice_pd.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
