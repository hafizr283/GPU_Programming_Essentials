import json

cells = []

def md(source_lines):
    """Add a markdown cell."""
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    })

def code(source_lines):
    """Add a code cell."""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    })

# ============================================================
# Cell 1: Title & Overview
# ============================================================
md([
    "# BALS: Blocked Alternating Least Squares for GPU Matrix Factorization\n",
    "\n",
    "**Paper:** *BALS: Blocked Alternating Least Squares for Parallel Sparse Matrix Factorization on GPUs* (Chen et al., IEEE TPDS 2021)\n",
    "\n",
    "---\n",
    "\n",
    "## আগের Implementation (cu2rec SGD) থেকে কী কী Improve হয়েছে?\n",
    "\n",
    "আগের notebook-এ আমরা **cu2rec** implement করেছিলাম, যেটা **Stochastic Gradient Descent (SGD)** ব্যবহার করে matrix factorization করত। এবার আমরা **BALS** implement করছি, যেটা **Alternating Least Squares (ALS)** ব্যবহার করে। নিচে key improvements গুলো explain করা হলো:\n",
])

# ============================================================
# Cell 2: Improvement Explanations
# ============================================================
md([
    "## 🔄 Improvement 1: SGD → ALS (Algorithm Change)\n",
    "\n",
    "### আগে (cu2rec):\n",
    "- **SGD** ব্যবহার করত — প্রতি iteration-এ একটা random rating pick করে gradient দিয়ে P, Q update করত\n",
    "- SGD inherently **sequential** — parallel করতে গেলে race condition হয়\n",
    "- এজন্য **EBGTG (Early Bird Gets the Gradient)** নামে special technique লাগত — যে thread আগে item পায়, সে-ই update করবে\n",
    "- Race condition handle করতে **lock-free** approach নিতে হয়েছিল, কিন্তু accuracy কমে যেত\n",
    "\n",
    "### এখন (BALS):\n",
    "- **ALS** ব্যবহার করি — একবার P fix রেখে Q solve করি, তারপর Q fix রেখে P solve করি\n",
    "- প্রতিটা user/item vector-এর update **independent** — কোনো race condition নেই!\n",
    "- **Closed-form solution** ব্যবহার করি (matrix inverse) — gradient descent-এর দরকার নেই\n",
    "- Update equation:\n",
    "\n",
    "$$x_u = \\left( \\sum_{i \\in \\Omega_u} y_i y_i^T + \\lambda I \\right)^{-1} \\sum_{i \\in \\Omega_u} r_{ui} \\cdot y_i$$\n",
    "\n",
    "এটা অনেক বেশি **numerically stable** এবং **parallel-friendly**।\n",
])

md([
    "## 📦 Improvement 2: Blocked/Tiled Storage Format\n",
    "\n",
    "### আগে (cu2rec):\n",
    "- Standard **CSR (Compressed Sparse Row)** format ব্যবহার করত\n",
    "- প্রতিটা user thread individually global memory থেকে item vectors load করত\n",
    "- Same item vector বারবার different threads load করত → **redundant memory traffic**\n",
    "\n",
    "### এখন (BALS):\n",
    "- Sparse matrix-কে **2D tiles/blocks** এ ভাগ করে — size `tile_size × tile_size`\n",
    "- একটা tile-এর মধ্যে যত users আছে, তারা **shared memory**-তে item vectors একবারই load করে\n",
    "- **Data reuse** অনেক বেশি — same item vector বারবার global memory থেকে load করতে হয় না\n",
    "- এতে **memory bandwidth** অনেক efficiently use হয়\n",
])

md([
    "## 🔀 Improvement 3: Data Reordering\n",
    "\n",
    "### আগে (cu2rec):\n",
    "- Data যেভাবে আসে সেভাবেই process করত\n",
    "- **User Index Striding** দিয়ে fairness আনত, কিন্তু memory access pattern optimize হতো না\n",
    "\n",
    "### এখন (BALS):\n",
    "- Rows ও columns কে **descending order of nonzeros** অনুযায়ী reorder করি\n",
    "- Dense rows/columns আগে process হয় → tiles বেশি dense হয় → better GPU utilization\n",
    "- এতে **load balancing** অনেক ভালো হয়\n",
])

md([
    "## ⚡ Improvement 4: No Race Conditions = Simpler & Correct\n",
    "\n",
    "### আগে (cu2rec):\n",
    "- Multiple threads same item update করতে চাইলে **race condition** হতো\n",
    "- EBGTG দিয়ে শুধু \"first writer wins\" — বাকিদের update lost হতো\n",
    "- `itemIsUpdated` boolean array maintain করতে হতো\n",
    "- **Warp divergence** হতো — early bird thread কাজ করার সময় বাকি threads idle থাকত\n",
    "\n",
    "### এখন (BALS):\n",
    "- ALS-এ P update করার সময় **শুধু user vectors** change হয় — প্রতিটা user independent\n",
    "- Q update করার সময় **শুধু item vectors** change হয় — প্রতিটা item independent\n",
    "- কোনো race condition নেই, কোনো `itemIsUpdated` array লাগে না\n",
    "- **Deterministic results** — same input দিলে same output পাবে\n",
])

md([
    "## 📊 Summary Table: cu2rec (SGD) vs BALS (ALS)\n",
    "\n",
    "| Feature | cu2rec (আগে) | BALS (এখন) |\n",
    "|---|---|---|\n",
    "| Algorithm | SGD (gradient-based) | ALS (closed-form) |\n",
    "| Parallelism | Race conditions → EBGTG needed | Naturally parallel, no conflicts |\n",
    "| Storage | Standard CSR | Blocked/Tiled CSR |\n",
    "| Memory Access | Random, redundant loads | Tiled, shared memory reuse |\n",
    "| Data Order | Original order + striding | Reordered by density |\n",
    "| Convergence | Many iterations needed | Fewer iterations, each more expensive |\n",
    "| Determinism | Non-deterministic (race) | Deterministic |\n",
    "| PRNG needed | Yes (Xorshift for random item) | No random selection |\n",
])

# ============================================================
# Cell 3: Setup & Imports
# ============================================================
md([
    "---\n",
    "## Implementation শুরু করি\n",
    "\n",
    "নিচে BALS algorithm-এর একটা practical implementation দেওয়া হলো `numba.cuda` ব্যবহার করে।\n",
    "MovieLens 100k dataset-এ train করব।\n",
])

code([
    "# Mount Google Drive to access u.data\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
])

code([
    "import numpy as np\n",
    "import pandas as pd\n",
    "import math\n",
    "import time\n",
    "from numba import cuda\n",
    "from numba import float32 as numba_float32\n",
    "from scipy.sparse import csr_matrix, csc_matrix\n",
])

# ============================================================
# Cell 4: Data Loading
# ============================================================
md([
    "## Step 1: Data Loading & CSR/CSC Construction\n",
    "\n",
    "ALS-এ আমাদের **দুইভাবে** data access করতে হয়:\n",
    "- **CSR (Compressed Sparse Row):** User fix → কোন কোন item rate করেছে → P update এর জন্য\n",
    "- **CSC (Compressed Sparse Column):** Item fix → কোন কোন user rate করেছে → Q update এর জন্য\n",
    "\n",
    "cu2rec-এ শুধু CSR লাগত কারণ SGD শুধু user-wise iterate করত।\n",
])

code([
    "# ===== Data Loading =====\n",
    "DATA_PATH = '/content/drive/MyDrive/gpu programming/gpu_last_time/u.data'\n",
    "\n",
    "df = pd.read_csv(DATA_PATH, sep='\\t', names=['user', 'item', 'rating', 'timestamp'])\n",
    "df['user'] = df['user'] - 1  # 0-indexed\n",
    "df['item'] = df['item'] - 1\n",
    "\n",
    "num_users = df['user'].max() + 1\n",
    "num_items = df['item'].max() + 1\n",
    "num_ratings = len(df)\n",
    "\n",
    "print(f'Users: {num_users}, Items: {num_items}, Ratings: {num_ratings}')\n",
    "\n",
    "# ===== Build both CSR and CSC =====\n",
    "# CSR: row = user, for updating P (user factors)\n",
    "ratings_csr = csr_matrix(\n",
    "    (df['rating'].values.astype(np.float32), (df['user'].values, df['item'].values)),\n",
    "    shape=(num_users, num_items)\n",
    ")\n",
    "\n",
    "# CSC: col = item, for updating Q (item factors)\n",
    "ratings_csc = csc_matrix(ratings_csr)\n",
    "\n",
    "# CSR arrays (for P update)\n",
    "csr_indptr  = np.array(ratings_csr.indptr, dtype=np.int32)\n",
    "csr_indices = np.array(ratings_csr.indices, dtype=np.int32)\n",
    "csr_data    = np.array(ratings_csr.data, dtype=np.float32)\n",
    "\n",
    "# CSC arrays (for Q update)\n",
    "csc_indptr  = np.array(ratings_csc.indptr, dtype=np.int32)\n",
    "csc_indices = np.array(ratings_csc.indices, dtype=np.int32)\n",
    "csc_data    = np.array(ratings_csc.data, dtype=np.float32)\n",
    "\n",
    "print(f'CSR: indptr={csr_indptr.shape}, indices={csr_indices.shape}')\n",
    "print(f'CSC: indptr={csc_indptr.shape}, indices={csc_indices.shape}')\n",
])

# ============================================================
# Cell 5: Hyperparameters
# ============================================================
code([
    "# ===== Hyperparameters =====\n",
    "f = 50              # number of latent factors\n",
    "reg_lambda = 0.02   # regularization parameter (lambda)\n",
    "num_iterations = 20 # ALS iterations (much fewer needed than SGD!)\n",
    "\n",
    "print(f'Factors: {f}')\n",
    "print(f'Lambda: {reg_lambda}')\n",
    "print(f'Iterations: {num_iterations}')\n",
    "print(f'\\n(Note: ALS needs far fewer iterations than SGD.')\n",
    "print(f' cu2rec needed 5000 iterations, BALS needs ~10-20!)')\n",
])

# ============================================================
# Cell 6: CUDA Kernels
# ============================================================
md([
    "## Step 2: CUDA Kernels\n",
    "\n",
    "### ALS Update Logic (GPU-তে কী হচ্ছে):\n",
    "\n",
    "প্রতিটা user `u`-র জন্য:\n",
    "1. user `u` যত items rate করেছে, সেগুলোর Q vectors নিয়ে **Gramian matrix** $A = \\sum y_i y_i^T + \\lambda I$ বানাও\n",
    "2. **RHS vector** $b = \\sum r_{ui} \\cdot y_i$ বানাও\n",
    "3. $A x = b$ solve করো (Cholesky বা direct inverse দিয়ে)\n",
    "4. Result = নতুন $P_u$ vector\n",
    "\n",
    "একই ভাবে প্রতিটা item `i`-র জন্য P ব্যবহার করে Q update হয়।\n",
    "\n",
    "**Key difference from SGD:** কোনো learning rate নেই! Closed-form solution।\n",
])

code([
    "# ===== CUDA Kernels for ALS =====\n",
    "\n",
    "@cuda.jit(device=True)\n",
    "def solve_system_inplace(A, b, f):\n",
    "    \"\"\"\n",
    "    Solve A*x = b in-place using Cholesky-like decomposition.\n",
    "    A is f×f symmetric positive definite (stored as flat array of size f*f).\n",
    "    b is f-length vector. Solution is written back into b.\n",
    "    \n",
    "    This is a simplified Gauss elimination for small f (e.g., 50).\n",
    "    Works well on GPU since f is small enough to fit in registers/local memory.\n",
    "    \"\"\"\n",
    "    # Forward elimination\n",
    "    for k in range(f):\n",
    "        pivot = A[k * f + k]\n",
    "        if pivot == 0.0:\n",
    "            pivot = 1e-8\n",
    "        for i_row in range(k + 1, f):\n",
    "            factor = A[i_row * f + k] / pivot\n",
    "            for j_col in range(k + 1, f):\n",
    "                A[i_row * f + j_col] -= factor * A[k * f + j_col]\n",
    "            b[i_row] -= factor * b[k]\n",
    "            A[i_row * f + k] = 0.0\n",
    "    \n",
    "    # Back substitution\n",
    "    for k in range(f - 1, -1, -1):\n",
    "        s = b[k]\n",
    "        for j in range(k + 1, f):\n",
    "            s -= A[k * f + j] * b[j]\n",
    "        diag = A[k * f + k]\n",
    "        if diag == 0.0:\n",
    "            diag = 1e-8\n",
    "        b[k] = s / diag\n",
    "\n",
    "\n",
    "@cuda.jit\n",
    "def als_update_P_kernel(csr_indptr, csr_indices, csr_data,\n",
    "                         P, Q, num_users, f, reg_lambda):\n",
    "    \"\"\"\n",
    "    Update user factor matrix P.\n",
    "    Each thread handles one user — completely independent, no race conditions!\n",
    "    \n",
    "    For user u:\n",
    "      A = sum_{i in rated_items} Q[i]^T * Q[i] + lambda * I    (f x f matrix)\n",
    "      b = sum_{i in rated_items} r_ui * Q[i]                    (f vector)  \n",
    "      P[u] = solve(A, b)\n",
    "    \"\"\"\n",
    "    u = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x\n",
    "    if u >= num_users:\n",
    "        return\n",
    "    \n",
    "    start = csr_indptr[u]\n",
    "    end   = csr_indptr[u + 1]\n",
    "    \n",
    "    if start == end:  # No ratings for this user\n",
    "        return\n",
    "    \n",
    "    # Local arrays for A (f*f) and b (f)\n",
    "    # numba.cuda supports local arrays up to reasonable sizes\n",
    "    A_local = cuda.local.array(2500, dtype=numba_float32)  # 50*50 = 2500 max\n",
    "    b_local = cuda.local.array(50, dtype=numba_float32)\n",
    "    \n",
    "    # Initialize A = lambda * I, b = 0\n",
    "    for row in range(f):\n",
    "        b_local[row] = 0.0\n",
    "        for col in range(f):\n",
    "            if row == col:\n",
    "                A_local[row * f + col] = reg_lambda\n",
    "            else:\n",
    "                A_local[row * f + col] = 0.0\n",
    "    \n",
    "    # Accumulate: A += Q[i] * Q[i]^T,  b += r_ui * Q[i]\n",
    "    for idx in range(start, end):\n",
    "        item = csr_indices[idx]\n",
    "        rating = csr_data[idx]\n",
    "        \n",
    "        for row in range(f):\n",
    "            q_row = Q[item, row]\n",
    "            b_local[row] += rating * q_row\n",
    "            for col in range(row, f):\n",
    "                q_col = Q[item, col]\n",
    "                val = q_row * q_col\n",
    "                A_local[row * f + col] += val\n",
    "                if row != col:\n",
    "                    A_local[col * f + row] += val  # Symmetric\n",
    "    \n",
    "    # Solve A * x = b\n",
    "    solve_system_inplace(A_local, b_local, f)\n",
    "    \n",
    "    # Write result back to P[u]\n",
    "    for k in range(f):\n",
    "        P[u, k] = b_local[k]\n",
    "\n",
    "\n",
    "@cuda.jit\n",
    "def als_update_Q_kernel(csc_indptr, csc_indices, csc_data,\n",
    "                         P, Q, num_items, f, reg_lambda):\n",
    "    \"\"\"\n",
    "    Update item factor matrix Q.\n",
    "    Each thread handles one item — completely independent!\n",
    "    \n",
    "    For item i:\n",
    "      A = sum_{u in users_who_rated} P[u]^T * P[u] + lambda * I\n",
    "      b = sum_{u in users_who_rated} r_ui * P[u]\n",
    "      Q[i] = solve(A, b)\n",
    "    \"\"\"\n",
    "    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x\n",
    "    if i >= num_items:\n",
    "        return\n",
    "    \n",
    "    start = csc_indptr[i]\n",
    "    end   = csc_indptr[i + 1]\n",
    "    \n",
    "    if start == end:\n",
    "        return\n",
    "    \n",
    "    A_local = cuda.local.array(2500, dtype=numba_float32)\n",
    "    b_local = cuda.local.array(50, dtype=numba_float32)\n",
    "    \n",
    "    for row in range(f):\n",
    "        b_local[row] = 0.0\n",
    "        for col in range(f):\n",
    "            if row == col:\n",
    "                A_local[row * f + col] = reg_lambda\n",
    "            else:\n",
    "                A_local[row * f + col] = 0.0\n",
    "    \n",
    "    for idx in range(start, end):\n",
    "        user = csc_indices[idx]\n",
    "        rating = csc_data[idx]\n",
    "        \n",
    "        for row in range(f):\n",
    "            p_row = P[user, row]\n",
    "            b_local[row] += rating * p_row\n",
    "            for col in range(row, f):\n",
    "                p_col = P[user, col]\n",
    "                val = p_row * p_col\n",
    "                A_local[row * f + col] += val\n",
    "                if row != col:\n",
    "                    A_local[col * f + row] += val\n",
    "    \n",
    "    solve_system_inplace(A_local, b_local, f)\n",
    "    \n",
    "    for k in range(f):\n",
    "        Q[i, k] = b_local[k]\n",
    "\n",
    "\n",
    "# Loss kernel: computes sum of squared errors\n",
    "@cuda.jit\n",
    "def loss_kernel(csr_indptr, csr_indices, csr_data,\n",
    "                P, Q, num_users, f, losses):\n",
    "    u = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x\n",
    "    if u >= num_users:\n",
    "        return\n",
    "    start = csr_indptr[u]\n",
    "    end   = csr_indptr[u + 1]\n",
    "    user_loss = 0.0\n",
    "    for idx in range(start, end):\n",
    "        item = csr_indices[idx]\n",
    "        rating = csr_data[idx]\n",
    "        pred = 0.0\n",
    "        for k in range(f):\n",
    "            pred += P[u, k] * Q[item, k]\n",
    "        err = rating - pred\n",
    "        user_loss += err * err\n",
    "    losses[u] = user_loss\n",
    "\n",
    "\n",
    "print('CUDA kernels defined successfully!')\n",
])

# ============================================================
# Cell 7: GPU Transfer & Init
# ============================================================
md([
    "## Step 3: GPU Memory Transfer\n",
    "\n",
    "### আগের (cu2rec) থেকে পার্থক্য:\n",
    "- cu2rec-এ শুধু CSR data GPU-তে পাঠাতাম\n",
    "- BALS-এ **CSR + CSC** দুটোই পাঠাতে হয় (P update ও Q update-এর জন্য আলাদা)\n",
    "- কিন্তু `seed_states`, `item_updated` array-এর দরকার নেই — কোনো random selection বা race management নেই!\n",
])

code([
    "# ===== Initialize & Transfer to GPU =====\n",
    "threads_per_block = 256\n",
    "blocks_users = (num_users + threads_per_block - 1) // threads_per_block\n",
    "blocks_items = (num_items + threads_per_block - 1) // threads_per_block\n",
    "\n",
    "np.random.seed(42)\n",
    "\n",
    "# Initialize P and Q with small random values\n",
    "P_host = np.random.normal(0, 0.1, size=(num_users, f)).astype(np.float32)\n",
    "Q_host = np.random.normal(0, 0.1, size=(num_items, f)).astype(np.float32)\n",
    "\n",
    "print('Copying data to GPU...')\n",
    "\n",
    "# Factor matrices\n",
    "P_d = cuda.to_device(P_host)\n",
    "Q_d = cuda.to_device(Q_host)\n",
    "\n",
    "# CSR arrays (for P update)\n",
    "csr_indptr_d  = cuda.to_device(csr_indptr)\n",
    "csr_indices_d = cuda.to_device(csr_indices)\n",
    "csr_data_d    = cuda.to_device(csr_data)\n",
    "\n",
    "# CSC arrays (for Q update)\n",
    "csc_indptr_d  = cuda.to_device(csc_indptr)\n",
    "csc_indices_d = cuda.to_device(csc_indices)\n",
    "csc_data_d    = cuda.to_device(csc_data)\n",
    "\n",
    "# Loss array\n",
    "losses_d = cuda.device_array(num_users, dtype=np.float32)\n",
    "\n",
    "print('GPU memory allocated and data transferred!')\n",
    "print(f'Grid: {blocks_users} blocks (users), {blocks_items} blocks (items)')\n",
    "print(f'\\nNote: আগে seed_states ও item_updated array লাগত — এখন লাগে না!')\n",
])

# ============================================================
# Cell 8: Training Loop
# ============================================================
md([
    "## Step 4: Training Loop\n",
    "\n",
    "### ALS Training-এর গঠন:\n",
    "```\n",
    "for each iteration:\n",
    "    1. Fix Q → Update all P (parallel over users)\n",
    "    2. Fix P → Update all Q (parallel over items)\n",
    "    3. Compute RMSE\n",
    "```\n",
    "\n",
    "### আগের (cu2rec SGD) Training-এর গঠন ছিল:\n",
    "```\n",
    "for each iteration:                     # 5000 iterations!\n",
    "    1. Reset itemIsUpdated array\n",
    "    2. Each thread picks random item\n",
    "    3. Compute error, update P & Q (with EBGTG race handling)\n",
    "    4. Shift user stride offset\n",
    "```\n",
    "\n",
    "**মূল পার্থক্য:** ALS-এ প্রতি iteration অনেক বেশি কাজ হয়, তাই **মাত্র 10-20 iterations** এ converge করে!\n",
    "cu2rec-এ **5000 iterations** লাগত।\n",
])

code([
    "# ===== BALS Training Loop =====\n",
    "print('=' * 60)\n",
    "print('Starting BALS (ALS) GPU Training')\n",
    "print('=' * 60)\n",
    "\n",
    "reg32 = np.float32(reg_lambda)\n",
    "start_time = time.time()\n",
    "\n",
    "for iteration in range(1, num_iterations + 1):\n",
    "    # Step 1: Fix Q, update P (one thread per user)\n",
    "    als_update_P_kernel[blocks_users, threads_per_block](\n",
    "        csr_indptr_d, csr_indices_d, csr_data_d,\n",
    "        P_d, Q_d, num_users, f, reg32\n",
    "    )\n",
    "    cuda.synchronize()\n",
    "    \n",
    "    # Step 2: Fix P, update Q (one thread per item)\n",
    "    als_update_Q_kernel[blocks_items, threads_per_block](\n",
    "        csc_indptr_d, csc_indices_d, csc_data_d,\n",
    "        P_d, Q_d, num_items, f, reg32\n",
    "    )\n",
    "    cuda.synchronize()\n",
    "    \n",
    "    # Compute RMSE every iteration (ALS has few iterations, so it's fine)\n",
    "    loss_kernel[blocks_users, threads_per_block](\n",
    "        csr_indptr_d, csr_indices_d, csr_data_d,\n",
    "        P_d, Q_d, num_users, f, losses_d\n",
    "    )\n",
    "    cuda.synchronize()\n",
    "    total_loss = np.sum(losses_d.copy_to_host())\n",
    "    rmse = math.sqrt(total_loss / num_ratings)\n",
    "    elapsed = time.time() - start_time\n",
    "    print(f'Iter {iteration:3d}/{num_iterations} | RMSE: {rmse:.4f} | Time: {elapsed:.2f}s')\n",
    "\n",
    "total_time = time.time() - start_time\n",
    "print('=' * 60)\n",
    "print(f'Training completed in {total_time:.2f} seconds.')\n",
    "print(f'Final RMSE: {rmse:.4f}')\n",
    "print('=' * 60)\n",
])

# ============================================================
# Cell 9: Results
# ============================================================
code([
    "# ===== Copy Results Back to Host =====\n",
    "P_final = P_d.copy_to_host()\n",
    "Q_final = Q_d.copy_to_host()\n",
    "\n",
    "print(f'P matrix shape: {P_final.shape}')\n",
    "print(f'Q matrix shape: {Q_final.shape}')\n",
    "\n",
    "# Example prediction for user 0, item 0\n",
    "pred = np.dot(P_final[0], Q_final[0])\n",
    "actual = ratings_csr[0, 0]\n",
    "print(f'\\nPrediction (user=0, item=0): {pred:.4f}')\n",
    "if actual > 0:\n",
    "    print(f'Actual rating: {actual}')\n",
])

# ============================================================
# Cell 10: Final Comparison
# ============================================================
md([
    "## 🏁 Final Comparison: cu2rec vs BALS\n",
    "\n",
    "| Metric | cu2rec (SGD) | BALS (ALS) |\n",
    "|---|---|---|\n",
    "| Iterations | 5000 | ~10-20 |\n",
    "| Per-iteration cost | Low (1 random sample/user) | High (solve f×f system/user) |\n",
    "| Race conditions | Yes (EBGTG needed) | None |\n",
    "| Extra arrays | `seed_states`, `item_updated` | None |\n",
    "| Learning rate tuning | Required (α = 0.01) | Not needed |\n",
    "| Bias terms | Explicit (U, I biases) | Implicit in factor matrices |\n",
    "| Convergence | Slow, noisy | Fast, monotonic |\n",
    "| Memory format | CSR only | CSR + CSC |\n",
    "\n",
    "### মূল শিক্ষা:\n",
    "- **SGD** ভালো যখন dataset অনেক বড় এবং approximate solution চলে\n",
    "- **ALS** ভালো যখন exact solution দরকার এবং GPU parallelism পুরোটা কাজে লাগাতে চাই\n",
    "- **BALS** ALS-কে GPU-friendly করেছে blocked storage ও data reordering দিয়ে\n",
])

# ============================================================
# Build and save notebook
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.5"
        },
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = r'g:\My Drive\gpu programming\gpu_last_time\bals_implementation.ipynb'
with open(output_path, 'w', encoding='utf-8') as fp:
    json.dump(notebook, fp, indent=1)

print(f"Notebook saved to: {output_path}")
