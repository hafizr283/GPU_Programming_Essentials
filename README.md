# ALS Matrix Factorization

## Rating Matrix

```math
R = \begin{bmatrix}
5 & - & 4 & - & 1 \\
- & 3 & - & 5 & 2 \\
4 & - & - & 3 & 2 \\
4 & 4 & 2 & - & 5 \\
2 & - & 5 & 1 & -
\end{bmatrix}
```

## Matrix Factorization

```math
R \approx X \cdot Y^T
```

- $X$: $(5 \times 3)$
- $Y^T$: $(3 \times 5)$

```math
X = \begin{bmatrix}
0.9 & 0.2 & 0.7 \\
0.1 & 0.8 & 0.3 \\
0.7 & 0.9 & 0.6 \\
0.2 & 0.9 & 0.1 \\
0.5 & 0.6 & 0.8
\end{bmatrix}, \qquad
Y^T = \begin{bmatrix}
0.6 & 0.3 & 0.8 & 0.1 & 0.5 \\
0.2 & 0.9 & 0.4 & 0.7 & 0.3 \\
0.5 & 0.1 & 0.7 & 0.9 & 0.9
\end{bmatrix}
```

## Loss Function with Regularization ($\lambda = 0.1$)

```math
L(x, y) = \sum (r_{ui} - x_u^T y_i)^2 + \lambda \left( \|x_u\|^2 + \|y_u\|^2 \right)
```

> For known ratings only.

## ALS: Freeze $Y$, Find $X$

```math
L(x_u) = \sum (r_{ui} - x_u^T y_i)^2 + \lambda \|x_u\|^2
```

($Y_u$ part is constant, don't need to include.)

Error sum should be minimum:

```math
\frac{\partial L}{\partial x_u} = \sum 2(r_{ui} - x_u^T y_i)(-y_i) + 2\lambda x_u = 0
```

```math
\Rightarrow \sum (r_{ui} - x_u^T y_i)\, y_i = \lambda x_u
```

## Derivation of Normal Equation

```math
\sum r_{ui} y_i - \sum (x_u^T y_i)\, y_i = \lambda x_u
```

```math
\Rightarrow \sum r_{ui} y_i - \sum y_i (y_i^T x_u) = \lambda x_u
```

```math
\Rightarrow \sum r_{ui} y_i - \left\{ \sum (y_i y_i^T) \right\} x_u = \lambda x_u \qquad [x_u \text{ const}]
```

```math
\Rightarrow \left( \sum y_i y_i^T + \lambda I \right) x_u = \sum r_{ui} y_i
```

## Key Identity: $\sum y_i y_i^T = Y_u^T Y_u$

```math
Y_u = \begin{bmatrix} y_{i_1}^T \\ y_{i_2}^T \\ y_{i_3}^T \end{bmatrix}
```

**Example** with 2D factor vectors $[a_1,\, a_2]^T$ and $[b_1,\, b_2]^T$:

```math
y_i y_i^T = \begin{bmatrix} a_1^2 & a_1 a_2 \\ a_2 a_1 & a_2^2 \end{bmatrix}
```

```math
\sum y_i y_i^T = \begin{bmatrix} a_1^2 + b_1^2 & a_1 a_2 + b_1 b_2 \\ a_1 a_2 + b_1 b_2 & a_2^2 + b_2^2 \end{bmatrix}
```

```math
Y_u^T Y_u = \begin{bmatrix} a_1 & b_1 \\ a_2 & b_2 \end{bmatrix} \begin{bmatrix} a_1 & a_2 \\ b_1 & b_2 \end{bmatrix} = \begin{bmatrix} a_1^2 + b_1^2 & a_1 a_2 + b_1 b_2 \\ a_1 a_2 + b_1 b_2 & a_2^2 + b_2^2 \end{bmatrix} = \sum y_i y_i^T
```

## Key Identity: $\sum r_{ui} y_i = Y_u^T r_u$

```math
r_{u0} \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + r_{u1} \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} + \cdots = \begin{bmatrix} r_{u0} a_1 + r_{u1} b_1 \\ r_{u0} a_2 + r_{u1} b_2 \end{bmatrix} = Y_u^T r_u
```

## Normal Equation

```math
(Y_u^T Y_u + \lambda I)\, x_u = Y_u^T r_u \qquad \cdots (i)
```

```math
\boxed{x_u = (Y_u^T Y_u + \lambda I)^{-1}\, Y_u^T r_u} \qquad \cdots (ii)
```

## Solving via Cholesky Factorization

Let $A = Y_u^T Y_u + \lambda I$ and $b = Y_u^T r_u$. Equation (i) is $Ax = b$.

$A$ is **symmetric positive definite** — guaranteed by $+\lambda I$.

We can always factorize $A = L \cdot L^T$ by Cholesky factorization:

```math
L_{jj} = \sqrt{A_{jj} - \sum_{p < j} L_{jp}^2}
```

```math
L_{ij} = \frac{1}{L_{jj}} \left( A_{ij} - \sum_{p < j} L_{ip}\, L_{jp} \right) \qquad (i > j)
```

**Example** (Cholesky factor of the given matrix):

```math
L = \begin{bmatrix}
1.162 & 0 & 0 \\
0.508 & 0.363 & 0 \\
-0.128 & 0.212 & 0.577
\end{bmatrix}
```

## Solving $Ax_u = b$

```math
Ax_u = b \;\Rightarrow\; L \cdot L^T x_u = b
```

Let $z = L^T x_u \;\cdots (iii)$, so:

```math
L \cdot z = b \;\cdots (iv)
```

**Step 1 — Forward substitution** (solve eq. (iv) for $z$):

$L$ is lower triangular, so solve easily by forward substitution:

```math
\begin{bmatrix}
1.162 & 0 & 0 \\
0.508 & 0.363 & 0 \\
-0.128 & 0.212 & 0.577
\end{bmatrix}
\begin{bmatrix} z_1 \\ z_2 \\ z_3 \end{bmatrix}
= \begin{bmatrix} 6.7 \\ 2.9 \\ 6.2 \end{bmatrix}
```

```math
1.162\, z_1 = 6.7 \;\Rightarrow\; z_1 = 5.767
```

From $z_1$ find $z_2$; from $z_1$ and $z_2$ find $z_3$.

> The $b$ vector will be found by calculation; assume for now.

**Step 2 — Backward substitution** (solve eq. (iii) for $x_u$):

Using the $z$ values, solve $L^T x_u = z$ by backward substitution.

## Full ALS Algorithm

**Step 1** — Solve $x_u$:

```math
(Y_u^T Y_u + \lambda I)\, x_u = Y_u^T r_u
```

**Step 2** — Solve $y_i$ (same procedure):

```math
(X_i^T X_i + \lambda I)\, y_i = X_i^T r_i
```

**Step 3** — Check convergence. If not converged, repeat from Step 1.

## The Scalar Path: Computing $A$ and $b$

```math
A := (Y_u^T Y_u + \lambda I) = \text{L.H.S}
```

```math
b = \text{R.H.S}
```

In the left hand side, we have to calculate $Y_u^T Y_u$. We use the **register tiling** approach — register is the fastest local memory in a thread. General approach:

- `thread(0,0)` → calculates row $0 \times$ col $0$
- `thread(1,0)` → calculates row $1 \times$ col $0$

Let $u = 3$ has ratings in items $0, 3, 9$.

```math
\text{L.H.S} = Y_u^T \times Y_u = \begin{bmatrix} 0.6 & 0.1 & 0.5 \\ 0.2 & 0.7 & 0.3 \\ 0.5 & 0.4 & 0.9 \end{bmatrix} \times \begin{bmatrix} 0.6 & 0.2 & 0.5 \\ 0.1 & 0.7 & 0.4 \\ 0.5 & 0.3 & 0.9 \end{bmatrix}
```

`LHS[0][0]` = (row 0 of $Y_u^T$) $\times$ (col 0 of $Y_u$) → calculated by `thread(0,0)`

```math
\text{LHS}[0][0] = \begin{bmatrix} 0.6 & 0.1 & 0.5 \end{bmatrix} \times \begin{bmatrix} 0.6 \\ 0.1 \\ 0.5 \end{bmatrix} = 0.36 + 0.01 + 0.25 = 0.62
```

Similarly, `thread(1,0)` calculates row $1 \times$ col $0$ = `LHS[1][0]`:

```math
\text{LHS}[1][0] = \begin{bmatrix} 0.2 & 0.7 & 0.3 \end{bmatrix} \times \begin{bmatrix} 0.6 \\ 0.1 \\ 0.5 \end{bmatrix} = 0.12 + 0.07 + 0.15 = 0.34
```

### Thread Block Optimization

For $K = 64$, calculating L.H.S needs $64 \times 64 = 4096$ threads in a block. If 1 thread calculates 4 values, we only need $4096 / 4 = 1024$ threads — fits in one block (GPU max is 1024 threads per block).

- `thread(0,0)` → Row 0 col 0;  Row 0 col 1;  Row 1 col 0;  Row 1 col 1 — stored in local registers
- `thread(1,0)` → Row 2 col 0;  Row 3 col 0;  Row 2 col 1;  Row 3 col 1

Generally: `thread x*2` and `thread x*2+1` will calculate.

**Example for `thread(0,0)`** (computes $2 \times 2$ block: rows $0,1$ $\times$ cols $0,1$):

```math
\text{LHS}[0][0] = 0.62, \quad \text{LHS}[0][1] = (0.6)(0.2)+(0.1)(0.7)+(0.5)(0.3) = 0.12+0.07+0.15 = 0.34
```

```math
\text{LHS}[1][0] = 0.34, \quad \text{LHS}[1][1] = (0.2)(0.2)+(0.7)(0.7)+(0.3)(0.3) = 0.04+0.49+0.09 = 0.62
```

### R.H.S = $b$ Calculation

$Y_u^T$ portion already has calculation in L.H.S. $r_u$ is a column vector — we don't need a full R.H.S thread. Condition: if `thread_y == 0`, do $b$ calculation.

- `thread(0,0)` → `b[0]`, `b[1]`
- `thread(2,0)` → `b[2]`, `b[3]`

and so on.

For `thread(0,0)`, which already has row 0, row 1, col 0, col 1 stored, and $r_u = [4,\ 3,\ 2]^T$ (ratings of $u=3$ for items $0, 3, 9$):

```math
b[0] = \text{row 0 of } Y_u^T \times r_u = \begin{bmatrix} 0.6 & 0.1 & 0.5 \end{bmatrix} \times \begin{bmatrix} 4 \\ 3 \\ 2 \end{bmatrix} = 2.4 + 0.3 + 1.0 = 3.7
```

```math
b[1] = \text{row 1 of } Y_u^T \times r_u = \begin{bmatrix} 0.2 & 0.7 & 0.3 \end{bmatrix} \times \begin{bmatrix} 4 \\ 3 \\ 2 \end{bmatrix} = 0.8 + 2.1 + 0.6 = 3.5
```

## The Tensor Core Path (wmma — warp matrix multiplication)

Tensor core can solve matrix multiplication in just one cycle:

```math
C = A \times B + C
```

where $A$, $B$, $C$ are all $16 \times 16$ matrices.

1 warp (32 threads) can compute this multiplication-addition. The reason: every thread in a warp stores 8 register data, so 32 threads store $32 \times 8 = 256$ data ($16 \times 16$). With every thread, parallel data computes $16 \times 16$ matrix multiplication with specialized hardware acceleration — tensor core.

### What if the matrix is $32 \times 32$?

By calculating submatrix multiplications we can calculate the whole matrix ($32 \times 32$).

```math
\begin{bmatrix} A_1 & A_2 \\ B_1 & B_2 \end{bmatrix} \times \begin{bmatrix} C_1 & D_1 \\ C_2 & D_2 \end{bmatrix} = \begin{bmatrix} A_1C_1 + A_2C_2 & A_1D_1 + A_2D_2 \\ B_1C_1 + B_2C_2 & B_1D_1 + B_2D_2 \end{bmatrix}
```

All $A$, $B$, $C$, $D$ matrices are $16 \times 16$ — so we can use warp matrix multiplication here.
