# ALS Matrix Factorization

## Rating Matrix

$$R = \begin{bmatrix} 5 & - & 4 & - & 1 \\ - & 3 & - & 5 & 2 \\ 4 & - & - & 3 & 2 \\ 4 & 4 & 2 & - & 5 \\ 2 & - & 5 & 1 & - \end{bmatrix}$$

## Matrix Factorization

$$R \approx X \cdot Y^T$$

- $X$: $(5 \times 3)$
- $Y^T$: $(3 \times 5)$

$$X = \begin{bmatrix} 0.9 & 0.2 & 0.7 \\ 0.1 & 0.8 & 0.3 \\ 0.7 & 0.9 & 0.6 \\ 0.2 & 0.9 & 0.1 \\ 0.5 & 0.6 & 0.8 \end{bmatrix}, \qquad Y^T = \begin{bmatrix} 0.6 & 0.3 & 0.8 & 0.1 & 0.5 \\ 0.2 & 0.9 & 0.4 & 0.7 & 0.3 \\ 0.5 & 0.1 & 0.7 & 0.9 & 0.9 \end{bmatrix}$$

## Loss Function with Regularization ($\lambda = 0.1$)

$$L(x, y) = \sum (r_{ui} - x_u^T y_i)^2 + \lambda \left( \|x_u\|^2 + \|y_u\|^2 \right)$$

> For known ratings only.

## ALS: Freeze $Y$, Find $X$

$$L(x_u) = \sum (r_{ui} - x_u^T y_i)^2 + \lambda \|x_u\|^2$$

($Y_u$ part is constant, don't need to include.)

Error sum should be minimum:

$$\frac{\partial L}{\partial x_u} = \sum 2(r_{ui} - x_u^T y_i)(-y_i) + 2\lambda x_u = 0$$

$$\Rightarrow \sum (r_{ui} - x_u^T y_i)\, y_i = \lambda x_u$$

## Derivation of Normal Equation

$$\sum r_{ui} y_i - \sum (x_u^T y_i)\, y_i = \lambda x_u$$

$$\Rightarrow \sum r_{ui} y_i - \sum y_i (y_i^T x_u) = \lambda x_u$$

$$\Rightarrow \sum r_{ui} y_i - \left\{ \sum (y_i y_i^T) \right\} x_u = \lambda x_u \qquad [x_u \text{ const}]$$

$$\Rightarrow \left( \sum y_i y_i^T + \lambda I \right) x_u = \sum r_{ui} y_i$$

## Key Identity: $\sum y_i y_i^T = Y_u^T Y_u$

$$Y_u = \begin{bmatrix} y_{i_1}^T \\ y_{i_2}^T \\ y_{i_3}^T \end{bmatrix}$$

**Example** with 2D factor vectors $[a_1,\, a_2]^T$ and $[b_1,\, b_2]^T$:

$$y_i y_i^T = \begin{bmatrix} a_1^2 & a_1 a_2 \\ a_2 a_1 & a_2^2 \end{bmatrix}$$

$$\sum y_i y_i^T = \begin{bmatrix} a_1^2 + b_1^2 & a_1 a_2 + b_1 b_2 \\ a_1 a_2 + b_1 b_2 & a_2^2 + b_2^2 \end{bmatrix}$$

$$Y_u^T Y_u = \begin{bmatrix} a_1 & b_1 \\ a_2 & b_2 \end{bmatrix} \begin{bmatrix} a_1 & a_2 \\ b_1 & b_2 \end{bmatrix} = \begin{bmatrix} a_1^2 + b_1^2 & a_1 a_2 + b_1 b_2 \\ a_1 a_2 + b_1 b_2 & a_2^2 + b_2^2 \end{bmatrix} = \sum y_i y_i^T$$

## Key Identity: $\sum r_{ui} y_i = Y_u^T r_u$

$$r_{u0} \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + r_{u1} \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} + \cdots = \begin{bmatrix} r_{u0} a_1 + r_{u1} b_1 \\ r_{u0} a_2 + r_{u1} b_2 \end{bmatrix} = Y_u^T r_u$$

## Normal Equation

$$(Y_u^T Y_u + \lambda I)\, x_u = Y_u^T r_u \qquad \cdots (i)$$

$$\boxed{x_u = (Y_u^T Y_u + \lambda I)^{-1}\, Y_u^T r_u}$$

## Solving via Cholesky Factorization

Let $A = Y_u^T Y_u + \lambda I$ and $b = Y_u^T r_u$. Equation (i) is $Ax = b$.

$A$ is **symmetric positive definite** — guaranteed by $+\lambda I$.

We can always factorize $A = L \cdot L^T$ by Cholesky factorization:

$$L_{jj} = \sqrt{A_{jj} - \sum_{p < j} L_{jp}^2}$$

$$L_{ij} = \frac{1}{L_{jj}} \left( A_{ij} - \sum_{p < j} L_{ip}\, L_{jp} \right) \qquad (i > j)$$

**Example** (Cholesky factor of the given matrix):

$$L = \begin{bmatrix} 1.162 & 0 & 0 \\ 0.508 & 0.363 & 0 \\ -0.128 & 0.212 & 0.577 \end{bmatrix}$$

## Solving $Ax_u = b$

$$Ax_u = b \;\Rightarrow\; L \cdot L^T x_u = b$$

Let $z = L^T x_u \;\cdots (iii)$, so:

$$L \cdot z = b \;\cdots (iv)$$

**Step 1 — Forward substitution** (solve eq. (iv) for $z$):

$L$ is lower triangular, so solve easily by forward substitution:

$$\begin{bmatrix} 1.162 & 0 & 0 \\ 0.508 & 0.363 & 0 \\ -0.128 & 0.212 & 0.577 \end{bmatrix} \begin{bmatrix} z_1 \\ z_2 \\ z_3 \end{bmatrix} = \begin{bmatrix} 6.7 \\ 2.9 \\ 6.2 \end{bmatrix}$$

$$1.162\, z_1 = 6.7 \;\Rightarrow\; z_1 = 5.767$$

From $z_1$ find $z_2$; from $z_1$ and $z_2$ find $z_3$.

> The $b$ vector will be found by calculation; assume for now.

**Step 2 — Backward substitution** (solve eq. (iii) for $x_u$):

Using the $z$ values, solve $L^T x_u = z$ by backward substitution.

## Full ALS Algorithm

**Step 1** — Solve $x_u$:

$$(Y_u^T Y_u + \lambda I)\, x_u = Y_u^T r_u$$

**Step 2** — Solve $y_i$ (same procedure):

$$(X_i^T X_i + \lambda I)\, y_i = X_i^T p_u$$

**Step 3** — Check convergence. If not converged, repeat from Step 1.
