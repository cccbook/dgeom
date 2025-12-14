### 第五章：N 維陣列與座標表示 (N-dimensional Arrays & Explicit Components)
*本章介紹如何將抽象張量落實到具體的數值或函數陣列中。*
5.1 **SymPy 的 Array 模組**
   - 建立矩陣與高維陣列
5.2 **張量積與縮併 (Array Operations)**
   - `tensorproduct` (外積)
   - `tensorcontraction` (內積/縮併)
5.3 **陣列導數 (Array Derivatives)**
   - `derive_by_array`: 對張量進行偏微分

*(請複製以下程式碼至新的儲存格執行)*

```python
from sympy import symbols, sin, cos, init_printing
from sympy import Array, tensorproduct, tensorcontraction, derive_by_array
from sympy.abc import x, y, z # 引入常用符號

init_printing(use_latex='mathjax')

# --- 5.1 建立 N 維陣列 (Array) ---

# 建立一個簡單的向量 (1D Array)
v = Array([x, y, z])

# 建立一個矩陣 (2D Array)
# 這裡模擬一個簡單的度規矩陣
M = Array([[1, 0, 0], 
           [0, x, 0], 
           [0, 0, x**2]])

print("向量 v:")
display(v)
print("矩陣 M:")
display(M)
print(f"M 的維度 (Rank): {M.rank()}")
print(f"M 的形狀 (Shape): {M.shape}")


# --- 5.2 張量積與縮併 (Tensor Product & Contraction) ---

# 1. 張量積 (Tensor Product)
# v (rank 1) (x) v (rank 1) -> T (rank 2)
# 這相當於數學上的 v_i v_j
T = tensorproduct(v, v)

print("張量積 (v (x) v):")
display(T)

# 2. 張量縮併 (Contraction)
# 計算 Trace (跡)：M^i_i
# contraction(tensor, (index1, index2))
# 這裡我們對 M 的第 0 和第 1 個維度進行縮併
trace_M = tensorcontraction(M, (0, 1))

print("矩陣 M 的跡 (Trace):")
display(trace_M)

# 3. 矩陣乘法 (Matrix Multiplication via Contraction)
# (M v)_i = M_ij v_j
# 先做外積產生 M_ij v_k (Rank 3)
# 再縮併第 1 個維度 (j) 和第 2 個維度 (k) (注意 index 從 0 開始)
Mv_outer = tensorproduct(M, v)
Mv = tensorcontraction(Mv_outer, (1, 2))

print("矩陣乘法 (M . v):")
display(Mv)


# --- 5.3 陣列導數 (Array Derivatives) ---

# 對張量進行微分是非常強大的功能
# 例如：計算梯度 (Gradient)
# grad(f) = [df/dx, df/dy, df/dz]

f = x**2 * y * z
vars = (x, y, z)

grad_f = derive_by_array(f, vars)
print("純量場 f 的梯度 (Gradient):")
display(grad_f)

# 計算向量場 v = [x, y, z] 的散度 (Divergence) 需要先求 Jacobian 再縮併
# Jacobian J_ij = dv_i / dx_j
jacobian_v = derive_by_array(v, vars)

print("向量場 v 的雅可比矩陣 (Jacobian):")
display(jacobian_v)

# 散度 = Trace(Jacobian)
divergence_v = tensorcontraction(jacobian_v, (0, 1))
print("向量場 v 的散度 (Divergence):")
display(divergence_v)
```

**本章重點：**
1.  **`Array`** 是 SymPy 處理具體分量的工具，它比 `Matrix` 更通用（可以處理 3 階、4 階甚至更高階的張量）。
2.  **`derive_by_array`** 是計算廣義相對論中「克里斯多福符號」的神器，因為克里斯多福符號涉及度規張量對座標的偏微分。

