from sympy import symbols, IndexedBase, Idx, init_printing

# 設定漂亮顯示
init_printing()

# 1. 定義維度符號 (可以是具體數字，也可以是符號)
n, m = symbols('n m', integer=True)

# 2. 定義基底符號 (Base)
# 這代表數學上的 A 和 B，它們將被視為陣列
A = IndexedBase('A')
B = IndexedBase('B')

# 3. 定義指標 (Indices)
# 定義指標 i，範圍從 0 到 n-1
# 定義指標 j，範圍從 0 到 m-1
i = Idx('i', n) 
j = Idx('j', m)

print("Base objects:", A, B)
print("Index objects:", i, j)

# 建立一個表達式：A_i + B_i
expr1 = A[i] + B[i]

# 建立二維元素：A_{ij}
expr2 = A[i, j]

print("1D Expression:")
display(expr1)  # Jupyter 中會顯示 A[i] + B[i]

print("2D Element:")
display(expr2)  # Jupyter 中會顯示 A[i, j]

# 檢查物件類型
print(f"Type of A[i]: {type(A[i])}")
# 輸出: <class 'sympy.tensor.indexed.Indexed'>

from sympy import Sum

# 例子 1：內積 (Inner Product)
# 數學式: S = sum(A_i * B_i, i=0..n-1)
inner_product = Sum(A[i] * B[i], (i, 0, n - 1))

print("Symbolic Summation:")
display(inner_product)

# 例子 2：矩陣乘法分量 (Matrix Multiplication Component)
# C_ij = sum(A_ik * B_kj)
# 我們需要一個新的指標 k
k = Idx('k', n) # 假設 A 是 n*n, B 是 n*n

# 定義矩陣乘法的第 (i, j) 個元素
matrix_mult_element = Sum(A[i, k] * B[k, j], (k, 0, n - 1))

print("Matrix Multiplication Element C_{ij}:")
display(matrix_mult_element)

# 設定一個具體維度為 3 的指標
dim = 3
k_concrete = Idx('k', dim)

# 定義簡單的平方和
sum_squares = Sum(A[k_concrete]**2, (k_concrete, 0, dim - 1))

print("Concrete Summation (N=3):")
display(sum_squares)

print("Expanded Result:")
expanded = sum_squares.doit()
display(expanded)
# 預期輸出: A[0]**2 + A[1]**2 + A[2]**2

