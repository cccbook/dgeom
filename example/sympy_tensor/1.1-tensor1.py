import sympy
from sympy import symbols, init_printing

# 引入抽象張量相關的核心類別
from sympy.tensor.tensor import TensorIndexType, TensorIndex, TensorHead, tensor_indices

# 啟用漂亮的數學顯示 (Pretty Printing)
# use_latex='mathjax' 確保在 Jupyter 中渲染出完美的數學公式
init_printing(use_latex='mathjax')

print(f"SymPy Version: {sympy.__version__}")

# 1. 定義一個指標類型 (Index Type)
# 例如：Lorentz 空間 (L)，通常用於相對論
Lorentz = TensorIndexType('Lorentz', dummy_name='L')

# 2. 定義具體的指標符號 (Indices)
# mu, nu 是這個空間中的指標
mu, nu = tensor_indices('mu nu', Lorentz)

# 3. 定義一個張量實體 (Tensor Head)
# A 是一個張量，且我們指定它通常帶有一個指標
A = TensorHead('A', [Lorentz])

# 4. 建立張量表達式
# 創建一個表達式：A 的 mu 分量
expr = A(mu)

# 5. 顯示結果
print("Text Output:", expr)
# 在 Jupyter Notebook 中，直接輸入變數名稱會顯示 LaTeX 格式
expr

