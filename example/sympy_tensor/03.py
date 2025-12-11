from sympy import symbols, init_printing, diag
from sympy.tensor.tensor import TensorIndexType, TensorIndex, TensorHead, tensor_indices

# 啟用 LaTeX 數學顯示
init_printing(use_latex='mathjax')

# 1. 定義一個洛倫茲空間 (Lorentzian Manifold)
# structure=TensorIndexType.Lorentz 表示這是一個相對論時空 (擁有 Minkowski 度規)
# dummy_name='L' 用於自動生成求和時的虛設指標名稱
L = TensorIndexType('L', dummy_name='L')

# 2. 定義歐幾里得空間 (Euclidean Space)
# metric_symmetry=1 表示度規是對稱的 (g_ij = g_ji)
# dim=3 指定維度為 3
E = TensorIndexType('E', dummy_name='E', dim=3, metric_symmetry=1)

# 3. 定義指標符號 (Indices)
# mu, nu, rho 屬於洛倫茲空間 L
mu, nu, rho, sigma = tensor_indices('mu nu rho sigma', L)

# i, j, k 屬於歐幾里得空間 E
i, j, k = tensor_indices('i j k', E)

print("空間定義完成。")
print(f"Lorentz Indices: {mu}, {nu}")
print(f"Euclidean Indices: {i}, {j}")

# 定義張量頭 (Tensor Heads)
# 括號內的 [L] 表示這個張量的第一個指標屬於 L 空間
# 如果是二階張量，可以寫 [L, L]

# 1. 定義一個 4-向量 P (例如動量)
P = TensorHead('P', [L])

# 2. 定義一個二階張量 F (例如電磁張量)
F = TensorHead('F', [L, L])

# 3. 定義一個純量 phi (沒有指標)
phi = TensorHead('phi', [])

# 建立張量表達式
# 在 SymPy 中：
# mu  (正) 代表 上指標 (Contravariant, ^mu)
# -mu (負) 代表 下指標 (Covariant, _mu)

expr_vec_up = P(mu)   # P^mu
expr_vec_down = P(-mu)  # P_mu
expr_tensor = F(mu, -nu) # F^mu_nu

display(expr_vec_up)
display(expr_vec_down)
display(expr_tensor)

# 1. 張量加法
# 只有指標結構相同的張量才能相加
term1 = F(mu, nu)
term2 = F(nu, mu) # 注意指標順序不同
add_expr = term1 + term2

print("張量加法:")
display(add_expr)

# 2. 張量乘法 (外積)
# P^mu * P^nu -> T^{mu nu}
mult_expr = P(mu) * P(nu)

print("張量外積:")
display(mult_expr)

# 3. 愛因斯坦求和 (縮併 Contraction)
# P^mu * P_mu -> Scalar (P^2)
# 注意：這裡我們用 -mu 來表示下指標
contraction = P(mu) * P(-mu)

print("愛因斯坦求和 (自動縮併):")
display(contraction)

# 4. 檢查自由指標與虛設指標
# 自由指標：未被求和的指標
# 虛設指標：被求和掉的指標
complex_expr = F(mu, -nu) * P(nu) # F^mu_nu * P^nu -> 結果應該剩下 mu (上標)

print("複雜縮併 (矩陣乘法向量):")
display(complex_expr)

# 查看指標資訊
print(f"Free indices: {complex_expr.get_free_indices()}")
# 注意：虛設指標在內部會被重新命名，以避免衝突

# 取得度規張量
g = L.metric

print("度規張量形式:")
display(g(mu, nu))      # g^{mu nu}
display(g(-mu, -nu))    # g_{mu nu}

# 1. Kronecker Delta
# 混合指標的度規就是 Kronecker Delta
delta = g(mu, -nu)
print("Kronecker Delta:")
display(delta)

# 2. 指標升降 (Raising and Lowering)
# 我們手動將 P^mu 降標： g_{mu nu} P^nu
lowering_expr = g(-mu, -nu) * P(nu)

print("指標降標 (原始式子):")
display(lowering_expr)

# 使用 canonicalize() 來簡化結果
# SymPy 應該會自動將其識別為 P_mu
print("指標降標 (標準化後):")
# from sympy.tensor.tensor import simplify, canonicalize
from sympy import simplify
simplified = simplify(lowering_expr) #canonicalize(lowering_expr)
display(simplified)

# 3. 複雜的度規收縮
# g^{mu nu} F_{mu rho} -> F^nu_rho
complex_metric = g(mu, nu) * F(-mu, -rho)
display(complex_metric)
display(canonicalize(complex_metric))

