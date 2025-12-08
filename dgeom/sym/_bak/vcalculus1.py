from sympy import symbols, diff, Matrix

# 宣告符號變數
x, y, z = symbols('x y z')

def gradient(f, variables=[x, y, z]):
    """
    計算純量場 f 的梯度 (Gradient, ∇f)。
    輸入: f (SymPy 運算式), variables (偏微分的變數列表)
    輸出: 梯度向量 (SymPy Matrix)
    """
    # 梯度是 f 對每個變數的偏導數所組成的向量
    grad_components = [diff(f, var) for var in variables]
    return Matrix(grad_components)

def divergence(F, variables=[x, y, z]):
    """
    計算向量場 F 的散度 (Divergence, ∇ · F)。
    輸入: F (SymPy Matrix，向量場的各分量), variables (偏微分的變數列表)
    輸出: 散度純量 (SymPy 運算式)
    """
    # 散度是每個分量對應變數的偏導數之和
    if len(F) != len(variables):
        raise ValueError("向量場分量數與變數數不匹配")
        
    div_sum = 0
    for i in range(len(variables)):
        # F[i] 是 F 的第 i 個分量 (Fx, Fy, Fz)
        # variables[i] 是對應的變數 (x, y, z)
        div_sum += diff(F[i], variables[i])
        
    return div_sum

def curl(F, variables=[x, y, z]):
    """
    計算三維向量場 F 的旋度 (Curl, ∇ × F)。
    輸入: F (SymPy Matrix，向量場的各分量), variables (變數列表，必須是 [x, y, z])
    輸出: 旋度向量 (SymPy Matrix)
    """
    if variables != [x, y, z] or len(F) != 3:
        raise ValueError("旋度運算僅實用於三維直角坐標系 ([x, y, z]) 的三個分量向量場。")
    
    Fx, Fy, Fz = F[0], F[1], F[2]
    
    # 計算旋度的三個分量 (∇ × F)
    # i 分量: dFz/dy - dFy/dz
    i_comp = diff(Fz, y) - diff(Fy, z)
    
    # j 分量: dFx/dz - dFz/dx (注意，這裡的公式是 (d/dz)(Fx) - (d/dx)(Fz))
    # 這是行列式展開的第二項，需要變號，或依照 (dFx/dz - dFz/dx)
    j_comp = diff(Fx, z) - diff(Fz, x)
    
    # k 分量: dFy/dx - dFx/dy
    k_comp = diff(Fy, x) - diff(Fx, y)
    
    return Matrix([i_comp, j_comp, k_comp])

if __name__ == "__main__":
    # --- 範例測試 ---

    # 1. 測試梯度 (Gradient)
    print("--- 1. 測試梯度 (Gradient) ---")
    f = x**2 * y + z**3
    grad_f = gradient(f)
    print(f"純量場 f: {f}")
    print(f"梯度 ∇f: {grad_f}")
    # 預期結果: [2*x*y, x**2, 3*z**2]

    print("\n")

    # 2. 測試散度 (Divergence)
    print("--- 2. 測試散度 (Divergence) ---")
    # 向量場 F = Fx i + Fy j + Fz k
    # Fx = x*z, Fy = x*y, Fz = -y*z
    F_div = Matrix([x*z, x*y, -y*z])
    div_F = divergence(F_div)
    print(f"向量場 F: {F_div}")
    print(f"散度 ∇·F: {div_F}")
    # 預期結果: d(xz)/dx + d(xy)/dy + d(-yz)/dz = z + x - y

    print("\n")

    # 3. 測試旋度 (Curl)
    print("--- 3. 測試旋度 (Curl) ---")
    # 向量場 G = Gx i + Gy j + Gz k
    # Gx = y, Gy = -x, Gz = 0 (表示一個繞 z 軸旋轉的場)
    G_curl = Matrix([y, -x, 0])
    curl_G = curl(G_curl)
    print(f"向量場 G: {G_curl}")
    print(f"旋度 ∇×G: {curl_G}")
    # 預期結果:
    # i 分量: d(0)/dy - d(-x)/dz = 0 - 0 = 0
    # j 分量: d(y)/dz - d(0)/dx = 0 - 0 = 0
    # k 分量: d(-x)/dx - d(y)/dy = -1 - 1 = -2
    # 最終結果: [0, 0, -2]