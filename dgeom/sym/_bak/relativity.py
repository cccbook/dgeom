from .constants import G, c, M_sun
import sympy as sp

def metric_tensor(old_coords_funcs, new_params):
    """
    計算由參數函數定義的流形在新參數座標系下的度規張量。

    參數:
    - old_coords_funcs (list of SymPy expressions): 
        一個列表，包含舊座標 (例如 x, y, z) 關於新參數的函數表示。
        例如：[r * sp.cos(theta), r * sp.sin(theta)]
    - new_params (list of SymPy symbols): 
        一個列表，包含新座標或參數 (例如 r, theta) 的 SymPy 符號。
        例如：[r, theta]

    回傳:
    - G_bar (SymPy Matrix): 
        新參數座標系下的度規張量矩陣 G_bar。
    """
    
    # 1. 將舊座標函數表示為 SymPy 矩陣形式 (代表嵌入函數 X)
    X_func_matrix = sp.Matrix(old_coords_funcs)
    
    # 2. 計算雅可比矩陣 J (即 Lambda_inv)
    # J 是 X 相對於新參數 (u) 的偏微分矩陣: J_ik = partial x^i / partial u^k
    # 這裡的 X_func_matrix.jacobian(new_params) 正好計算 J
    J_matrix = X_func_matrix.jacobian(new_params)
    
    # 3. 計算度規張量 G_bar
    # 公式: G_bar = J^T * G_old * J
    # 由於我們假設舊座標系是歐幾里得空間 (G_old = I, 單位矩陣)
    # 所以簡化為: G_bar = J^T * J
    
    J_transpose = J_matrix.transpose()
    G_bar = J_transpose * J_matrix
    
    # 4. 簡化結果
    G_bar_simplified = G_bar.applyfunc(sp.simplify)
    
    return G_bar_simplified

# --- 黎曼曲率張量計算函式 ---
def christoffel(mu, alpha, beta, G_cont, G_cov, coords):
    result = 0
    dim = len(coords)
    for gamma in range(dim):
        term1 = sp.diff(G_cov[gamma, alpha], coords[beta])
        term2 = sp.diff(G_cov[gamma, beta], coords[alpha])
        term3 = sp.diff(G_cov[alpha, beta], coords[gamma])
        # 由於 Christoffel 符號的計算中，上標 mu 決定了係數 g^mu_gamma 的維度
        # 此處的 mu 和 gamma 確實應該在流形維度內
        term = G_cont[mu, gamma] * sp.Rational(1, 2) * (term1 + term2 - term3) # 修正: 加上 sp.Rational(1, 2)
        result += term
    return sp.simplify(result)

# --- 黎曼曲率張量計算函式 ---
def riemann_tensor(rho, sigma, mu, nu, G_cont, G_cov, coords):
    """計算 R^rho_{sigma mu nu}"""
    
    # 應使用實際的座標維度
    dim = len(coords)

    # 計算第一、二項（導數項）
    # R^rho_sigma_mu_nu = d_mu Gamma^rho_nu_sigma - d_nu Gamma^rho_mu_sigma + ...
    Gamma_nu_sigma = christoffel(rho, nu, sigma, G_cont, G_cov, coords)
    term1 = sp.diff(Gamma_nu_sigma, coords[mu])
    
    Gamma_mu_sigma = christoffel(rho, mu, sigma, G_cont, G_cov, coords)
    term2 = sp.diff(Gamma_mu_sigma, coords[nu])
    
    # 計算第三、四項（二次項）
    term3_4 = 0
    for alpha in range(dim): 
        # 為了清晰，這裡按照標準公式 R = dGamma - dGamma + Gamma*Gamma - Gamma*Gamma
        # 正項 (+): Gamma^rho_mu_alpha * Gamma^alpha_nu_sigma
        # 負項 (-): Gamma^rho_nu_alpha * Gamma^alpha_mu_sigma
        
        Gamma_rho_mu_alpha = christoffel(rho, mu, alpha, G_cont, G_cov, coords)
        Gamma_alpha_nu_sigma = christoffel(alpha, nu, sigma, G_cont, G_cov, coords)
        
        Gamma_rho_nu_alpha = christoffel(rho, nu, alpha, G_cont, G_cov, coords)
        Gamma_alpha_mu_sigma = christoffel(alpha, mu, sigma, G_cont, G_cov, coords)
        
        # [修正] 這是正項 (與 mu 相關的 Gamma 在前)
        positive_term = Gamma_rho_mu_alpha * Gamma_alpha_nu_sigma
        
        # [修正] 這是負項 (與 nu 相關的 Gamma 在前)
        negative_term = Gamma_rho_nu_alpha * Gamma_alpha_mu_sigma
        
        term3_4 += positive_term - negative_term

    return sp.simplify(term1 - term2 + term3_4)

def ricci_tensor(G_cont, G_cov, coords):
    """
    計算 Ricci 曲率張量 R_mu_nu
    定義: R_mu_nu = R^lambda_mu_lambda_nu (黎曼張量的跡)
    """
    dim = len(coords)
    R_mn = sp.zeros(dim, dim)
    
    for mu in range(dim):
        for nu in range(dim):
            # 對第一個和第三個索引進行縮約 (Contraction)
            sum_val = 0
            for lam in range(dim):
                sum_val += riemann_tensor(lam, mu, lam, nu, G_cont, G_cov, coords)
            R_mn[mu, nu] = sp.simplify(sum_val)
            
    return R_mn

def ricci_scalar(R_mn, G_cont):
    """
    計算 Ricci 純量 R (Scalar Curvature)
    定義：R = g^mu_nu * R_mu_nu
    意義：從 p 點發出的所有方向的平均截面曲率
    """
    # 矩陣乘法 G_cont * R_mn 得到 R^mu_nu，然後取 Trace (跡)
    # 或者直接使用雙重迴圈縮約
    dim = G_cont.shape[0]
    scalar_R = 0
    for mu in range(dim):
        for nu in range(dim):
            scalar_R += G_cont[mu, nu] * R_mn[mu, nu]
            
    return sp.simplify(scalar_R)

# 愛因斯坦張量 G_{mu nu} 的計算
def einstein_tensor(R_mn, R_scalar, G_cov):
    """
    計算愛因斯坦張量 G_{mu nu} (矩陣形式)
    公式: G = R_mn - 1/2 * R * g_mn
    
    參數:
    - R_mn: Ricci 張量 (SymPy Matrix)
    - R_scalar: Ricci 純量 (SymPy Expression)
    - G_cov: 協變度規張量 (SymPy Matrix)
    """
    
    # SymPy 支援矩陣與純量的直接運算
    # 0.5 * R_scalar * G_cov 會自動對 G_cov 的每個元素進行乘法
    G_matrix = R_mn - 0.5 * R_scalar * G_cov
    
    # 對矩陣每個元素進行化簡
    return sp.simplify(G_matrix)

def einstein_field_equation(G_matrix, G_cov, T_matrix, Lambda=0, kappa='8*pi*G/c**4'):
    """
    計算愛因斯坦場方程 (EFE) 的差異 (LHS - RHS)，矩陣形式。
    當結果為零矩陣時，表示度規張量 G_cov 滿足 EFE。

    公式: G_{mu nu} + Lambda * g_{mu nu} = kappa * T_{mu nu}
    或:   G_{mu nu} + Lambda * g_{mu nu} - kappa * T_{mu nu} = 0

    參數:
    - G_matrix: 愛因斯坦張量 G_{mu nu} (SymPy Matrix)
    - G_cov: 協變度規張量 g_{mu nu} (SymPy Matrix)
    - T_matrix: 能量-動量張量 T_{mu nu} (SymPy Matrix)
    - Lambda: 宇宙學常數 (SymPy Expression or float, 預設為 0)
    - kappa: 愛因斯坦引力常數 (SymPy Expression or string, 預設為 '8*pi*G/c**4')

    回傳:
    - EFE_diff (SymPy Matrix): EFE 的差異矩陣
    """
    
    # 將 kappa 轉為 SymPy Expression，以便在沒有數值替換時也能進行符號運算
    if isinstance(kappa, str):
        # 定義物理常數符號，以便於符號運算
        G_sym, c_sym = sp.symbols('G c')
        kappa_expr = sp.sympify(kappa, locals={'G': G_sym, 'c': c_sym, 'pi': sp.pi})
    else:
        kappa_expr = sp.sympify(kappa)
        
    Lambda_expr = sp.sympify(Lambda)

    # LHS: G_{mu nu} + Lambda * g_{mu nu}
    LHS = G_matrix + Lambda_expr * G_cov
    
    # RHS: kappa * T_{mu nu}
    RHS = kappa_expr * T_matrix
    
    # 差異: LHS - RHS
    EFE_diff = LHS - RHS
    
    # 對矩陣每個元素進行化簡
    return sp.simplify(EFE_diff)

# 黑洞：史瓦西半徑計算函式
def calculate_schwarzschild_radius(M):
    rs_expr = 2 * G * M / c**2
    return rs_expr
