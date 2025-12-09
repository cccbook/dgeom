from .riemann import *
from .constants import G, c, M_sun
import sympy as sp

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
