import sympy as sp
from dgeom.sym import Metric, SCHWARZSCHILD_METRIC

# --------------------------------------------------
# IX. 克里斯多夫符號計算函數 (基於 Metric 類別)
# --------------------------------------------------

def compute_christoffel_symbols(metric_object):
    """
    計算並返回第二類克里斯多夫符號 Gamma^lambda_mu_nu。
    
    輸入: metric_object - Metric 類別的實例。
    輸出: christoffel_symbols - 一個包含所有 Gamma^lambda_mu_nu 符號的字典。
    """
    g = metric_object.g
    g_inv = metric_object.g_inv
    coords = metric_object.coords
    dim = metric_object.dim
    
    christoffel_symbols = {}
    
    # 遍歷所有的 lambda, mu, nu 組合
    for lam in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                
                # 計算第一類克里斯多夫符號 Gamma_mu_nu_sigma (只計算 sigma = lam 的情況)
                # 因為 g_inv 是對角的，所以只需要考慮 sigma = lam
                # Gamma_mu_nu_lam = 0.5 * (d g_lam_mu / d x_nu + d g_lam_nu / d x_mu - d g_mu_nu / d x_lam)
                
                # 1. 計算三個偏導數
                # ∂g_λμ/∂x^ν
                term1 = sp.diff(g[lam, mu], coords[nu])
                # ∂g_λν/∂x^μ
                term2 = sp.diff(g[lam, nu], coords[mu])
                # -∂g_μν/∂x^λ
                term3 = -sp.diff(g[mu, nu], coords[lam])
                
                Gamma_mu_nu_lam = sp.Rational(1, 2) * (term1 + term2 + term3)
                
                # 計算第二類克里斯多夫符號 Gamma^lambda_mu_nu = g^lambda_sigma * Gamma_mu_nu_sigma
                # 由於是施瓦西對角矩陣，g^lambda_sigma 只有 g^lambda_lambda 非零 (sigma=lambda)
                # Gamma^lambda_mu_nu = g^lambda_lambda * Gamma_mu_nu_lambda
                
                Gamma_lam_mu_nu = g_inv[lam, lam] * Gamma_mu_nu_lam
                
                # 簡化結果並檢查是否為零
                simplified_Gamma = sp.simplify(Gamma_lam_mu_nu)
                
                # 僅儲存非零或非平凡的符號
                if simplified_Gamma != 0:
                    symbol_key = (lam, mu, nu)
                    christoffel_symbols[symbol_key] = simplified_Gamma
                    
    return christoffel_symbols

def test_schwarzschild_christoffel_symbols():
    """
    測試施瓦西度規的克里斯多夫符號計算。
    """
    print("\n" + "="*60)
    print("測試案例: test_schwarzschild_christoffel_symbols (施瓦西度規的克里斯多夫符號)")
    print("="*60)
    
    schwarzschild_christoffels = compute_christoffel_symbols(SCHWARZSCHILD_METRIC)
    
    print("--- 施瓦西度規的非零克里斯多夫符號 $\Gamma^{\lambda}_{\mu\nu}$ ---")
    print("坐標索引: 0=t, 1=r, 2=θ, 3=φ\n")
    
    for (lam, mu, nu), value in schwarzschild_christoffels.items():
        # 利用 sympy 的 latex 輸出格式化
        symbol_str = r'\Gamma^{%d}_{%d%d}' % (lam, mu, nu)
        latex_output = sp.latex(value)
        print(f'${symbol_str} = {latex_output}$')
        # 由於 Gamma^lambda_mu_nu 對稱於 mu 和 nu (即 Gamma^lambda_mu_nu = Gamma^lambda_nu_mu)，
        # 且程式碼在 mu < nu 時會計算 mu, nu 和 nu, mu 兩次，但結果相同。
        # 這裡我們只輸出一次，避免重複，但對於不對稱的 (mu, nu) 組合，我們也列出
        if mu != nu:
             symbol_str_sym = r'\Gamma^{%d}_{%d%d}' % (lam, nu, mu)
             print(f'${symbol_str_sym} = {latex_output}$')

schwarzschild_christoffels = compute_christoffel_symbols(SCHWARZSCHILD_METRIC)

print("--- 施瓦西度規的非零克里斯多夫符號 $\Gamma^{\lambda}_{\mu\nu}$ ---")
print("坐標索引: 0=t, 1=r, 2=θ, 3=φ\n")

for (lam, mu, nu), value in schwarzschild_christoffels.items():
    # 利用 sympy 的 latex 輸出格式化
    symbol_str = r'\Gamma^{%d}_{%d%d}' % (lam, mu, nu)
    latex_output = sp.latex(value)
    print(f'${symbol_str} = {latex_output}$')
    # 由於 Gamma^lambda_mu_nu 對稱於 mu 和 nu (即 Gamma^lambda_mu_nu = Gamma^lambda_nu_mu)，
    # 且程式碼在 mu < nu 時會計算 mu, nu 和 nu, mu 兩次，但結果相同。
    # 這裡我們只輸出一次，避免重複，但對於不對稱的 (mu, nu) 組合，我們也列出
    if mu != nu:
         symbol_str_sym = r'\Gamma^{%d}_{%d%d}' % (lam, nu, mu)
         print(f'${symbol_str_sym} = {latex_output}$')