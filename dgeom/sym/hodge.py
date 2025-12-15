import sympy as sp
import itertools
from sympy import MutableDenseNDimArray, Matrix
from .metric import MetricTensor
from .d_operator import TangentVector, Form, d_operator

# ==========================================
# 霍奇星算子 Hodge Star (Refactored)
# ==========================================

class HodgeMetric(MetricTensor):
    """
    支援 Hodge Star (*) 運算與指標升降的度規物件。
    繼承自 MetricTensor。
    """
    def __init__(self, data, coords):
        # 初始化父類別 (MetricTensor)
        super().__init__(data, coords)
        
        # 預計算 Hodge Star 所需的純量與逆度規數據
        # 雖然 MetricTensor 有 inverse()，但為了 star() 迴圈效能，我們緩存純數據
        # 注意: MetricTensor.data 是 NDimArray
        
        # 計算行列式 (轉為 Matrix 計算較方便)
        g_mat = Matrix(self.data.tolist())
        self.det_g = g_mat.det()
        self.sqrt_det_g = sp.sqrt(sp.Abs(self.det_g))
        
        # 緩存逆度規數據 (NDimArray)
        self.inv_g_data = self.inverse().data

    def get_basis_vector(self, i):
        """取得第 i 個座標基底向量 e_i"""
        comp = [0] * self.dim
        comp[i] = 1
        return TangentVector(comp, self.coords, name=f"e_{i}")

    def flat(self, vector):
        """
        降指標運算 (Musical Isomorphism: Flat b)
        Vector -> 1-Form
        v_i = g_ij v^j
        """
        if not isinstance(vector, TangentVector):
             # 若傳入的是純數據列表，嘗試轉換
             vector = TangentVector(vector, self.coords)

        # 計算協變分量 coeffs_i = sum_j (g_ij * v^j)
        coeffs = []
        for i in range(self.dim):
            val = 0
            for j in range(self.dim):
                val += self.data[i, j] * vector.data[j]
            coeffs.append(val)
        
        # 定義 1-Form 的評估函數
        # omega(u) = sum(omega_i * u^i)
        def evaluator(v_target):
            res = 0
            for i in range(self.dim):
                res += coeffs[i] * v_target.data[i]
            return sp.simplify(res)
            
        return Form(1, evaluator)

    def sharp(self, one_form):
        """
        升指標運算 (Musical Isomorphism: Sharp #)
        1-Form -> Vector
        v^i = g^ij v_j
        """
        if one_form.k != 1:
            raise ValueError("Sharp only defined for 1-forms")
        
        # 1. 提取 Form 的分量 omega_j = one_form(e_j)
        omega_components = []
        for j in range(self.dim):
            basis_vec = self.get_basis_vector(j)
            omega_components.append(one_form(basis_vec))
            
        # 2. 升指標 vec^i = sum_j (g^ij * omega_j)
        vec_components = []
        for i in range(self.dim):
            val = 0
            for j in range(self.dim):
                val += self.inv_g_data[i, j] * omega_components[j]
            vec_components.append(sp.simplify(val))
            
        return TangentVector(vec_components, self.coords, name="Sharp(w)")

    def star(self, form):
        """
        通用 Hodge Star Operator (*): k-Form -> (n-k)-Form
        (MetricTensor Version)
        """
        k = form.k
        n = self.dim
        target_k = n - k
        
        # 1. 提取輸入 Form 的分量
        # 遍歷所有可能的 k 維基底組合 K (排序過的 index tuple)
        source_bases = list(itertools.combinations(range(n), k))
        omega_comps = {} 
        for K in source_bases:
            vectors = [self.get_basis_vector(i) for i in K]
            omega_comps[K] = form(*vectors)

        # 2. 預計算目標 Form 對應於每個基底 dx^J 的係數 C_J
        target_bases = list(itertools.combinations(range(n), target_k))
        target_coeffs = {}

        # 核心公式: (*dx^K) = sqrt|g| * g^{K L} * epsilon_{L J} * dx^J
        # 實作邏輯: 線性疊加原理
        
        for J in target_bases: # 對應目標 (n-k)-Form 的基底
            total_coeff_J = 0
            
            for K in source_bases: # 對應來源 k-Form 的基底
                w_val = omega_comps[K]
                if w_val == 0: continue
                
                # 計算投影係數
                # 需遍歷所有排列 I (長度 k) 來進行縮併 g^{ik}
                all_I = itertools.product(range(n), repeat=k)
                
                term_sum = 0
                for I_tuple in all_I:
                    # Levi-Civita 符號: indices = I + J
                    # 檢查這組 indices 是否構成 0..n-1 的排列
                    full_indices = I_tuple + J
                    
                    # sp.LeviCivita 接受不定長度參數
                    try:
                        lev_val = sp.LeviCivita(*full_indices)
                    except:
                        lev_val = 0 # 若 index 重複或不合法
                    
                    if lev_val == 0: continue
                    
                    # 度規縮併: product( g^{ I[m], K[m] } )
                    metric_contraction = 1
                    for idx in range(k):
                        # inv_g_data 是 NDimArray，使用 [row, col] 存取
                        metric_contraction *= self.inv_g_data[I_tuple[idx], K[idx]]
                    
                    term_sum += lev_val * metric_contraction
                
                total_coeff_J += w_val * term_sum

            target_coeffs[J] = sp.simplify(total_coeff_J * self.sqrt_det_g)

        # 3. 定義輸出 Form 的評估函數
        def star_evaluator(*vectors):
            if len(vectors) != target_k:
                raise ValueError(f"Output form expects {target_k} vectors")
            
            # 將輸入向量組合成矩陣 (col vectors) 以計算行列式
            # vectors 是 TangentVector 列表
            # 我們需要構建一個 SymPy Matrix: cols = vector components
            cols = [v.data.tolist() for v in vectors]
            # Transpose to make vectors columns: Matrix(cols).T
            V_matrix = Matrix(cols).T
            
            result = 0
            for J, coeff in target_coeffs.items():
                if coeff == 0: continue
                
                # J 是 tuple (row_indices)
                # 取出對應的 rows 組成子矩陣 (代表 dx^J 作用在 V 上)
                sub_matrix = V_matrix[list(J), :] 
                det_val = sub_matrix.det()
                
                result += coeff * det_val
                
            return sp.simplify(result)

        return Form(target_k, star_evaluator)

# ==========================================
# Part 3: 向量微積分 (Wrapping d_operator)
# ==========================================

def h_gradient(f, metric): 
    """Grad f = (df)^#"""
    # d_operator(Form(0, f)) 回傳 df (1-Form)
    # sharp 將其轉為向量
    return metric.sharp(d_operator(Form(0, f)))

def h_curl(vector, metric): 
    """Curl V = (* d V^b)^# (僅限 3D)"""
    if metric.dim != 3: raise ValueError("Curl only for 3D")
    
    v_flat = metric.flat(vector)   # Vector -> 1-Form
    dv = d_operator(v_flat)        # d(1-Form) -> 2-Form
    star_dv = metric.star(dv)      # *(2-Form) -> 1-Form
    
    return metric.sharp(star_dv)   # 1-Form -> Vector

def h_divergence(vector, metric): 
    """Div V = * d (* V^b)"""
    v_flat = metric.flat(vector)    # Vector -> 1-Form
    star_v = metric.star(v_flat)    # *(1-Form) -> (n-1)-Form
    d_star_v = d_operator(star_v)   # d((n-1)-Form) -> n-Form
    res_form = metric.star(d_star_v)# *(n-Form) -> 0-Form
    
    return res_form() # 0-form is operator returning scalar
