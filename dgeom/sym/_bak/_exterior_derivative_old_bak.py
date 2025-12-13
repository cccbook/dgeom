import sympy as sp
import itertools
from functools import reduce

# 外微分相關模組：TangentVector / lie_bracket / d_operator / Form
# AI 解說：https://gemini.google.com/share/ad4fdede0c8f

class TangentVector:
    """
    符號切向量場
    components: SymPy 表達式列表/矩陣，例如 [y, -x, 0]
    coords: 定義流形的座標符號，例如 [x, y, z]
    """
    def __init__(self, components, coords, name="V"):
        self.components = sp.Matrix(components) # 轉為直列向量
        self.coords = sp.Matrix(coords)
        self.dim = len(coords)
        self.name = name
        
        if len(self.components) != self.dim:
            raise ValueError("向量維度與座標維度不符")

    def __call__(self, f):
        """
        作用於純量函數 f (SymPy 表達式) -> 方向導數 V(f)
        V(f) = sum v^i * (df/dx^i)
        """
        # 計算梯度 (符號微分)
        grad_f = sp.Matrix([sp.diff(f, var) for var in self.coords])
        # 內積
        res = self.components.dot(grad_f)
        return sp.simplify(res)

    def at(self, point_dict):
        """
        在特定點評估向量數值
        point_dict: {x: 1, y: 2}
        """
        return self.components.subs(point_dict)

def lie_bracket(u, v):
    """
    計算李括號 [u, v] = v.jacobian * u - u.jacobian * v
    這是解析解，無需數值近似
    """
    if u.coords != v.coords:
        raise ValueError("向量場必須定義在相同的座標系")
    
    coords = u.coords
    # 計算 Jacobian 矩陣: J_ij = d(v_i)/d(x_j)
    J_u = u.components.jacobian(coords)
    J_v = v.components.jacobian(coords)
    
    # [u, v] = (v \cdot \nabla) u - (u \cdot \nabla) v
    # 注意：在矩陣乘法表示中，通常寫作 J_v * u - J_u * v
    # J_v 是 v 的導數矩陣，乘上 u 向量代表在 u 方向的變化率
    w_components = J_v * u.components - J_u * v.components
    
    return TangentVector(w_components, coords, name=f"[{u.name},{v.name}]")

class Form:
    """
    k-Form
    k: 階數
    op: 函數，接受 k 個 TangentVector，回傳一個 SymPy 表達式
    """
    def __init__(self, degree, evaluator):
        self.k = degree
        self.op = evaluator 
        
    def __call__(self, *vectors):
        # [修正] 針對 0-form (純量場)
        if self.k == 0: 
            # 如果 op 是函數 (例如 lambda)，執行它以取得表達式
            if callable(self.op):
                return self.op()
            # 如果 op 本身已經是表達式，直接回傳
            return self.op
            
        if len(vectors) != self.k: 
            raise ValueError(f"Need {self.k} vectors, got {len(vectors)}")
        # 這裡回傳的是 SymPy 表達式
        return sp.simplify(self.op(*vectors))

def d_operator(omega):
    """
    外微分算子 (Exterior Derivative)
    使用 Palais 不變量公式 (Invariant Formula)
    """
    k = omega.k
    
    def d_omega_evaluator(*vectors): # vectors: X0...Xk
        total = 0
        n = len(vectors)
        
        # Part A: X_i(omega(...))
        # 這項代表向量場 X_i 作用在 (k-1) form 評估後的純量場上
        for i in range(n):
            X_i = vectors[i]
            others = vectors[:i] + vectors[i+1:]
            
            # omega(*others) 是一個 SymPy 表達式
            scalar_field = omega(*others)
            
            # X_i(scalar_field) 是方向導數
            val = X_i(scalar_field)
            
            term = val if i % 2 == 0 else -val
            total += term
            
        # Part B: omega([X_i, X_j], ...)
        # 使用李括號修正項
        if n >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    bracket = lie_bracket(vectors[i], vectors[j])
                    others = vectors[:i] + vectors[i+1:j] + vectors[j+1:]
                    
                    val = omega(bracket, *others)
                    
                    sign = (-1)**(i + j)
                    total += sign * val
                    
        return sp.simplify(total)
    
    return Form(k + 1, d_omega_evaluator)

