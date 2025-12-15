import sympy as sp
from sympy import MutableDenseNDimArray
from .tensor import GeometricTensor

# ===================================================================
# 1. 切向量 (Tangent Vector) - 新版
# ===================================================================

class TangentVector(GeometricTensor):
    """
    符號切向量場 V = v^i * (d/dx^i)
    繼承自 GeometricTensor，固定為 Rank 1 逆變張量 ([1])。
    """
    def __init__(self, components, coords, name="V"):
        # 確保輸入是 NDimArray
        if not isinstance(components, MutableDenseNDimArray):
            # 處理 Matrix 或 list
            if hasattr(components, 'tolist'): # Matrix
                data = MutableDenseNDimArray(components.tolist())
                # Matrix 轉過來通常是二維 (n, 1)，需要壓平
                if data.rank() == 2:
                    data = MutableDenseNDimArray([x for x in data])
            else: # list
                data = MutableDenseNDimArray(components)
        else:
            data = components

        # 初始化父類別: Rank 1, Contravariant (+1)
        super().__init__(data, coords, [1])
        self.name = name

    def __call__(self, f):
        """
        作用於純量函數 f (SymPy 表達式) -> 方向導數 V(f)
        V(f) = v^i * (df/dx^i)
        """
        # 如果 f 也是 GeometricTensor (Rank 0 scalar)，取其數據
        if hasattr(f, 'data') and hasattr(f, 'rank'):
             if f.rank == 0:
                 f = f.data[()]
        
        # 方向導數計算: sum( v[i] * diff(f, x[i]) )
        res = 0
        for i, x in enumerate(self.coords):
            res += self.data[i] * sp.diff(f, x)
            
        return sp.simplify(res)

    def at(self, point_dict):
        """
        在特定點評估向量數值
        """
        # 利用 GeometricTensor 內建的 substitute 機制 (需自行實作或手動 subs)
        # 這裡手動處理 NDimArray 的 subs
        try:
            new_data = self.data.applyfunc(lambda x: x.subs(point_dict))
        except AttributeError: # 針對 Rank 0 或純量元素 (較少見)
            new_data = self.data.subs(point_dict)
            
        return TangentVector(new_data, self.coords, self.name)

# ===================================================================
# 2. 李括號 (Lie Bracket)
# ===================================================================

def lie_bracket(u, v):
    """
    計算李括號 [u, v]。
    定義: [u, v](f) = u(v(f)) - v(u(f))
    分量公式: [u, v]^k = u^j (dv^k/dx^j) - v^j (du^k/dx^j)
    這等價於: [u, v]^k = u(v^k) - v(u^k)
    """
    if u.coords != v.coords:
        raise ValueError("向量場必須定義在相同的座標系")
    
    # 利用 TangentVector.__call__ (方向導數) 來計算分量變化
    # w^k = u(v^k) - v(u^k)
    w_comps = []
    dim = len(u.coords)
    
    for k in range(dim):
        # u 作用在 v 的第 k 個分量上 (視為純量場)
        term1 = u(v.data[k])
        # v 作用在 u 的第 k 個分量上
        term2 = v(u.data[k])
        
        w_comps.append(term1 - term2)
        
    return TangentVector(w_comps, u.coords, name=f"[{u.name},{v.name}]")

# ===================================================================
# 3. 微分形式 (Differential Form) - 運算子觀點
# ===================================================================
# Form 類別本身是「函數的容器」，它不需要繼承 GeometricTensor，
# 因為它的 .op 是動態計算的。但它的輸入必須接受新的 TangentVector。

class Form:
    """
    k-Form (Operator View)
    k: 階數
    op: 函數，接受 k 個 TangentVector，回傳 SymPy 表達式
    """
    def __init__(self, degree, evaluator):
        self.k = degree
        self.op = evaluator 
        
    def __call__(self, *vectors):
        if self.k == 0: 
            if callable(self.op): return self.op()
            return self.op
            
        if len(vectors) != self.k: 
            raise ValueError(f"Need {self.k} vectors, got {len(vectors)}")
        
        # 確保傳入的是 TangentVector (雖然鴨子型別也可以)
        return sp.simplify(self.op(*vectors))
    
    # TODO: 未來可以增加 .to_tensor() 方法，將 Operator 轉換為 GeometricTensor (Rank k Covariant)

# ===================================================================
# 4. 外微分算子 (Exterior Derivative)
# ===================================================================
# 邏輯不變，完全依賴 TangentVector.__call__ 和 lie_bracket

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
        for i in range(n):
            X_i = vectors[i]
            others = vectors[:i] + vectors[i+1:]
            
            scalar_field = omega(*others)
            
            # 這裡呼叫的是新版 TangentVector.__call__ (方向導數)
            val = X_i(scalar_field)
            
            term = val if i % 2 == 0 else -val
            total += term
            
        # Part B: omega([X_i, X_j], ...)
        if n >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    # 這裡呼叫的是新版 lie_bracket
                    bracket = lie_bracket(vectors[i], vectors[j])
                    others = vectors[:i] + vectors[i+1:j] + vectors[j+1:]
                    
                    val = omega(bracket, *others)
                    
                    sign = (-1)**(i + j)
                    total += sign * val
                    
        return sp.simplify(total)
    
    return Form(k + 1, d_omega_evaluator)