import sympy as sp
from sympy import MutableDenseNDimArray, Matrix, diff
import pytest

# ==========================================
# 核心類別: GeometricTensor
# ==========================================

class GeometricTensor:
    """
    符合數學定義的張量物件 (Component-based Tensor)。
    支援協變/逆變指標區分與座標變換。
    """

    def __init__(self, data, coords, index_config):
        # 確保 data 是 SymPy 的 NDimArray
        if not isinstance(data, MutableDenseNDimArray):
            self.data = MutableDenseNDimArray(data)
        else:
            self.data = data
            
        self.coords = list(coords)
        self.index_config = list(index_config)
        self.rank = len(self.index_config)
        self.dim = len(self.coords)
        
        # 驗證數據階數
        if self.data.rank() != self.rank:
            raise ValueError(f"數據階數 ({self.data.rank()}) 與指標設定長度 ({self.rank}) 不符。")
        
        # 驗證數據維度 (針對 Rank > 0)
        if self.rank > 0 and any(d != self.dim for d in self.data.shape):
            raise ValueError(f"張量分量維度必須與座標維度 ({self.dim}) 一致。")

    def __repr__(self):
        idx_str = ""
        for t in self.index_config:
            idx_str += "U" if t == 1 else "D"
        return f"Tensor(Type=[{idx_str}], Coords={self.coords})\n{self.data}"

    def __getitem__(self, item):
        return self.data[item]

    def _get_jacobian(self, new_coords, old_coords_funcs):
        """
        計算座標變換的雅可比矩陣。
        """
        # J_cov[i, j] = d(old_i) / d(new_j) -> Row=Old, Col=New
        matrix_rows = []
        for old_var in self.coords:
            if old_var not in old_coords_funcs:
                raise ValueError(f"缺少座標 {old_var} 的變換規則。")
            expr = old_coords_funcs[old_var]
            row = [diff(expr, new_var) for new_var in new_coords]
            matrix_rows.append(row)
            
        jacobian_cov = Matrix(matrix_rows)
        # J_contra = d(new) / d(old)
        jacobian_contra = jacobian_cov.inv()
        
        return jacobian_contra, jacobian_cov

    def transform(self, new_coords, transformation_rules):
        """
        執行座標變換。
        """
        # 1. 取得變換矩陣
        J_contra, J_cov = self._get_jacobian(new_coords, transformation_rules)
        
        # 2. 替換舊座標變數
        # 使用 applyfunc 並捕捉 Rank 0 的特殊情況
        try:
            current_data = self.data.applyfunc(lambda x: sp.sympify(x).subs(transformation_rules))
        except TypeError: 
            # 針對 Rank 0 Array (純量)
            val = self.data[()].subs(transformation_rules)
            current_data = MutableDenseNDimArray(val)

        # 3. 進行張量縮併運算
        temp_array = current_data
        
        for idx_pos, idx_type in enumerate(self.index_config):
            # 決定變換矩陣: 必須是 M[Row=New, Col=Old] 以配合縮併邏輯
            if idx_type == 1:
                # 逆變 (+1): J_contra 本身就是 dx'/dx (Row=New, Col=Old)
                M = J_contra
            else:
                # 協變 (-1): J_cov 是 dx/dx' (Row=Old, Col=New)
                # 需轉置為 (Row=New, Col=Old)
                M = J_cov.T
            
            # Tensor Product: M (rank 2) (x) Temp_Array (rank N)
            product = sp.tensorproduct(M, temp_array)
            
            # 縮併: M 的 column (index 1, Old) 與 Tensor 的目標 index
            contracted = sp.tensorcontraction(product, (1, idx_pos + 2))
            
            # 將新的 index (目前在 0) 移回 idx_pos
            if idx_pos > 0:
                new_perm = []
                for i in range(self.rank):
                    if i < idx_pos:
                        new_perm.append(i + 1)
                    elif i == idx_pos:
                        new_perm.append(0)
                    else:
                        new_perm.append(i + 1)
                temp_array = sp.permutedims(contracted, new_perm)
            else:
                temp_array = contracted

        # 簡化結果
        if temp_array.rank() == 0:
            final_data = MutableDenseNDimArray(sp.simplify(temp_array[()]))
        else:
            final_data = sp.simplify(temp_array)
        
        return GeometricTensor(final_data, new_coords, self.index_config)

    def contract(self, pos1, pos2):
        if self.index_config[pos1] + self.index_config[pos2] != 0:
            raise ValueError("必須縮併一個協變指標和一個逆變指標。")
        
        new_data = sp.tensorcontraction(self.data, (pos1, pos2))
        new_config = [c for i, c in enumerate(self.index_config) if i not in (pos1, pos2)]
        return GeometricTensor(new_data, self.coords, new_config)

    def tensor_product(self, other):
        if self.coords != other.coords:
            raise ValueError("必須在相同座標系下進行運算。")
        new_data = sp.tensorproduct(self.data, other.data)
        new_config = self.index_config + other.index_config
        return GeometricTensor(new_data, self.coords, new_config)


# ==========================================
# Pytest 測試案例
# ==========================================

@pytest.fixture
def polar_setup():
    """
    Fixtures: 設定笛卡爾與極座標的符號及變換規則
    """
    x, y = sp.symbols('x y')
    r, theta = sp.symbols('r theta', real=True, positive=True)
    
    rules = {
        x: r * sp.cos(theta),
        y: r * sp.sin(theta)
    }
    
    return (x, y), (r, theta), rules

def test_vector_transformation(polar_setup):
    """
    測試逆變向量 (Contravariant Vector) 的變換: V^i
    笛卡爾: [x, y] -> 極座標: [r, 0]
    """
    (x, y), (r, theta), rules = polar_setup
    
    # 定義 V = x \hat{x} + y \hat{y}
    V_cart = GeometricTensor([x, y], [x, y], [1])
    
    # 執行變換
    V_polar = V_cart.transform([r, theta], rules)
    
    # Assertions
    assert V_polar.coords == [r, theta]
    assert V_polar.index_config == [1]
    
    # 驗證分量數值: [r, 0]
    # 使用 sp.simplify(a - b) == 0 來比較符號表達式
    assert sp.simplify(V_polar.data[0] - r) == 0
    assert sp.simplify(V_polar.data[1]) == 0

def test_metric_transformation(polar_setup):
    """
    測試協變張量 (Metric Tensor) 的變換: g_ij
    笛卡爾: I -> 極座標: diag(1, r^2)
    """
    (x, y), (r, theta), rules = polar_setup
    
    # 定義 Euclidean Metric
    g_cart = GeometricTensor([[1, 0], [0, 1]], [x, y], [-1, -1])
    
    # 執行變換
    g_polar = g_cart.transform([r, theta], rules)
    
    # Assertions
    assert g_polar.index_config == [-1, -1]
    
    # 驗證分量: [[1, 0], [0, r^2]]
    assert sp.simplify(g_polar.data[0, 0] - 1) == 0
    assert sp.simplify(g_polar.data[0, 1]) == 0
    assert sp.simplify(g_polar.data[1, 0]) == 0
    assert sp.simplify(g_polar.data[1, 1] - r**2) == 0

def test_scalar_invariant(polar_setup):
    """
    測試純量不變量 (Invariant): V^2 = g_ij V^i V^j
    驗證不管在哪個座標系計算，長度平方都應該是 r^2 (即 x^2 + y^2)
    """
    (x, y), (r, theta), rules = polar_setup
    
    # 準備極座標下的張量
    V_cart = GeometricTensor([x, y], [x, y], [1])
    g_cart = GeometricTensor([[1, 0], [0, 1]], [x, y], [-1, -1])
    
    V_polar = V_cart.transform([r, theta], rules)
    g_polar = g_cart.transform([r, theta], rules)
    
    # 計算 g_ij (x) V^k -> Rank 3 Tensor
    T_mixed = g_polar.tensor_product(V_polar)
    
    # 下標化 (Lowering Index): g_ij V^j -> V_i
    # 縮併 index 1 (g的第二個下標) 和 index 2 (V的上標)
    V_covector = T_mixed.contract(1, 2)
    
    # 驗證中間產物 V_i = [r, 0]
    assert sp.simplify(V_covector.data[0] - r) == 0
    assert sp.simplify(V_covector.data[1]) == 0
    
    # 最後縮併 V_i V^i -> Scalar
    scalar_tensor = V_covector.tensor_product(V_polar).contract(0, 1)
    
    # 取出純量數值 (注意使用 [()] 存取 Rank 0 array)
    scalar_value = scalar_tensor.data[()]
    
    # Assert: 結果應為 r^2
    assert sp.simplify(scalar_value - r**2) == 0

def test_mixed_tensor_transformation():
    """
    測試混合張量 T^i_j 的變換
    這裡測試 Identity 變換 T^i_j = delta^i_j
    """
    x, y = sp.symbols('x y')
    u, v = sp.symbols('u v')
    # 簡單線性變換: x = u + v, y = u - v
    rules = {x: u + v, y: u - v}
    
    # Kronecker delta
    delta_data = [[1, 0], [0, 1]]
    T_cart = GeometricTensor(delta_data, [x, y], [1, -1]) # Up, Down
    
    T_new = T_cart.transform([u, v], rules)
    
    # 混合張量如果是 Identity (Kronecker delta)，在任何座標系下都應該保持不變
    # 但這只對 Tensor 定義成立。如果是 Matrix 則不一定。
    # 數學上 delta^i_j 是不變張量。
    
    # 驗證 T'^i_j 依然是 identity
    assert sp.simplify(T_new.data[0, 0] - 1) == 0
    assert sp.simplify(T_new.data[1, 1] - 1) == 0
    assert sp.simplify(T_new.data[0, 1]) == 0
    assert sp.simplify(T_new.data[1, 0]) == 0

def test_error_handling():
    """
    測試錯誤捕捉機制
    """
    x, y = sp.symbols('x y')
    
    # 測試1: 縮併相同類型的指標 (應報錯)
    T = GeometricTensor([[1, 0], [0, 1]], [x, y], [1, 1]) # 兩個上標
    with pytest.raises(ValueError, match="必須縮併一個協變指標和一個逆變指標"):
        T.contract(0, 1)

    # 測試2: 不同座標系運算 (應報錯)
    z = sp.symbols('z')
    T1 = GeometricTensor([1, 0], [x, y], [1])
    T2 = GeometricTensor([1, 0], [x, z], [1])
    with pytest.raises(ValueError, match="必須在相同座標系下"):
        T1.tensor_product(T2)

if __name__ == "__main__":
    # 允許直接運行檔案進行測試
    import sys
    sys.exit(pytest.main(["-v", __file__]))