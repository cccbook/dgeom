import sympy as sp
from sympy import MutableDenseNDimArray, Matrix, diff

class GeometricTensor:
    """
    通用幾何張量 (Geometric Tensor Base)。
    
    職責：
    1. 數據容器 (NDimArray)
    2. 指標管理 (Index Configuration: Covariant [-1] / Contravariant [1])
    3. 張量代數 (Addition, Scalar Multiplication, Tensor Product, Contraction)
    4. 座標變換 (Transformation laws)
    """

    def __init__(self, data, coords, index_config):
        # 1. 統一數據格式為 MutableDenseNDimArray
        if not isinstance(data, MutableDenseNDimArray):
            if hasattr(data, 'tolist'): # 處理 Matrix
                data = data.tolist()
            self.data = MutableDenseNDimArray(data)
        else:
            self.data = data
            
        self.coords = list(coords)
        self.index_config = list(index_config) # e.g. [1, -1] for T^u_v
        self.rank = len(self.index_config)
        self.dim = len(self.coords)
        
        # 2. 基礎驗證
        if self.data.rank() != self.rank:
            raise ValueError(f"數據階數 ({self.data.rank()}) 與指標設定 ({self.rank}) 不符。")
        
        if self.rank > 0 and any(d != self.dim for d in self.data.shape):
            raise ValueError(f"張量維度 {self.data.shape} 與座標維度 ({self.dim}) 不一致。")

    def __repr__(self):
        # 顯示指標類型: U(上), D(下)
        idx_str = "".join(["U" if t == 1 else "D" for t in self.index_config])
        return f"Tensor(Type=[{idx_str}], Coords={self.coords})\n{self.data}"

    def __getitem__(self, item):
        return self.data[item]

    # ==========================================
    # 算術運算 Overloads (讓公式寫起來更像數學)
    # ==========================================
    
    def __add__(self, other):
        """張量相加: T1 + T2"""
        if not isinstance(other, GeometricTensor):
            raise TypeError("只能與 GeometricTensor 相加")
        if self.index_config != other.index_config:
            raise ValueError(f"指標組態不同無法相加: {self.index_config} vs {other.index_config}")
        if self.coords != other.coords:
            raise ValueError("座標系不同無法相加")
            
        return GeometricTensor(self.data + other.data, self.coords, self.index_config)

    def __sub__(self, other):
        """張量相減: T1 - T2"""
        return self.__add__(other * -1)

    def __mul__(self, other):
        """純量乘法: T * scalar"""
        # 注意: 這裡不處理 Tensor * Tensor (那是 tensor_product)
        # 這裡處理 Tensor * 3 或 Tensor * Symbol('a')
        new_data = self.data.applyfunc(lambda x: x * other)
        return GeometricTensor(new_data, self.coords, self.index_config)

    def __rmul__(self, other):
        """純量乘法 (反向): scalar * T"""
        return self.__mul__(other)

    # ==========================================
    # 張量運算 (Tensor Operations)
    # ==========================================

    def tensor_product(self, other):
        """張量積 (Outer Product): T (x) S"""
        if self.coords != other.coords:
            raise ValueError("必須在相同座標系下進行運算")
            
        new_data = sp.tensorproduct(self.data, other.data)
        new_config = self.index_config + other.index_config
        return GeometricTensor(new_data, self.coords, new_config)

    def contract(self, pos1, pos2):
        """
        張量縮併 (Contraction/Trace)
        pos1, pos2: 要縮併的指標位置 (0-based)
        """
        if self.index_config[pos1] + self.index_config[pos2] != 0:
            raise ValueError(f"必須縮併一上一下指標: {pos1}, {pos2}")
        
        new_data = sp.tensorcontraction(self.data, (pos1, pos2))
        
        # 移除已縮併的指標組態
        new_config = [c for i, c in enumerate(self.index_config) if i not in (pos1, pos2)]
        
        # 如果縮併後變純量，處理 rank=0 的情況
        if not new_config: 
            new_data = MutableDenseNDimArray(new_data)
            
        return GeometricTensor(new_data, self.coords, new_config)

    # ==========================================
    # 座標變換 (Coordinate Transformation)
    # ==========================================

    def _get_jacobian(self, new_coords, old_coords_funcs):
        # J_cov[i, j] = d(old_i) / d(new_j)
        matrix_rows = []
        for old_var in self.coords:
            if old_var not in old_coords_funcs:
                raise ValueError(f"缺少座標 {old_var} 的變換規則")
            expr = old_coords_funcs[old_var]
            row = [diff(expr, new_var) for new_var in new_coords]
            matrix_rows.append(row)
            
        jacobian_cov = Matrix(matrix_rows)
        jacobian_contra = jacobian_cov.inv()
        return jacobian_contra, jacobian_cov

    def transform(self, new_coords, transformation_rules):
        """
        座標變換: T' = J * T * J...
        """
        J_contra, J_cov = self._get_jacobian(new_coords, transformation_rules)
        
        # 1. 變數替換
        try:
            current_data = self.data.applyfunc(lambda x: sp.sympify(x).subs(transformation_rules))
        except TypeError: 
            # Fallback for Scalar
            val = self.data[()].subs(transformation_rules)
            current_data = MutableDenseNDimArray(val)

        temp_array = current_data
        
        # 2. 逐指標縮併
        for idx_pos, idx_type in enumerate(self.index_config):
            # 上標用 J_contra (New/Old)，下標用 J_cov.T (New/Old)
            M = J_contra if idx_type == 1 else J_cov.T
            
            # Product: M (rank 2) * T (rank N) -> Rank N+2
            product = sp.tensorproduct(M, temp_array)
            
            # Contract: M_col (index 1) with T_target (index idx_pos + 2)
            contracted = sp.tensorcontraction(product, (1, idx_pos + 2))
            
            # Permute: 將新的 index (目前在 0) 移回 idx_pos
            if idx_pos > 0:
                new_perm = []
                for i in range(self.rank):
                    if i < idx_pos: new_perm.append(i + 1)
                    elif i == idx_pos: new_perm.append(0)
                    else: new_perm.append(i + 1)
                temp_array = sp.permutedims(contracted, new_perm)
            else:
                temp_array = contracted

        # 3. 簡化
        if temp_array.rank() == 0:
            final_data = MutableDenseNDimArray(sp.simplify(temp_array[()]))
        else:
            final_data = sp.simplify(temp_array)
        
        return GeometricTensor(final_data, new_coords, self.index_config)