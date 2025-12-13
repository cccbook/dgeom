import sympy as sp
from sympy import MutableDenseNDimArray, Matrix, diff

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
