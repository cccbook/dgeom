import sympy as sp
# 引入 SymPy 庫，用於所有的符號運算，如微分、簡化、矩陣求逆等。
from sympy import MutableDenseNDimArray, Matrix, diff

# 從 SymPy 引入特定的類和函數
# MutableDenseNDimArray: 用於儲存張量分量的 N 維可變數組。
# Matrix: 用於處理 Jacobian 矩陣的運算。
# diff: 用於計算偏導數，是座標變換的基礎。

class GeometricTensor:
    """
    通用幾何張量 (Geometric Tensor Base)。
    
    這是用於表示和操作幾何張量的核心類別。
    
    職責：
    1. 數據容器 (NDimArray)：儲存張量的分量數據。
    2. 指標管理 (Index Configuration: Covariant [-1] / Contravariant [1])：定義張量的類型 (p, q)。
    3. 張量代數 (Addition, Scalar Multiplication, Tensor Product, Contraction)：基本的代數運算。
    4. 座標變換 (Transformation laws)：張量在不同座標系之間變換的規則。
    """

    def __init__(self, data, coords, index_config):
        """
        初始化張量物件。
        :param data: 張量分量的數組/列表/SymPy矩陣。
        :param coords: 當前座標系中的變數列表，例如 [Symbol('x'), Symbol('y')]。
        :param index_config: 指標配置列表，1 代表反變(上標)，-1 代表協變(下標)，例如 [1, -1] 代表 T^u_v。
        """
        # 1. 統一數據格式為 MutableDenseNDimArray
        if not isinstance(data, MutableDenseNDimArray):
            if hasattr(data, 'tolist'): # 檢查物件是否有 tolist 方法 (通常是 SymPy Matrix)
                data = data.tolist()    # 將 Matrix 轉換為 Python 列表
            self.data = MutableDenseNDimArray(data) # 轉換為 SymPy 的 N 維數組
        else:
            self.data = data # 如果已經是 NDimArray，則直接使用
            
        self.coords = list(coords)                 # 儲存座標變數，並確保是列表類型
        self.index_config = list(index_config)     # 儲存指標配置，1 (上標) 或 -1 (下標)
        self.rank = len(self.index_config)         # 張量的階數 (Rank) = 指標的數量
        self.dim = len(self.coords)                # 座標空間的維度 (Dimension)
        
        # 2. 基礎驗證
        # 檢查數據陣列的階數 (維度數量) 是否與指標數量一致
        if self.data.rank() != self.rank:
            raise ValueError(f"數據階數 ({self.data.rank()}) 與指標設定 ({self.rank}) 不符。")
        
        # 對於非純量張量 (rank > 0)，檢查每個軸的維度是否與座標維度 (dim) 一致
        if self.rank > 0 and any(d != self.dim for d in self.data.shape):
            raise ValueError(f"張量維度 {self.data.shape} 與座標維度 ({self.dim}) 不一致。")

    def __repr__(self):
        """定義物件的字串表示，方便印出除錯。"""
        # 將指標配置轉換為可讀的 'U' (Up/Contravariant) 或 'D' (Down/Covariant)
        idx_str = "".join(["U" if t == 1 else "D" for t in self.index_config])
        return f"Tensor(Type=[{idx_str}], Coords={self.coords})\n{self.data}"

    def __getitem__(self, item):
        """允許使用索引來存取分量，例如 T[0, 1]。"""
        return self.data[item]

    # ==========================================
    # 算術運算 Overloads (讓公式寫起來更像數學)
    # 這裡處理逐分量操作
    # ==========================================
    
    def __add__(self, other):
        """
        張量相加: T1 + T2
        要求：必須具有相同的指標組態、相同的階數和相同的座標系。
        """
        if not isinstance(other, GeometricTensor):
            raise TypeError("只能與 GeometricTensor 相加") # 檢查類型
        if self.index_config != other.index_config:
            # 張量加法要求類型 (p, q) 必須完全相同
            raise ValueError(f"指標組態不同無法相加: {self.index_config} vs {other.index_config}")
        if self.coords != other.coords:
            raise ValueError("座標系不同無法相加") # 必須在同一個座標系下
            
        # 數據逐分量相加 (NDimArray + NDimArray)
        return GeometricTensor(self.data + other.data, self.coords, self.index_config)

    def __sub__(self, other):
        """張量相減: T1 - T2"""
        # 透過加法和純量乘法實現，利用 T1 - T2 = T1 + (-1 * T2)
        return self.__add__(other * -1)

    def __mul__(self, other):
        """
        純量乘法: T * scalar
        每個張量分量都乘以純量 (可以是數值或 SymPy 符號)。
        """
        # 使用 applyfunc 函數將純量乘法應用到數據陣列中的每個分量上
        new_data = self.data.applyfunc(lambda x: x * other)
        # 階數和組態不變
        return GeometricTensor(new_data, self.coords, self.index_config)

    def __rmul__(self, other):
        """純量乘法 (反向): scalar * T，確保交換律成立"""
        return self.__mul__(other)

    # ==========================================
    # 張量運算 (Tensor Operations)
    # ==========================================

    def tensor_product(self, other):
        """
        張量積 (Outer Product): T (x) S
        結果階數為兩者之和，分量為兩者分量的簡單乘積。
        """
        if self.coords != other.coords:
            raise ValueError("必須在相同座標系下進行運算")
            
        # 使用 SymPy 的 tensorproduct 函數計算外積
        new_data = sp.tensorproduct(self.data, other.data)
        # 新的指標組態是將兩者的指標配置列表串接起來
        new_config = self.index_config + other.index_config
        return GeometricTensor(new_data, self.coords, new_config)

    def contract(self, pos1, pos2):
        """
        張量縮併 (Contraction/Trace)
        操作：選擇一對上標和下標，進行愛因斯坦求和約定，階數減少 2 階。
        :param pos1: 要縮併的第一個指標位置 (0-based)。
        :param pos2: 要縮併的第二個指標位置 (0-based)。
        """
        # 縮併要求必須是一上標 (1) 一下標 (-1) 的配對
        if self.index_config[pos1] + self.index_config[pos2] != 0:
            raise ValueError(f"必須縮併一上一下指標: {pos1}, {pos2}")
        
        # 使用 SymPy 的 tensorcontraction 函數，傳入要縮併的軸的位置元組
        new_data = sp.tensorcontraction(self.data, (pos1, pos2))
        
        # 移除已縮併的兩個指標組態，得到新張量的組態
        new_config = [c for i, c in enumerate(self.index_config) if i not in (pos1, pos2)]
        
        # 如果縮併後變為純量 (Rank 0)，確保數據結構是 NDimArray
        if not new_config: 
            new_data = MutableDenseNDimArray(new_data)
            
        return GeometricTensor(new_data, self.coords, new_config)

    # ==========================================
    # 座標變換 (Coordinate Transformation)
    # ==========================================

    def _get_jacobian(self, new_coords, old_coords_funcs):
        """
        計算座標變換所需的 Jacobian 矩陣 (J) 和其逆矩陣 (J_inv)。
        J_cov (協變變換矩陣) 和 J_contra (反變變換矩陣)。
        """
        # J_cov[i, j] = d(old_i) / d(new_j)
        matrix_rows = []
        for old_var in self.coords: # 遍歷舊座標變數 (e.g., x)
            if old_var not in old_coords_funcs:
                raise ValueError(f"缺少座標 {old_var} 的變換規則")
            expr = old_coords_funcs[old_var] # 舊座標用新座標表示的表達式
            # row = [ d(old_var) / d(new_var_1), d(old_var) / d(new_var_2), ... ]
            row = [diff(expr, new_var) for new_var in new_coords] # 計算偏導數
            matrix_rows.append(row)
            
        # 協變變換矩陣 J_cov: (J_cov)^i_j = d(x^i) / d(x'^j)
        jacobian_cov = Matrix(matrix_rows)
        # 反變變換矩陣 J_contra: (J_contra)^i_j = d(x'^i) / d(x^j) = (J_cov)^-1
        jacobian_contra = jacobian_cov.inv()
        return jacobian_contra, jacobian_cov

    def transform(self, new_coords, transformation_rules):
        """
        執行座標變換: T' = J * T * J...
        每個上標指標與 J_contra 縮併，每個下標指標與 J_cov.T 縮併。
        """
        # 取得反變 Jacobian 和協變 Jacobian
        J_contra, J_cov = self._get_jacobian(new_coords, transformation_rules)
        
        # 1. 變數替換 (將張量分量從舊座標表示替換為新座標表示)
        try:
            # 對 NDimArray 中的每個元素應用 subs 替換規則
            current_data = self.data.applyfunc(lambda x: sp.sympify(x).subs(transformation_rules))
        except TypeError: 
            # 處理 Rank 0 (純量) 的特殊情況
            val = self.data[()].subs(transformation_rules)
            current_data = MutableDenseNDimArray(val)

        temp_array = current_data
        
        # 2. 逐指標縮併
        # 遍歷張量的每個指標，依次進行變換矩陣的乘積和縮併
        for idx_pos, idx_type in enumerate(self.index_config):
            # 選擇變換矩陣 M
            # 上標 (1): 使用 J_contra
            # 下標 (-1): 使用 J_cov 的轉置 (J_cov.T)
            M = J_contra if idx_type == 1 else J_cov.T
            
            # Product: M (rank 2) * T (rank N) -> Rank N+2
            # 變換矩陣 M 的指標會放在最前面 (索引 0 和 1)
            product = sp.tensorproduct(M, temp_array)
            
            # Contract: 將 M 的舊指標 (索引 1) 與 張量的目標舊指標 (索引 idx_pos + 2) 進行縮併
            # 這樣新張量指標就取代了舊張量指標的位置
            contracted = sp.tensorcontraction(product, (1, idx_pos + 2))
            
            # Permute: 將變換後的新指標 (目前在索引 0) 移回其在張量中的原始位置 idx_pos
            if idx_pos > 0:
                # 建立新的排列順序
                new_perm = []
                for i in range(self.rank):
                    if i < idx_pos: new_perm.append(i + 1) # 讓前面的新指標往後移一位
                    elif i == idx_pos: new_perm.append(0)  # 將變換產生的新指標 (在 0) 移到此處
                    else: new_perm.append(i + 1)           # 後面的指標也往後移一位
                # 執行指標重排
                temp_array = sp.permutedims(contracted, new_perm)
            else:
                # 如果是第一個指標 (idx_pos=0)，新指標已經在位置 0，不需重排
                temp_array = contracted

        # 3. 簡化
        # 對所有分量進行代數簡化
        if temp_array.rank() == 0:
            final_data = MutableDenseNDimArray(sp.simplify(temp_array[()]))
        else:
            final_data = sp.simplify(temp_array)
        
        # 使用新的數據和新的座標返回張量物件，指標組態不變
        return GeometricTensor(final_data, new_coords, self.index_config)