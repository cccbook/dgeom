from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np

class AbstractManifold(ABC):
    """
    抽象流形類別 (Abstract Manifold Class)。
    定義了n維微分流形所需的基本結構和操作。
    """
    
    def __init__(self, dimension: int):
        """
        初始化流形。
        :param dimension: 流形的維度 n。
        """
        if dimension <= 0:
            raise ValueError("流形的維度必須是正整數。")
        self.dimension = dimension
        # 坐標圖集的儲存結構（在實際子類中會被填充）
        self.atlas: Dict[str, Any] = {}
        print(f"創建了一個 {self.dimension} 維抽象流形物件。")

    @abstractmethod
    def chart_domain_point(self, point:np.ndarray) -> str:
        """
        判斷流形上的點屬於哪個坐標圖的定義域 U_alpha。
        由於流形上的點表示可以是複雜的，這裡假設 point 是一個代表流形上點的 n 維向量。
        :param point: 流形上的點 (n維向量或其表示)。
        :return: 坐標圖的 ID (例如 'chart_1', 'chart_2')。
        """
        pass

    @abstractmethod
    def chart_map(self, point:np.ndarray, chart_id: str) -> np.ndarray:
        """
        坐標圖映射 φ_alpha: U_alpha -> V_alpha ⊂ R^n。
        將流形上的點映射到歐氏空間中的坐標。
        :param point: 流形上的點。
        :param chart_id: 坐標圖 ID。
        :return: R^n 中的坐標 (n維向量)。
        """
        pass

    @abstractmethod
    def inverse_chart_map(self, coordinates, chart_id: str) -> np.ndarray:
        """
        逆坐標圖映射 φ_alpha^{-1}: V_alpha -> U_alpha。
        將歐氏空間中的坐標拉回到流形上的點。
        :param coordinates: R^n 中的坐標。
        :param chart_id: 坐標圖 ID。
        :return: 流形上的點。
        """
        pass

    @abstractmethod
    def transition_map(self, coords_alpha, chart_id_alpha: str, chart_id_beta: str) -> np.ndarray:
        """
        坐標變換 τ_alpha_beta = φ_beta ◦ φ_alpha^{-1}。
        將一個坐標圖中的坐標轉換到另一個坐標圖中。
        :param coords_alpha: 第一個坐標圖中的坐標。
        :param chart_id_alpha: 第一個坐標圖 ID。
        :param chart_id_beta: 第二個坐標圖 ID。
        :return: 第二個坐標圖中的坐標。
        """
        pass

class SphereManifold(AbstractManifold):
    """
    2維球面流形 S^2 的實作。
    使用立體投影 (Stereographic Projection) 作為坐標圖。
    """
    
    def __init__(self):
        # 球面是 2 維流形
        super().__init__(dimension=2)
        # 定義圖集名稱
        self.chart_ids = ['chart_north', 'chart_south']
        
    def chart_domain_point(self, point: np.ndarray) -> str:
        """
        決定給定點適合使用哪個坐標圖。
        point: 3維向量 (x, y, z)，且 x^2+y^2+z^2 = 1
        
        策略：
        如果 z > 0 (北半球)，使用南極投影 (chart_south) 以避免北極奇異點。
        如果 z <= 0 (南半球)，使用北極投影 (chart_north) 以避免南極奇異點。
        """
        z = point[2]
        # 為了數值穩定，我們盡量選離奇異點遠的那個圖
        if z >= 0:
            return 'chart_south' # 投影點在南極，覆蓋北半球沒問題
        else:
            return 'chart_north' # 投影點在北極，覆蓋南半球沒問題

    def chart_map(self, point: np.ndarray, chart_id: str) -> np.ndarray:
        """
        將球面上的點 (x, y, z) 映射到 R^2 平面 (u, v)。
        """
        x, y, z = point
        
        if chart_id == 'chart_north':
            # 北極投影：奇異點在 z = 1
            if np.isclose(z, 1.0):
                raise ValueError("北極點無法在 chart_north 中表示。")
            denom = 1.0 - z
            return np.array([x / denom, y / denom])
            
        elif chart_id == 'chart_south':
            # 南極投影：奇異點在 z = -1
            if np.isclose(z, -1.0):
                raise ValueError("南極點無法在 chart_south 中表示。")
            denom = 1.0 + z
            return np.array([x / denom, y / denom])
            
        else:
            raise ValueError(f"未知的坐標圖 ID: {chart_id}")

    def inverse_chart_map(self, coordinates: np.ndarray, chart_id: str) -> np.ndarray:
        """
        將 R^2 平面上的點 (u, v) 拉回球面 (x, y, z)。
        """
        u, v = coordinates
        denom = u**2 + v**2 + 1
        
        # 公共部分
        x = 2 * u / denom
        y = 2 * v / denom
        
        if chart_id == 'chart_north':
            # 北極投影的反函數，z = (r^2 - 1) / (r^2 + 1)
            z = (u**2 + v**2 - 1) / denom
        elif chart_id == 'chart_south':
            # 南極投影的反函數，z 的符號相反 (或者說公式推導稍微不同)
            # z = (1 - r^2) / (1 + r^2)
            z = (1 - u**2 - v**2) / denom
        else:
            raise ValueError(f"未知的坐標圖 ID: {chart_id}")
            
        return np.array([x, y, z])

    def transition_map(self, coords_alpha: np.ndarray, chart_id_alpha: str, chart_id_beta: str) -> np.ndarray:
        """
        坐標變換。
        如果 alpha 和 beta 相同，則是恆等映射。
        如果是南北極互換，則執行圓反演 (Circle Inversion)。
        """
        if chart_id_alpha == chart_id_beta:
            return coords_alpha
            
        # 檢查是否為有效的切換
        valid_charts = {'chart_north', 'chart_south'}
        if chart_id_alpha not in valid_charts or chart_id_beta not in valid_charts:
            raise ValueError("無效的坐標圖 ID。")

        # 從北轉南 或 從南轉北，公式皆為 u' = u / (u^2+v^2)
        # 這是因為立體投影的特性：兩個投影點互為對徑點時，平面坐標互為倒數關係(幾何上)
        u, v = coords_alpha
        r2 = u**2 + v**2
        
        if np.isclose(r2, 0):
            raise ValueError("無法轉換原點 (對應到另一個圖的無窮遠處/奇異點)。")
            
        return np.array([u / r2, v / r2])

# --- 測試範例 (Main) ---
if __name__ == "__main__":
    # 1. 初始化球面流形
    s2 = SphereManifold()
    
    # 2. 定義赤道上的一個點 P = (1, 0, 0)
    # 這個點在兩個圖中都是合法的
    point_P = np.array([1.0, 0.0, 0.0])
    print(f"\n流形上的點 P: {point_P}")
    
    # 3. 測試 chart_domain_point
    suggested_chart = s2.chart_domain_point(point_P)
    print(f"建議的坐標圖: {suggested_chart}") # 赤道 z=0，我們邏輯預設給 chart_north
    
    # 4. 測試映射到 R^2 (chart_map)
    # 使用北極投影
    coords_north = s2.chart_map(point_P, 'chart_north')
    print(f"在 chart_north 中的坐標 (u, v): {coords_north}") # 預期 (1, 0)
    
    # 使用南極投影
    coords_south = s2.chart_map(point_P, 'chart_south')
    print(f"在 chart_south 中的坐標 (u', v'): {coords_south}") # 預期 (1, 0)
    
    # 5. 測試轉移映射 (Transition Map)
    # 我們選一個不在赤道的點，例如 (3/5, 4/5, 0) -> 還是赤道，換一個
    # 選一個北半球的點 Q，但不要是北極，例如 Q = (0.6, 0, 0.8)
    point_Q = np.array([0.6, 0.0, 0.8])
    
    print(f"\n流形上的點 Q: {point_Q}")
    
    # Q 在 chart_north 的坐標
    q_coords_north = s2.chart_map(point_Q, 'chart_north')
    print(f"Q 在 chart_north (u, v): {q_coords_north}")
    
    # 直接計算 Q 在 chart_south 的坐標
    q_coords_south_direct = s2.chart_map(point_Q, 'chart_south')
    print(f"Q 在 chart_south (直接計算): {q_coords_south_direct}")
    
    # 使用 transition_map 從 north 轉到 south
    q_coords_south_trans = s2.transition_map(q_coords_north, 'chart_north', 'chart_south')
    print(f"Q 在 chart_south (經由轉移映射): {q_coords_south_trans}")
    
    # 驗證兩者是否相等
    assert np.allclose(q_coords_south_direct, q_coords_south_trans)
    print(">> 驗證成功：直接映射與轉移映射結果一致。")

    # 6. 測試逆映射 (Inverse Map)
    q_recovered = s2.inverse_chart_map(q_coords_north, 'chart_north')
    print(f"從 chart_north 坐標拉回流形的點: {q_recovered}")
    assert np.allclose(point_Q, q_recovered)
    print(">> 驗證成功：逆映射準確還原了點 Q。")