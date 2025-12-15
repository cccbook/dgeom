import numpy as np
import pytest
from dgeom.sym import *

@pytest.fixture
def s2_manifold(request):
    return SphereManifold()

def test_manifold_initialization_and_dimension(s2_manifold):
    """
    測試流形物件的初始化及其維度是否正確。
    """
    assert s2_manifold.dimension == 2
    assert 'chart_north' in s2_manifold.chart_ids
    assert 'chart_south' in s2_manifold.chart_ids

def test_chart_domain_point(s2_manifold):
    """
    測試 chart_domain_point 函數的邏輯。
    """
    # 北極 (N)：z=1，應該建議使用 chart_south
    point_N = np.array([0.0, 0.0, 1.0])
    assert s2_manifold.chart_domain_point(point_N) == 'chart_south'

    # 南極 (S)：z=-1，應該建議使用 chart_north
    point_S = np.array([0.0, 0.0, -1.0])
    assert s2_manifold.chart_domain_point(point_S) == 'chart_north'
    
    # 赤道點 (E)：z=0，邏輯上會給 chart_south 或 chart_north (取決於 >= 0 的設計)
    point_E = np.array([1.0, 0.0, 0.0])
    assert s2_manifold.chart_domain_point(point_E) == 'chart_south'

def test_chart_map_and_inverse_consistency(s2_manifold):
    """
    測試 chart_map 和 inverse_chart_map 的相互一致性。
    """
    # 測試點 Q (北半球，非北極)
    point_Q = np.array([0.6, 0.0, 0.8])
    
    # --- North Chart 測試 ---
    coords_north = s2_manifold.chart_map(point_Q, 'chart_north')
    # Q=(0.6, 0, 0.8), (u, v) = (0.6/(1-0.8), 0) = (0.6/0.2, 0) = (3.0, 0.0)
    expected_north = np.array([3.0, 0.0])
    assert np.allclose(coords_north, expected_north)
    
    recovered_north = s2_manifold.inverse_chart_map(coords_north, 'chart_north')
    assert np.allclose(point_Q, recovered_north)

    # --- South Chart 測試 ---
    coords_south = s2_manifold.chart_map(point_Q, 'chart_south')
    # Q=(0.6, 0, 0.8), (u', v') = (0.6/(1+0.8), 0) = (0.6/1.8, 0) = (1/3, 0)
    expected_south = np.array([1/3, 0.0])
    assert np.allclose(coords_south, expected_south)
    
    recovered_south = s2_manifold.inverse_chart_map(coords_south, 'chart_south')
    assert np.allclose(point_Q, recovered_south)

def test_transition_map(s2_manifold):
    """
    測試轉移映射 τ_NS 和 τ_SN 是否正確執行圓反演。
    """
    # 1. 測試點 Q 在 chart_north 中的坐標 (3.0, 0.0)
    coords_north = np.array([3.0, 0.0])
    
    # 從 North 轉到 South
    coords_south_trans = s2_manifold.transition_map(coords_north, 'chart_north', 'chart_south')
    # 預期結果: u' = u / (u^2+v^2) = 3 / 9 = 1/3
    expected_south = np.array([1/3, 0.0])
    assert np.allclose(coords_south_trans, expected_south)
    
    # 2. 測試逆向轉移：從 South 轉回 North
    coords_north_trans = s2_manifold.transition_map(coords_south_trans, 'chart_south', 'chart_north')
    # 預期結果: u = u' / (u'^2+v'^2) = (1/3) / (1/9) = 3
    expected_north = np.array([3.0, 0.0])
    assert np.allclose(coords_north_trans, expected_north)

def test_chart_map_exceptions(s2_manifold):
    """
    測試 chart_map 在奇異點處是否拋出 ValueError。
    """
    # 北極點 (1.0)
    point_N = np.array([0.0, 0.0, 1.0])
    # 北極投影 (chart_north) 在北極點處是奇異點，應拋出錯誤
    with pytest.raises(ValueError):
        s2_manifold.chart_map(point_N, 'chart_north')

    # 南極點 (-1.0)
    point_S = np.array([0.0, 0.0, -1.0])
    # 南極投影 (chart_south) 在南極點處是奇異點，應拋出錯誤
    with pytest.raises(ValueError):
        s2_manifold.chart_map(point_S, 'chart_south')

def test_transition_map_exceptions(s2_manifold):
    """
    測試 transition_map 在原點 (對應另一個圖的奇異點) 是否拋出 ValueError。
    """
    # 坐標原點 (u, v) = (0, 0)
    coords_origin = np.array([0.0, 0.0])
    
    # 原點對應球面上的北極點 (N)，此點的南極坐標是無窮遠，轉移映射應失敗
    with pytest.raises(ValueError):
        s2_manifold.transition_map(coords_origin, 'chart_north', 'chart_south')
        
    # 原點對應球面上的南極點 (S)，此點的北極坐標是無窮遠，轉移映射應失敗
    with pytest.raises(ValueError):
        s2_manifold.transition_map(coords_origin, 'chart_south', 'chart_north')