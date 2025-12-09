# test_example.py
# 假設 my_module.py 在同一個目錄下
from my_module import add, multiply
import pytest

# 1. 簡單的測試函式
def test_add_positive_numbers():
    """測試正數相加是否正確。"""
    # 斷言 (assert) 是測試的關鍵，它檢查實際結果是否等於預期結果
    assert add(1, 2) == 3

def test_add_negative_numbers():
    """測試負數相加是否正確。"""
    assert add(-1, -1) == -2

# 2. 測試預期會失敗的情況 (使用 pytest.mark.xfail)
@pytest.mark.xfail(reason="這個測試預期會失敗，例如處理浮點數精度問題。")
def test_add_float_error():
    """這個測試預期會因為浮點數精度問題而失敗。"""
    # 1.0 + 2.0 應該是 3.0
    assert add(1.0, 2.0) == 3.0000000000000001

# 3. 參數化測試 (使用 @pytest.mark.parametrize)
# 允許你用不同的輸入資料運行相同的測試邏輯。
@pytest.mark.parametrize("a, b, expected", [
    (1, 1, 1),      # 1 * 1 = 1
    (2, 3, 6),      # 2 * 3 = 6
    (-1, 5, -5),    # -1 * 5 = -5
    (0, 100, 0),    # 0 * 100 = 0
])
def test_multiply_various_cases(a, b, expected):
    """使用參數化測試各種乘法案例。"""
    assert multiply(a, b) == expected