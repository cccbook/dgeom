import sympy as sp
import numpy as np
import math
import pytest
from dgeom.sym import *

def test_relativity_minkowski_flatness():
    r"""
    ### 🧪 驗證 relativity.py：閔可夫斯基度規的平坦性
    數學公式: $R_{\mu \nu} = 0$
    """
    metric = MINKOWSKI_METRIC
    G_cov = metric.g
    G_cont = metric.g_inv
    coords = metric.coords
    
    # 1. 計算 Ricci 張量 $R_{\mu \nu}$
    # 閔可夫斯基度規是一個 4D 時空度規，回傳 4x4 矩陣
    R_mn = ricci_tensor(G_cont, G_cov, coords) 
    
    # 2. 閔可夫斯基時空是平坦的 (Flat Spacetime)，其 Ricci 張量應為零
    assert sp.simplify(R_mn) == sp.zeros(4, 4), \
        r"閔可夫斯基度規的 Ricci 張量 $R_{\mu \nu}$ 應為零"

