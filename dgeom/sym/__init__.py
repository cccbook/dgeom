# 向量微積分模組
from .vcalculus import * # 梯度、散度、旋度等函式

# 微分幾何模組
from ._tensor import *
# from ._tensor_metric import * # 張量 (GeometricTensor)
# from ._space_time import * # 廣義相對論時空（含愛因斯坦重力場方程式)
from ._metric import * # 度規與基本符號
from ._exterior_derivative import * # 外微分
from ._hodge import * # Hodge 指標升降與星算子
from .dgeometry import * # 通用微分幾何函式

# 相對論模組
from .constants import * # 基本物理常數
from .relativity import * # 相對論相關函式與度規
# from .relativity_old import * # 舊版對論相關函式與度規
