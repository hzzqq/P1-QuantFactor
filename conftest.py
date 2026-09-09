"""pytest 根配置：把项目根与 src 加入路径，使 `src`/`shared` 可导入。"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)                     # P1-QuantFactor（src 包）
sys.path.insert(0, os.path.dirname(_ROOT))   # sj 根（shared 包）
