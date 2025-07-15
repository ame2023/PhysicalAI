import pybullet as _p
from pathlib import Path

# ------------ data/ の実パスを取得 ----------------------------------
# Py ≥3.9
from importlib.resources import files as _files

_real_path = _files(__name__).joinpath("data").joinpath("")  # pathlib → str
_real_path = str(_real_path)
# --------------------------------------------------------------------

# ① 未接続なら DIRECT で軽量接続
if not _p.isConnected():
    _p.connect(_p.DIRECT)

# ② data/ を検索パスへ登録（すべてのクライアントで有効）
_p.setAdditionalSearchPath(_real_path)

# ③ 外部から data パスを取得
def get_data_path() -> str:
    """unitree_pybullet/data の絶対パスを返す"""
    return _real_path

# ④ QuadEnv をトップレベルで公開（任意）
try:
    from .QuadGymEnv import QuadEnv
    __all__ = ["QuadEnv", "get_data_path"]
except Exception:
    __all__ = ["get_data_path"]

