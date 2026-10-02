"""程序目录的判定。

开发时程序目录就是源码目录(本文件所在目录), 所以 `models/`、`settings.json`
都躺在工程根下。打包成 exe 之后 `__file__` 落在 PyInstaller 的解包目录里
(onedir 布局下是 `_internal/`), 而这两个东西是要跟着 exe 走的 —— 所以冻结后
统一以 exe 所在目录为准, 这样 `make` 打完包把 `models/` 复制到安装目录就能被找到。
"""

import sys
from pathlib import Path


def app_root() -> Path:
    """程序目录: 冻结后是 exe 所在目录, 否则是源码目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent
