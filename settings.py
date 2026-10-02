"""界面设置持久化(settings.json)。

目前只存一个东西: **上次选中的模型**。程序下次启动时会自动切回它。
找不到文件、内容损坏、或者记的模型文件已经不在/加载失败时, 都安静地回落到
默认模型, 不打扰用户。

文件位置在程序目录下(`settings.json`), 和 `models/` 摆在一起。里面记的模型路径
**尽量存相对程序目录的形式**: 这样整个工程搬个地方、或者模型在 `models/` 里
改个位置, 记录依然有效。程序目录外的模型只能记绝对路径。
"""

import json
import os
import tempfile
from pathlib import Path

from paths import app_root

FILE_NAME = "settings.json"
FORMAT_VERSION = 1


class Settings:
    """一份可读写的设置。任何异常都不会往上抛, 只会退回默认值。"""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else (app_root() / FILE_NAME)
        self.data: dict = {}
        self.load()

    # ------------------------------ 读写 ------------------------------ #
    def load(self) -> None:
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, ValueError, UnicodeDecodeError):
            self.data = {}
            return
        self.data = data if isinstance(data, dict) else {}

    def save(self) -> bool:
        """原子写入(先写临时文件再替换), 避免写一半留下坏文件。"""
        payload = dict(self.data)
        payload["version"] = FORMAT_VERSION
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle, temp_name = tempfile.mkstemp(
                prefix=".settings-", suffix=".tmp", dir=str(self.path.parent))
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
            os.replace(temp_name, self.path)
            return True
        except OSError:
            # 存不上(例如程序装在只读目录)不算错误, 只是这次记不住
            return False

    # ------------------------------ 字段 ------------------------------ #
    @property
    def model(self) -> dict | None:
        """上次用的模型: {"name": ..., "path": ...}。

        盘上记的可能是相对程序目录的路径, 这里一律还原成绝对路径再给调用方。
        """
        value = self.data.get("model")
        if not isinstance(value, dict):
            return None
        record = dict(value)
        raw = record.get("path")
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                path = self.path.parent / path
            record["path"] = str(path)
        return record

    def remember_model(self, name: str, path) -> None:
        record = {"name": str(name), "path": self._store_path(path)}
        if self.data.get("model") == record:
            return                       # 没变就别反复写盘
        self.data["model"] = record
        self.save()

    def _store_path(self, path) -> str | None:
        """程序目录里的模型存相对路径(用 `/` 分隔, 换平台也读得懂)。"""
        if not path:
            return None
        target = Path(path)
        try:
            return target.resolve().relative_to(self.path.parent.resolve()).as_posix()
        except (OSError, ValueError):
            return str(target)           # 目录外的模型只能记绝对路径
