"""训练进度对话框 —— 后台线程 + 进度 + 日志 + 可取消。

线程规则(重要): worker 线程**绝不碰任何 Tk 控件**, 只往 `queue.Queue` 里丢
消息; 主线程用 `after()` 轮询队列并刷新界面。
"""

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import arch as arch_mod
import trainer
from ui_widgets import COLOR_FAIL, COLOR_MUTED, COLOR_PASS, Dialog, ScrollFrame

POLL_MS = 120
LOG_LIMIT = 400


class TrainDialog(Dialog):
    """跑一次训练。

    返回 dict: {"saved": bool, "path": Path|None, "info": dict, "state_dict": dict}
    或 None(取消/失败/用户关闭且没有有效结果)。
    """

    def __init__(self, parent, workspace, architecture: dict, config: trainer.TrainConfig,
                 metrics=None, default_name: str = "自定义模型",
                 default_path: str = ""):
        super().__init__(parent, "训练模型", resizable=True)
        self.workspace = workspace
        self.architecture = arch_mod.normalize_arch(
            architecture, len(workspace.train_labels()))
        self.config = config
        self.metrics = metrics or self.metrics
        self.default_name = default_name
        self.default_path = default_path
        self.target_path: str | None = None      # 调用方预先选好的 .pt 路径

        self.queue: "queue.Queue[tuple]" = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.result_payload: trainer.TrainResult | None = None
        self.result: dict | None = None
        self._log_lines: list[str] = []
        self._started_at = 0.0
        px = self.metrics.px

        body = ttk.Frame(self, padding=px(12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)

        # ------------------------------ 概要 ------------------------------ #
        head = ttk.LabelFrame(body, text="本次训练", padding=px(8))
        head.grid(row=0, column=0, sticky="ew")
        self.head_text = ttk.Label(head, text="", font=("Segoe UI", 9),
                                   justify="left")
        self.head_text.pack(anchor="w")
        self._refresh_head()

        # ------------------------------ 进度 ------------------------------ #
        progress_box = ttk.Frame(body)
        progress_box.grid(row=1, column=0, sticky="ew", pady=(px(8), px(6)))
        progress_box.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(progress_box, mode="determinate",
                                        maximum=max(1, self.config.epochs))
        self.progress.grid(row=0, column=0, sticky="ew")
        self.status = ttk.Label(progress_box, text="等待开始…", font=("Segoe UI", 9),
                                foreground=COLOR_MUTED)
        self.status.grid(row=1, column=0, sticky="w", pady=(px(4), 0))

        # ------------------------------ 日志 ------------------------------ #
        log_box = ttk.LabelFrame(body, text="训练日志", padding=px(6))
        log_box.grid(row=2, column=0, sticky="nsew")
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self.scroll = ScrollFrame(log_box, width=px(560), height=px(220))
        self.scroll.grid(row=0, column=0, sticky="nsew")
        self.log_label = tk.Label(
            self.scroll.body, text="", justify="left", anchor="nw",
            font=("Consolas", 9), bg=self._log_bg(), fg="#333",
            padx=px(6), pady=px(4))
        self.log_label.pack(fill="both", expand=True)

        # ------------------------------ 按钮 ------------------------------ #
        buttons = ttk.Frame(body)
        buttons.grid(row=3, column=0, sticky="ew", pady=(px(10), 0))
        buttons.columnconfigure(0, weight=1)
        self.hint = ttk.Label(buttons, text="", font=("Segoe UI", 9),
                              foreground=COLOR_MUTED)
        self.hint.grid(row=0, column=0, sticky="w")
        self.cancel_btn = ttk.Button(buttons, text="取消", width=10,
                                     command=self._on_cancel)
        self.cancel_btn.grid(row=0, column=1, padx=(0, px(6)))
        self.save_btn = ttk.Button(buttons, text="保存模型…", width=12,
                                   command=self._on_save)
        self.save_btn.grid(row=0, column=2, padx=(0, px(6)))
        self.save_btn.state(["disabled"])
        self.close_btn = ttk.Button(buttons, text="关闭", width=10,
                                    command=self._on_close)
        self.close_btn.grid(row=0, column=3)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._append_log("已就绪，请点击「开始训练」启动训练。")
        self.center()

    # ------------------------------ 界面小工具 ------------------------------ #
    def _log_bg(self) -> str:
        try:
            return ttk.Style(self).lookup("TFrame", "background") or "#f0f0f0"
        except tk.TclError:
            return "#f0f0f0"

    def _refresh_head(self):
        train_samples = trainer.load_samples(self.workspace, "train")
        test_samples = trainer.load_samples(self.workspace, "test")
        labels = self.workspace.train_labels()
        cards = len(self.architecture["cards"])
        self.head_text.configure(
            text=(f"类别 {len(labels)} 个：{' '.join(labels)}\n"
                  f"训练集 {len(train_samples)} 张，测试集 {len(test_samples)} 张\n"
                  f"网络共 {cards} 层，输入模式 {self.architecture['input_mode']}\n"
                  f"轮数 {self.config.epochs}，批量 {self.config.batch_size}，"
                  f"学习率 {self.config.lr:g}，"
                  f"数据增强 {'开启' if self.config.augment else '关闭'}"))

    def _append_log(self, text: str):
        self._log_lines.append(text)
        if len(self._log_lines) > LOG_LIMIT:
            self._log_lines = self._log_lines[-LOG_LIMIT:]
        self.log_label.configure(text="\n".join(self._log_lines))
        self.after_idle(self.scroll.scroll_to, 1.0)

    # ------------------------------ 生命周期 ------------------------------ #
    def attach_target(self, path) -> None:
        """调用方已经选好了 .pt 路径, 训练完直接写过去, 不再弹另存为。"""
        self.target_path = str(path)
        self.default_path = str(Path(path).parent)
        self.default_name = Path(path).stem

    def result_payload_state_dict(self):
        if self.result_payload is None:
            return None
        return self.result_payload.state_dict

    def _on_cancel(self):
        if self.worker is not None and self.worker.is_alive():
            self.cancel_event.set()
            self.status.configure(text="正在取消…", foreground=COLOR_FAIL)
            self.cancel_btn.state(["disabled"])
        else:
            self._on_close()

    def _on_close(self):
        if self.worker is not None and self.worker.is_alive():
            self.cancel_event.set()
            return
        # 有训练成果就先问一下要不要保存
        if self.result_payload is not None and self.result_payload.ok \
                and self.result is None and self.result_payload.state_dict is not None:
            if not self._ask_save_before_close():
                return
        self.destroy()

    def _ask_save_before_close(self) -> bool:
        from tkinter import messagebox
        answer = messagebox.askyesnocancel(
            "模型尚未保存",
            "训练已经完成，但尚未保存为 .pt 文件。\n\n"
            "选择「是」立即保存；选择「否」放弃并关闭窗口。",
            parent=self)
        if answer is None:
            return False
        if answer:
            self._on_save()
            return self.result is not None
        return True

    def _on_save(self):
        if self.result_payload is None or not self.result_payload.ok:
            return
        path = self.target_path
        if not path:
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                parent=self, title="保存模型为 .pt",
                defaultextension=".pt",
                filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")],
                initialfile=f"{self.default_name}.pt",
                initialdir=self.default_path or None)
            if not path:
                self._append_log("已取消保存（训练结果仍保留在窗口中，可再次点击保存）。")
                return
        result = self.result_payload
        payload = trainer.make_checkpoint(
            {**self.architecture, "num_classes": len(result.labels)},
            result.labels, result.state_dict, result.mean, result.std,
            Path(path).stem, result.accuracy, result.epochs_done,
            extra={"test_passed": result.accuracy})
        try:
            trainer.save_checkpoint(path, payload)
        except OSError as exc:
            from tkinter import messagebox
            self.target_path = None
            messagebox.showerror("保存失败", f"无法写入以下文件：\n{path}\n\n{exc}",
                                 parent=self)
            return
        self.result = {"saved": True, "path": path, "info": payload,
                       "state_dict": result.state_dict}
        self._append_log(f"已保存：{path}")
        self._finish_ui("模型已保存，可以关闭本窗口。", COLOR_PASS)

    # ------------------------------ 训练线程 ------------------------------ #
    def start(self):
        """开始训练(由调用方在 show() 之前或之后调用)。"""
        if self.worker is not None:
            return
        self.cancel_event.clear()
        self._started_at = time.time()
        self.cancel_btn.state(["!disabled"])
        self.save_btn.state(["disabled"])
        self.status.configure(text="正在训练…", foreground=COLOR_MUTED)
        self._append_log("开始训练…")
        self.worker = threading.Thread(target=self._run_training, daemon=True,
                                       name="painter-train")
        self.worker.start()
        self.after(POLL_MS, self._poll)

    def _run_training(self):
        """worker 线程: 只跑训练 + 往队列丢消息。"""
        try:
            result = trainer.train_model(
                self.workspace, self.architecture, self.config,
                on_progress=lambda info: self.queue.put(("progress", info)),
                cancel=self.cancel_event.is_set)
            self.queue.put(("done", result))
        except trainer.TrainError as exc:
            self.queue.put(("error", str(exc)))
        except Exception as exc:                      # noqa: BLE001 - 都要报给用户
            import traceback
            self.queue.put(("error", f"训练过程中发生意外错误：\n{exc}\n\n"
                                     f"{traceback.format_exc(limit=3)}"))

    def _poll(self):
        """主线程: 消费队列消息。返回后决定是否继续轮询。"""
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    self._on_progress(payload)
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._on_error(payload)
        except queue.Empty:
            pass
        if not self.winfo_exists():
            return
        self.after(POLL_MS, self._poll)

    def _on_progress(self, info: dict):
        phase = info.get("phase")
        epoch = info.get("epoch", 0)
        total = info.get("total_epochs", self.config.epochs)
        if phase == "cancelled":
            self.progress.configure(value=epoch)
            return
        if phase == "train":
            self.progress.configure(value=epoch - 1)
            loss = info.get("loss")
            text = (f"第 {epoch}/{total} 轮  第 {info.get('step')}/"
                    f"{info.get('total_steps')} 批")
            if loss is not None:
                text += f"  loss {loss:.4f}"
            self.status.configure(text=text, foreground=COLOR_MUTED)
            return
        # epoch 结束
        self.progress.configure(value=epoch)
        accuracy = info.get("accuracy")
        train_accuracy = info.get("train_accuracy")
        elapsed = time.time() - self._started_at if self._started_at else None
        rate = "" if not elapsed or epoch <= 0 else f"  {elapsed / epoch:.1f}s/轮"
        accuracy_text = "—" if accuracy is None else f"{accuracy * 100:.1f}%"
        self.status.configure(
            text=f"第 {epoch}/{total} 轮完成  通过率 {accuracy_text}"
                 f"（训练集 {train_accuracy * 100:.1f}%）  "
                 f"loss {info.get('loss', 0):.4f}{rate}",
            foreground=COLOR_MUTED)
        self._append_log(
            f"轮 {epoch:>3}/{total}  loss {info.get('loss', 0):.4f}  "
            f"通过率 {accuracy_text}  训练集 {train_accuracy * 100:.1f}%  "
            f"lr {info.get('lr', 0):.2g}")

    def _on_done(self, result: trainer.TrainResult):
        self.result_payload = result
        if result.cancelled:
            self._finish_ui(f"已取消：{result.message}", COLOR_FAIL)
            return
        accuracy_text = "—" if result.accuracy is None else f"{result.accuracy * 100:.1f}%"
        self._finish_ui(
            f"训练完成：共 {result.epochs_done} 轮，用时 {result.seconds:.1f} 秒，"
            f"测试集通过率 {accuracy_text}", COLOR_PASS)
        self._append_log(f"— 完成：{result.message}")
        if result.accuracy is not None:
            self._append_log(f"— 测试集通过率 {accuracy_text}")
        self.save_btn.state(["!disabled"])
        self.cancel_btn.grid_remove()
        self.cancel_btn.state(["!disabled"])
        if self.target_path:
            self.save_btn.configure(text="保存")
            self._on_save()           # 目标路径已经选好了, 自动落盘
        else:
            self._append_log("请点击「保存模型…」选择 .pt 文件的存放位置。")

    def _on_error(self, message: str):
        from tkinter import messagebox
        self._finish_ui("训练失败", COLOR_FAIL)
        self._append_log(f"错误：{message}")
        messagebox.showerror("训练失败", message, parent=self)

    def _finish_ui(self, text: str, color: str):
        self.status.configure(text=text, foreground=color)
        self.save_btn.state(["!disabled"] if (
            self.result_payload and self.result_payload.ok) else ["disabled"])
        self.cancel_btn.state(["!disabled"])
        self.hint.configure(text=text, foreground=color)


def run_training_dialog(parent, workspace, architecture: dict,
                        config: trainer.TrainConfig, metrics=None,
                        name: str = "自定义模型", initial_dir: str = "",
                        auto_start: bool = True):
    """便捷入口: 打开训练对话框并立即开始训练。"""
    dialog = TrainDialog(parent, workspace, architecture, config, metrics=metrics,
                         default_name=name, default_path=initial_dir)
    if auto_start:
        dialog.start()
    return dialog.show()
