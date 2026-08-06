from __future__ import annotations
import os
from typing import List, Tuple
import gradio as gr
from service import ChatService


class GradioUI:
    def __init__(self, chat_service: ChatService, program_dir: str) -> None:
        self._svc = chat_service
        self._program_dir = program_dir

    def _get_program_list_text(self) -> str:
        try:
            programs = [n for n in os.listdir(self._program_dir) if os.path.isdir(os.path.join(self._program_dir, n))]
            return "\n".join(f"{i+1}: {p}" for i, p in enumerate(programs))
        except FileNotFoundError:
            return "指定的資料�夾不存在。"
        except Exception as e:  # noqa: BLE001
            return f"發生�錯�誤: {e}"

    def launch(self) -> None:
        def _chat_fn(message: str, history: List[Tuple[str, str]]):
            return self._svc.handle(message, history)

        with gr.Blocks() as iface:
            gr.Markdown(f"## 目前資料庫中的節目有：\n{self._get_program_list_text()}\n\n�請在下方提問：")
            gr.ChatInterface(
                _chat_fn,
                title="Podcast Q&A Assistant",
                description="Ask questions about podcast content, and I'll provide answers based on the retrieved information.",
                # theme="soft",
                examples=[
                    "�還有甚�麼節目與這個主題相關",
                    "�請告�訴我這個節目�討論了哪些主題？",
                    "這集節目中有提到哪些重要的觀點？",
                ],
                # retry_btn="重試",
                # undo_btn="�撤�銷",
                # clear_btn="清除",
            )
        iface.launch(share=True)