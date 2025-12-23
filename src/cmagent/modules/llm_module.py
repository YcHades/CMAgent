
from typing import List, Dict, Any, Generator, Optional, Callable

from .base import BaseModule
from ..llm import LLMManager


class LLMModule(BaseModule):
    """
    LLM 模块
    """
    def __init__(
        self,
        model_name: str,
        config_path: str = None,
        enable_thinking: bool = False,
        context_processor: Optional[Callable] = None,
        name: Optional[str] = None
    ):
        """
        Args:
            model_name: 模型名称
            config_path: LLM配置文件路径
            context_processor: 上下文处理器函数
            name: 模块名称
        """
        super().__init__(name, context_processor)
        self.model_name = model_name
        self.config_path = config_path
        self.enable_thinking = enable_thinking
        
        # 创建 LLM Manager
        self.llm_manager = LLMManager(config_path=config_path)
    
    def process(self, messages: List[Dict[str, Any]]) -> Generator[str, None, List[Dict[str, Any]]]:
        try:
            accumulated_text = ""
            for chunk in self.llm_manager.chat_stream(
                model_name=self.model_name,
                messages=messages,
                enable_thinking=self.enable_thinking
            ):
                accumulated_text += chunk
                yield chunk
        except Exception as e:
            self.log("error", f"LLM generation failed: {e}")
            accumulated_text = f"Error: {str(e)}"
            yield accumulated_text

        messages.append({"role": "assistant", "content": accumulated_text})
        return messages
