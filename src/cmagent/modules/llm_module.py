
from typing import List, Dict, Any, Generator, Optional, Callable, Union

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
        stop_sequences: List[str] = [],
        context_processor: Optional[Callable] = None,
        name: Optional[str] = None
    ):
        """
        Args:
            model_name: 模型名称
            config_path: LLM配置文件路径
            enable_thinking: 是否启用思考模式
            stop_sequences: 停止字符串列表，检测到这些字符串时提前终止输出
            context_processor: 上下文处理器函数
            name: 模块名称
        """
        super().__init__(name, context_processor)
        self.model_name = model_name
        self.config_path = config_path
        self.enable_thinking = enable_thinking
        self.stop_sequences = stop_sequences

        # 创建 LLM Manager
        self.llm_manager = LLMManager(config_path=config_path)
    
    def process(self, messages: List[Dict[str, Any]]) -> Generator[str, None, List[Dict[str, Any]]]:
        try:
            accumulated_text = ""
            stopped = False
            
            for chunk in self.llm_manager.chat_stream(
                model_name=self.model_name,
                messages=messages,
                enable_thinking=self.enable_thinking
            ):
                accumulated_text += chunk
                
                # 检查是否包含停止序列
                if self.stop_sequences and not stopped:
                    for stop_seq in self.stop_sequences:
                        if stop_seq in accumulated_text:
                            # 找到停止序列，截断到该位置
                            stop_index = accumulated_text.index(stop_seq)
                            accumulated_text = accumulated_text[:stop_index]
                            stopped = True
                            self.log("warning", f"Detected stop sequence: '{stop_seq}', stopping generation")
                            break
                
                # 如果已停止，不再 yield
                if stopped:
                    break
                
                yield chunk
                
        except Exception as e:
            self.log("error", f"LLM generation failed: {e}")
            accumulated_text = f"Error: {str(e)}"
            yield accumulated_text

        messages.append({"role": "assistant", "content": accumulated_text})
        return messages
