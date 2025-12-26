
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
        *,
        stop_sequences: Optional[Dict[str, int]] = None,
        context_processor: Optional[Callable] = None,
        name: Optional[str] = None
    ):
        """
        Args:
            model_name: 模型名称
            config_path: LLM配置文件路径
            enable_thinking: 是否启用思考模式
            stop_sequences: 停止字符串 -> 允许出现次数 n（允许出现 n 次，第 n+1 次出现时截断）
            context_processor: 上下文处理器函数
            name: 模块名称
        """
        super().__init__(name, context_processor)
        self.model_name = model_name
        self.config_path = config_path
        self.enable_thinking = enable_thinking

        if stop_sequences is None:
            stop_sequences = {}

        stop_allowances: Dict[str, int] = {}
        for key, value in stop_sequences.items():
            # 防止stop_sequences中出现空值，导致立即触发stop
            if not key:
                continue
            allowance = int(value)
            if allowance < 0:
                allowance = 0
            stop_allowances[str(key)] = allowance

        self.stop_allowances = stop_allowances
        self.stop_sequences = list(stop_allowances.keys())

        # 创建 LLM Manager
        self.llm_manager = LLMManager(config_path=config_path)
    
    def process(self, messages: List[Dict[str, Any]]) -> Generator[str, None, List[Dict[str, Any]]]:
        try:
            accumulated_text = ""
            stopped = False

            # 每次生成独立统计：stop_seq -> 已出现次数 / 下次扫描的最早起点（用于避免重复计数）
            stop_seen_counts: Dict[str, int] = {s: 0 for s in self.stop_sequences}
            stop_next_search_from: Dict[str, int] = {s: 0 for s in self.stop_sequences}
            
            for chunk in self.llm_manager.chat_stream(
                model_name=self.model_name,
                messages=messages,
                enable_thinking=self.enable_thinking
            ):
                accumulated_text += chunk

                # 检查是否包含停止序列：允许出现 n 次，超过后在第 (n+1) 次出现处截断
                if self.stop_sequences and not stopped:
                    best_stop_index: Optional[int] = None
                    best_stop_seq: Optional[str] = None

                    for stop_seq in self.stop_sequences:
                        allowance = self.stop_allowances.get(stop_seq, 0)
                        seq_len = len(stop_seq)
                        next_from = stop_next_search_from.get(stop_seq, 0)

                        # 为了捕获跨 chunk 边界的匹配，需要向前回看 (len(stop_seq)-1) 个字符
                        scan_from = max(0, next_from - (seq_len - 1))
                        start = scan_from

                        while True:
                            idx = accumulated_text.find(stop_seq, start)
                            if idx == -1:
                                break

                            # 避免对重叠窗口内的旧匹配重复计数
                            if idx >= next_from:
                                stop_seen_counts[stop_seq] = stop_seen_counts.get(stop_seq, 0) + 1
                                if stop_seen_counts[stop_seq] > allowance:
                                    if best_stop_index is None or idx < best_stop_index:
                                        best_stop_index = idx
                                        best_stop_seq = stop_seq
                                    break

                            start = idx + 1

                        # 扫描到当前末尾后，下一轮只需要从“尾部重叠窗口”开始重新检查
                        stop_next_search_from[stop_seq] = max(len(accumulated_text) - (seq_len - 1), 0)

                    if best_stop_index is not None:
                        accumulated_text = accumulated_text[:best_stop_index]
                        stopped = True
                        self.log(
                            "warning",
                            f"Detected stop sequence: '{best_stop_seq}', stopping generation",
                        )

                if stopped:
                    break
                
                yield chunk
                
        except Exception as e:
            self.log("error", f"LLM generation failed: {e}")
            accumulated_text = f"Error: {str(e)}"
            yield accumulated_text

        messages.append({"role": "assistant", "content": accumulated_text})
        return messages
