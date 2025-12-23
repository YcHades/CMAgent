
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator, Optional, Callable


@dataclass
class ModuleChunk:
    """模块输出的数据块
    
    设计原则：
    - content: 始终是字符串，用于流式显示
    - messages: 完整的消息列表，只在 finished=True 时有值
    - metadata: 额外的元数据信息
    """
    content: str = ""  # 流式文本内容
    finished: bool = False  # 是否是最后一个 chunk
    messages: Optional[List[Dict[str, Any]]] = None  # 最终的消息列表
    metadata: Optional[Dict[str, Any]] = None  # 用于传递额外信息


class BaseModule(ABC):
    """
    基础模块抽象类
    """
    
    def __init__(self, name: Optional[str] = None, context_processor: Optional[Callable] = None):
        """
        Args:
            name: 模块名称,用于日志和调试
            context_processor: 上下文处理器函数，用于自定义初始上下文（如添加系统消息、转换格式等）
        """
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.name)
        self.context: List[Dict[str, Any]] = []  # 模块隔离的临时上下文
        self.context_processor = context_processor  # 可选的上下文处理器

    @abstractmethod
    def process(self, messages: List[Dict[str, Any]]) -> Generator[str, None, List[Dict[str, Any]]]:
        """
        消息处理逻辑

        子类必须实现此方法，负责流式生成中间结果字符串，并返回最终的消息列表

        Args:
            messages: 输入消息列表（可以是副本，随意修改）

        Yields:
            str: 中间处理结果的文本片段
            
        Returns:
            List[Dict[str, Any]]: 最终的完整消息列表
        """
        pass
    
    def __call__(self, messages: List[Dict[str, Any]]) -> Generator[ModuleChunk, None, None]:
        """
        模块调用入口：messages in, chunks + messages out (流式)
        
        Args:
            messages: 输入消息列表 (OpenAI 格式)
            
        Yields:
            - 中间 chunk: str - 用于实时聊天显示
            - 结尾 chunk: List[Dict] - 完整的消息列表
        """
        if self.context_processor:
            processed_messages = self.context_processor(messages.copy())
        else:
            processed_messages = messages.copy()
        
        # 创建 generator
        generator = self.process(processed_messages)
        
        # 流式输出中间 chunk
        try:
            while True:
                chunk_text = next(generator)
                yield ModuleChunk(
                    content=chunk_text,
                    finished=False,
                    metadata=getattr(self, '_metadata', None)
                )
        except StopIteration as e:
            # 从 generator 返回值获取最终 messages
            final_messages = e.value if e.value is not None else processed_messages
            self.context = final_messages
            
            # 结尾 chunk 包含完整消息列表
            yield ModuleChunk(
                content="",  # 最后一个 chunk 可以没有文本内容
                finished=True,
                messages=self.context,
                metadata=getattr(self, '_metadata', None)
            )

    def log(self, level: str, message: str):
        """统一日志接口"""
        getattr(self.logger, level.lower())(message)
