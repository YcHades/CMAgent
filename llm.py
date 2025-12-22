"""
LLM统一接口框架
基于 OpenAI 风格的统一接口，兼容所有 OpenAI-compatible API
"""

import yaml
import json
import time
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from typing import Dict, Generator, List, Optional, Any, Union, Callable

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """LLM响应对象，包含内容和元数据"""
    content: str  # 返回的文本内容
    model: str  # 使用的模型名称
    prompt_tokens: int  # 输入token数
    completion_tokens: int  # 输出token数
    total_tokens: int  # 总token数
    duration: float  # 耗时（秒）
    timestamp: str  # 时间戳

    def __str__(self) -> str:
        return self.content


@dataclass
class ModelStatistics:
    """单个模型的统计信息"""
    model: str  # 模型名称
    calls: int  # 调用次数
    total_tokens: int  # 总token数
    prompt_tokens: int  # 输入token数
    completion_tokens: int  # 输出token数
    total_duration: float  # 总耗时（秒）
    
    @property
    def avg_tokens_per_call(self) -> float:
        """平均每次调用的token数"""
        return self.total_tokens / self.calls if self.calls > 0 else 0
    
    @property
    def avg_duration_per_call(self) -> float:
        """平均每次调用的耗时"""
        return self.total_duration / self.calls if self.calls > 0 else 0
    
    def __str__(self) -> str:
        return (
            f"Model: {self.model}\n"
            f"  Calls: {self.calls}\n"
            f"  Total Tokens: {self.total_tokens:,} "
            f"(Prompt: {self.prompt_tokens:,}, Completion: {self.completion_tokens:,})\n"
            f"  Avg Tokens/Call: {self.avg_tokens_per_call:.1f}\n"
            f"  Total Duration: {self.total_duration:.2f}s\n"
            f"  Avg Duration/Call: {self.avg_duration_per_call:.3f}s"
        )


@dataclass
class UsageStatistics:
    """使用量统计信息"""
    total_calls: int  # 总调用次数
    total_tokens: int  # 总token数
    total_prompt_tokens: int  # 总输入token数
    total_completion_tokens: int  # 总输出token数
    total_duration: float  # 总耗时（秒）
    by_model: Dict[str, ModelStatistics]  # 按模型的统计
    
    @property
    def avg_tokens_per_call(self) -> float:
        """平均每次调用的token数"""
        return self.total_tokens / self.total_calls if self.total_calls > 0 else 0
    
    @property
    def avg_duration_per_call(self) -> float:
        """平均每次调用的耗时"""
        return self.total_duration / self.total_calls if self.total_calls > 0 else 0
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "LLM Usage Statistics",
            "=" * 60,
            f"Total Calls: {self.total_calls}",
            f"Total Tokens: {self.total_tokens:,} "
            f"(Prompt: {self.total_prompt_tokens:,}, Completion: {self.total_completion_tokens:,})",
            f"Avg Tokens/Call: {self.avg_tokens_per_call:.1f}",
            f"Total Duration: {self.total_duration:.2f}s",
            f"Avg Duration/Call: {self.avg_duration_per_call:.3f}s",
            "\nBy Model:",
            "-" * 60,
        ]
        
        for model_stats in self.by_model.values():
            lines.append(str(model_stats))
            lines.append("-" * 60)
        
        return "\n".join(lines)


class LLMUsageTracker:
    """LLM使用量跟踪器，支持记录、统计和分析"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        初始化日志记录器
        
        Args:
            log_file: 日志文件路径，如果为None则只在内存中记录
        """
        self.log_file = log_file
        self.records: List[Dict[str, Any]] = []
    
    def record_to_log(self, response: LLMResponse):
        """记录一次调用"""
        record = asdict(response)
        self.records.append(record)
        
        # 如果指定了日志文件，追加写入
        if self.log_file:
            self._write_line_to_file(record)
    
    def _write_line_to_file(self, record: Dict[str, Any]):
        """逐行写入日志文件，避免歧义"""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    @staticmethod
    def _compute_statistics(records: List[Dict[str, Any]]) -> Optional[UsageStatistics]:
        """
        从记录列表计算统计信息
        
        Args:
            records: 记录列表
            
        Returns:
            统计信息对象，如果记录为空则返回None
        """
        if not records:
            return None
        
        total_calls = len(records)
        total_tokens = sum(r['total_tokens'] for r in records)
        total_prompt_tokens = sum(r['prompt_tokens'] for r in records)
        total_completion_tokens = sum(r['completion_tokens'] for r in records)
        total_duration = sum(r['duration'] for r in records)
        
        # 按模型统计
        by_model_dict: Dict[str, Dict[str, Any]] = {}
        for record in records:
            model = record['model']
            if model not in by_model_dict:
                by_model_dict[model] = {
                    'calls': 0,
                    'total_tokens': 0,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_duration': 0
                }
            by_model_dict[model]['calls'] += 1
            by_model_dict[model]['total_tokens'] += record['total_tokens']
            by_model_dict[model]['prompt_tokens'] += record['prompt_tokens']
            by_model_dict[model]['completion_tokens'] += record['completion_tokens']
            by_model_dict[model]['total_duration'] += record['duration']
        
        # 转换为ModelStatistics对象
        by_model = {
            model: ModelStatistics(
                model=model,
                calls=stats['calls'],
                total_tokens=stats['total_tokens'],
                prompt_tokens=stats['prompt_tokens'],
                completion_tokens=stats['completion_tokens'],
                total_duration=stats['total_duration']
            )
            for model, stats in by_model_dict.items()
        }
        
        return UsageStatistics(
            total_calls=total_calls,
            total_tokens=total_tokens,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_duration=total_duration,
            by_model=by_model
        )

    def get_statistics(self) -> Optional[UsageStatistics]:
        """获取统计信息"""
        return self._compute_statistics(self.records)
    
    @classmethod
    def get_statistics_from_log_file(cls, log_file: str) -> Optional[UsageStatistics]:
        """
        从日志文件分析统计信息（类方法，可用于其他项目）
        
        Args:
            log_file: 日志文件路径
            
        Returns:
            统计信息对象，如果文件不存在或为空则返回None
        """
        log_path = Path(log_file)
        
        if not log_path.exists():
            logger.error(f"日志文件不存在: {log_file}")
            return None
        
        records = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        logger.warning(f"跳过无效的JSON行: {line[:50]}...")
                        continue
        
        return LLMUsageTracker._compute_statistics(records)


class LLM:
    """OpenAI风格的统一LLM接口"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 usage_logger: Optional[LLMUsageTracker] = None,
                 callback: Optional[Callable[[LLMResponse], None]] = None):
        """
        初始化LLM实例
        
        Args:
            config: 配置字典，包含model, api_key, base_url等参数
            usage_logger: 使用量日志记录器
            callback: 每次调用后的回调函数
        """
        self.config = config
        self.model = config.get('model')
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.temperature = config.get('temperature')
        self.max_tokens = config.get('max_tokens')
        self.top_p = config.get('top_p')

        self.usage_logger = usage_logger
        self.callback = callback
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")

    def _build_chat_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # 将优先使用kwargs（即时调用时传入）中的参数，否则使用模型配置文件中的默认值
        params = {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
        }
        # 过滤 None，对于未传入参数保持默认行为
        params = {k: v for k, v in params.items() if v is not None}

        return params
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             enable_thinking: bool = False,
             **kwargs) -> str:
        """
        对话接口
        
        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
            return_response: 是否返回完整的LLMResponse对象，默认False只返回文本
            **kwargs: 其他参数（如temperature, max_tokens等）
            
        Returns:
            如果return_response=False，返回文本内容(str)
            如果return_response=True，返回LLMResponse对象
        """
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self._build_chat_params(kwargs),
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": enable_thinking}
                }
            )

            c0 = response.choices[0]
            if c0.finish_reason != 'stop':
                logging.warning(f"响应结束理由异常: {c0.finish_reason}")

            # 计算耗时
            duration = time.time() - start_time
            
            # 构造响应对象
            llm_response = LLMResponse(
                content=c0.message.content,
                model=self.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                duration=duration,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            # 记录日志
            if self.usage_logger:
                self.usage_logger.record_to_log(llm_response)
            
            # 执行回调
            if self.callback:
                self.callback(llm_response)
            
            # 返回结果
            return llm_response.content
            
        except Exception as e:
            logger.error(f"未预期的错误: {traceback.format_exc()}")
            raise RuntimeError(f"API调用失败: {e}") from e
    
    def chat_stream(self, 
                    messages: List[Dict[str, str]], 
                    enable_thinking: bool = False,
                    **kwargs) -> Generator[str, None, None]:
        """
        流式对话接口
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Yields:
            流式返回的文本片段
        """
        start_time = time.time()
        content_parts = []
        prompt_tokens = 0
        completion_tokens = 0
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **self._build_chat_params(kwargs),
                stream_options={"include_usage": True},
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": enable_thinking}
                }
            )
            
            for chunk in stream:
                if chunk.choices:
                    c0 = chunk.choices[0]
                    content = c0.delta.content
                    content_parts.append(content)
                    yield content
                    if c0.finish_reason and c0.finish_reason != 'stop':
                        logging.warning(f"响应结束理由异常: {c0.finish_reason}")
                elif chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                else:
                    logging.error(f"收到未知的流式块: {chunk}")
            
            # 计算耗时
            duration = time.time() - start_time
                        
            llm_response = LLMResponse(
                content=''.join(content_parts),
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                duration=duration,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            # 记录日志
            if self.usage_logger:
                self.usage_logger.record_to_log(llm_response)
            
            # 执行回调
            if self.callback:
                self.callback(llm_response)
                
        except Exception as e:
            logger.error(f"未预期的错误: {traceback.format_exc()}")
            raise RuntimeError(f"API调用失败: {e}") from e


class LLMManager:
    """LLM管理器，负责配置加载和模型实例管理"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 log_file: Optional[str] = None,
                 callback: Optional[Callable[[LLMResponse], None]] = None):
        """
        初始化LLM管理器
        
        Args:
            config_path: 配置文件路径，默认为./config.yaml
            log_file: 日志文件路径，如果指定则自动记录所有调用
            callback: 每次调用后的回调函数
        """
        if config_path is None:
            logger.warning("未指定LLM配置文件路径，使用默认路径 './config.yaml'")
            config_path = './config.yaml'
        self.config_path = config_path
        self.config = self._load_config()
        self.models: Dict[str, LLM] = {}
        self.usage_logger = LLMUsageTracker(log_file) if log_file else None
        self.callback = callback
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_file.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_file.suffix}")
        
        return config
    
    def get_model(self, model_name: str) -> LLM:
        """
        获取或创建模型实例
        
        Args:
            model_name: 模型名称，对应配置文件中的key
            
        Returns:
            LLM实例
        """
        # 如果已经创建过，直接返回
        if model_name in self.models:
            return self.models[model_name]
        
        # 从配置中获取模型配置
        llm_config = self.config.get('llm', {}).get(model_name)
        
        if not llm_config:
            raise ValueError(f"配置文件中未找到模型: {model_name}")
        
        # 创建模型实例，传递logger和callback
        model_instance = LLM(llm_config, usage_logger=self.usage_logger, callback=self.callback)
        self.models[model_name] = model_instance
        
        return model_instance
    
    def list_models(self) -> List[str]:
        """列出配置文件中所有可用的模型"""
        return list(self.config.get('llm', {}).keys())
    
    def chat(self, 
             model_name: str, 
             messages: Union[str, List[Dict[str, str]]], 
             enable_thinking: bool = False,
             **kwargs) -> str:
        """
        对话接口
        
        Args:
            model_name: 模型名称
            messages: 消息内容（可以是字符串或消息列表）
            **kwargs: 其他参数
            
        Returns:
            返回的文本
        """
        model = self.get_model(model_name)
        
        # 如果messages是字符串，转换为标准格式
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        return model.chat(messages, **kwargs, enable_thinking=enable_thinking)
    
    def chat_stream(self, 
                    model_name: str, 
                    messages: Union[str, List[Dict[str, str]]], 
                    enable_thinking: bool = False,
                    **kwargs) -> Generator[str, None, None]:
        """
        流式对话接口
        
        Args:
            model_name: 模型名称
            messages: 消息内容（可以是字符串或消息列表）
            **kwargs: 其他参数
            
        Yields:
            流式返回的文本片段
        """
        model = self.get_model(model_name)
        
        # 如果messages是字符串，转换为标准格式
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        yield from model.chat_stream(messages, **kwargs, enable_thinking=enable_thinking)
    
    def get_statistics(self) -> Optional[UsageStatistics]:
        """获取使用统计信息"""
        if not self.usage_logger:
            logger.warning("警告: 未启用日志记录")
            return None
        return self.usage_logger.get_statistics()


# 使用示例
if __name__ == "__main__":
    # 方式1: 不记录日志
    manager = LLMManager('config.yaml')
    response = manager.chat('Qwen3-14B', "你好")
    print(response)
    
    # 方式2: 记录到文件并查看统计信息
    manager_with_log = LLMManager('config.yaml', log_file='./output/llm_usage.log')
    response = manager_with_log.chat('Qwen3-14B', "你好")
    
    # 使用新的dataclass获取统计信息
    stats = manager_with_log.get_statistics()
    if stats:
        print(stats)  # 打印格式化的统计信息
        print(f"\n总调用次数: {stats.total_calls}")
        print(f"总Token数: {stats.total_tokens:,}")
        print(f"平均Token数/调用: {stats.avg_tokens_per_call:.1f}")
    
    # 方式3: 从日志文件分析统计（类方法，可用于其他项目）
    # 无需创建LLMManager实例，直接分析现有日志文件
    stats_from_file = LLMUsageTracker.get_statistics_from_log_file('./output/llm_usage.log')
    if stats_from_file:
        print("\n从日志文件分析的统计信息:")
        print(stats_from_file)
        
        # 访问按模型的统计
        for model_name, model_stats in stats_from_file.by_model.items():
            print(f"\n{model_name} 的详细统计:")
            print(model_stats)
    
    # 方式4: 使用回调函数
    def my_callback(response: LLMResponse):
        print(f"[回调] 模型: {response.model}, Tokens: {response.total_tokens}, 耗时: {response.duration:.2f}s")
    
    manager_with_callback = LLMManager('config.yaml', callback=my_callback)

    for chunk in manager_with_callback.chat_stream('Qwen3-14B', "你好"):
        print(chunk, end='', flush=True)