"""
Tool-use agent modules.
"""

import json
import re
import traceback
import uuid
from typing import List, Dict, Any, Generator, Optional, Tuple, Callable

from .base import BaseModule
from ..tool_manager import ToolManager


class ToolUseModule(BaseModule):
    """
    工具调用模块：提供工具加载、解析和单次调用能力
    
    支持两种类型的工具：
    1. 本地工具：从 Python 文件/文件夹加载的函数
    2. MCP 工具：通过 MCP 服务器提供的远程工具（HTTP/STDIO）
    
    特性：
    - MCP 工具名称自动添加命名空间前缀（避免冲突）
    - 支持同时配置多个 MCP 服务器
    - 懒加载机制：仅在首次使用时加载工具
    - 自动参数校验和结果序列化
    """
    
    def __init__(
        self,
        mcp_config: Optional[Dict[str, Any]] = None,
        local_tools_folder: Optional[str] = None,
        local_files_selector: Optional[List[str]] = None,
        tool_parse_template: str = r"<tool_call>\n(.*?)\n</tool_call>",
        context_processor: Optional[Callable] = None,
        name: Optional[str] = None
    ):
        """
        Args:
            mcp_config: MCP 配置字典，格式：
                {
                    "mcpServers": {
                        "server_name": {
                            "command": "uvx",
                            "args": [...],
                            "transport": "stdio"
                        }
                    }
                }
            local_tools_folder: 本地工具文件夹路径
            local_files_selector: 要加载的本地工具文件名列表（不含 .py 后缀）
            tool_parse_template: 工具调用解析正则表达式
            context_processor: 上下文处理器函数
            name: 模块名称
            
        注意：
            MCP 工具会自动添加服务器别名前缀，如: server_name_tool_name
        """
        super().__init__(name or "ToolUseModule", context_processor)
        
        self.mcp_config = mcp_config
        self.local_tools_folder = local_tools_folder
        self.local_files_selector = local_files_selector
        self.tool_parse_template = tool_parse_template
        
        self.tool_manager = ToolManager()
        self._tools_loaded = False

    def load_tools(self):
        """加载工具：从本地文件夹和 MCP 配置
        
        注意：MCP 工具名称会自动添加命名空间前缀，如: server_name_tool_name
        """
        if self._tools_loaded:
            return
        
        # 加载本地工具
        if self.local_tools_folder:
            self.tool_manager.load_from_folder(
                tools_folder=self.local_tools_folder,
                local_files_selector=self.local_files_selector
            )
        
        # 加载 MCP 工具配置（工具会在首次调用时才实际加载）
        if self.mcp_config:
            self.tool_manager.load_from_mcp_config(self.mcp_config)
        
        self._tools_loaded = True
        count = self.tool_manager.get_tool_count()
        mcp_servers = len(self.tool_manager.mcp_configs) if hasattr(self.tool_manager, 'mcp_configs') else 0
        if mcp_servers > 0:
            self.log("info", f"Loaded {count.get('local', 0)} local tools, registered {mcp_servers} MCP server(s) (tools will load on first use)")
        else:
            self.log("info", f"Loaded {sum(count.values())} tools: {count}")
    
    def tool_parser(self, text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        默认工具调用解析器：解析所有 <tool_call> 格式的工具调用
        
        Returns:
            List of (tool_name, tool_args) tuples，如果没有找到返回空列表
        """
        tool_calls = []
        # 使用 finditer 查找所有匹配
        for tool_match in re.finditer(self.tool_parse_template, text, re.DOTALL):
            tool_call = tool_match.group(1)
            try:
                tool_data = json.loads(tool_call)
                tool_name = tool_data.get("name")
                tool_args = tool_data.get("arguments")
                if tool_name and tool_args is not None:
                    tool_calls.append((tool_name, tool_args))
            except json.JSONDecodeError:
                self.log("error", f"Invalid tool call format: {tool_call}")
        return tool_calls
    
    def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """调用工具并返回结果
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            
        Returns:
            工具调用结果，如果失败则返回错误信息
        """
        if not self._tools_loaded:
            self.load_tools()
        
        try:
            result = self.tool_manager.call_tool(tool_name, tool_args)
            return result
            
        except ValueError as e:
            # 工具不存在或参数错误
            error_msg = f"Tool error: {str(e)}"
            self.log("error", f"Tool '{tool_name}' validation failed: {e}")
            return error_msg
            
        except RuntimeError as e:
            # MCP 连接或运行时错误
            error_msg = str(e)
            if "MCP server connection lost" in error_msg or "failed to connect" in error_msg:
                self.log("warning", f"MCP tool '{tool_name}' connection issue: {e}")
                return (
                    f"Tool '{tool_name}' temporarily unavailable due to connection issues. "
                    f"The MCP server may need to restart. Please try again or use an alternative approach."
                )
            else:
                self.log("error", f"Tool '{tool_name}' runtime error: {e}")
                return f"Tool error: {str(e)}"
                
        except Exception as e:
            # 其他未预期的错误
            self.log("error", f"Unexpected error calling tool '{tool_name}': {e}\n{traceback.format_exc()}")
            return f"Unexpected tool error: {str(e)}"
    
    def process(self, messages: List[Dict[str, Any]]) -> Generator[str, None, List[Dict[str, Any]]]:
        """
        解析最后一条消息中的所有工具调用并执行
        """
        assert len(messages) > 0, "No messages provided to ToolUseModule"
        
        # 解析最后一条消息中的所有工具调用
        last_message = messages[-1].get("content", "")
        self.log("info", f"Parsing tool calls in last message: {last_message}")
        parsed_tool_calls = self.tool_parser(last_message)
        
        if not parsed_tool_calls:
            self.log("warning", f"No tool call found in last message")
            # 设置元数据：没有找到 tool_call
            self._metadata = {'has_tool_call': False}
            return messages
        
        # 设置元数据：找到了 tool_call
        self._metadata = {
            'has_tool_call': True,
            'tool_calls_count': len(parsed_tool_calls),
            'tool_names': [name for name, _ in parsed_tool_calls]
        }
        
        # 执行所有工具并收集结果
        tool_call_results = []
        for tool_name, tool_args in parsed_tool_calls:
            tool_result = self.call_tool(tool_name, tool_args)
            tool_call_results.append({
                'name': tool_name,
                'args': tool_args,
                'result': tool_result,
                'id': f"chatcmpl-tool-{uuid.uuid4().hex}"
            })
            # 流式输出每个工具结果
            yield tool_result
        
        # 删除最后一条消息中的tool_call格式内容
        content_without_tool_calls = re.sub(self.tool_parse_template, '', last_message, flags=re.DOTALL).strip()

        # 构造tool_call之后的assistant消息
        messages[-1] = {
            'role': 'assistant',
            'content': content_without_tool_calls,
            'tool_calls': [
                {
                    'id': call['id'],
                    'function': {
                        'name': call['name'],
                        'arguments': json.dumps(call['args']) if isinstance(call['args'], dict) else call['args']
                    },
                    'type': 'function'
                }
                for call in tool_call_results
            ]
        }
        
        # 为每个工具调用添加一个 tool role 消息
        for call in tool_call_results:
            messages.append({
                'role': 'tool',
                'content': call['result'],
                'tool_call_id': call['id']
            })
        
        return messages
    
    def get_tools_description(self) -> str:
        """获取工具描述,用于注入到 prompt"""
        if not self._tools_loaded:
            self.load_tools()
        
        return self.tool_manager.get_tools_description()


class ToolUseLoopModule(BaseModule):
    """
    工具使用循环模块：自动检测并执行工具调用的循环
    
    组合 LLMModule 和 ToolUseModule，实现完整的循环：
    LLM 生成 -> 解析工具调用 -> 执行工具 -> LLM 继续响应 -> ...
    
    注意：这不是标准的 ReAct 模式（缺少显式的 Thought/Observation 结构）
    """
    
    def __init__(
        self,
        model_name: str,
        config_path: Optional[str] = None,
        mcp_config: Optional[Dict[str, Any]] = None,
        local_tools_folder: Optional[str] = None,
        local_files_selector: Optional[List[str]] = None,
        max_iterations: int = 10,
        context_processor: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
        name: Optional[str] = None
    ):
        """
        Args:
            model_name: 模型名称
            config_path: LLM 配置文件路径
            mcp_config: MCP 配置字典
            local_tools_folder: 本地工具文件夹路径
            local_files_selector: 本地工具文件名列表
            max_iterations: 最大 ReAct 循环迭代次数
            context_processor: 上下文处理器函数
            name: 模块名称
        """
        from .llm_module import LLMModule

        super().__init__(name or "ToolUseLoopModule", context_processor)
        
        self.model_name = model_name
        self.config_path = config_path
        self.max_iterations = max_iterations
        
        # 组合 LLM 和 Tool 模块
        self.llm_module = LLMModule(model_name, config_path)
        self.tool_module = ToolUseModule(
            mcp_config=mcp_config,
            local_tools_folder=local_tools_folder,
            local_files_selector=local_files_selector,
        )
    
    def process(self, messages: List[Dict[str, Any]]) -> Generator[str, None, List[Dict[str, Any]]]:
        """
        执行工具使用循环：LLM 生成 -> 工具执行 -> LLM 继续
        
        主循环条件：只要模型输出中有工具调用，就继续推理
        安全阀：max_iterations 防止无限循环
        """
        iterations = 0
        has_tool_call = True
        current_messages = messages
        
        while has_tool_call:
            # 安全阀：检查是否超过最大迭代次数
            if iterations >= self.max_iterations:
                self.log("warning", f"Reached max iterations ({self.max_iterations})")
                current_messages.append({
                    "role": "assistant",
                    "content": "[SYSTEM WARNING] Reached maximum ReAct iterations. Ending process."
                })
                break
            
            iterations += 1
            
            # 1. Reasoning: 调用 LLM 生成思考和行动            
            for chunk in self.llm_module(current_messages):
                if chunk.content:
                    yield chunk.content
                if chunk.finished:
                    current_messages = chunk.messages

            # 2. Acting: 检查是否有工具调用并执行
            has_tool_call = False
            for chunk in self.tool_module(current_messages):
                if chunk.content:
                    yield chunk.content
                if chunk.finished:
                    current_messages = chunk.messages
                    if chunk.metadata and chunk.metadata.get('has_tool_call'):
                        has_tool_call = True
            
            # 如果没有工具调用，下一轮循环会自然结束
            if not has_tool_call:
                self.log("info", "No tool call found, ending tool use loop")
        
        return current_messages  # 返回最终的消息列表
