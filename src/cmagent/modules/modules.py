"""
Module Framework for Agent Construction

核心设计理念：
1. 以模块为最小单元构建 agent
2. 所有模块统一接口：messages in, messages out (流式)
3. 模块可以嵌套组合，agent 本身也是一个模块
4. 调用者只关心输入输出，不关心内部实现
5. 使用流式调用，支持实时输出
6. 每个模块有独立的上下文记忆
"""

import re
import json
import traceback
import uuid
from typing import List, Dict, Any, Generator, Optional, Tuple, Callable

from .base import BaseModule, ModuleChunk
from ..llm import LLMManager
from ..tool_manager import ToolManager
from ..utils import extract_json_codeblock


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
            yield tool_result + "\n"
        
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


class PlanModule(BaseModule):
    def __init__(
        self, 
        model_name: str,
        config_path: str = None,
        context_processor = None,
        name = None, 
    ):
        super().__init__(name, context_processor)
        self.llm_module = LLMModule(
            model_name=model_name,
            config_path=config_path,
            enable_thinking=True
        )
        self.plan_prompt = """
你是一个任务规划专家（Planner Agent）。

你的职责是：  
在不执行任何具体工作的前提下，将给定的复杂任务拆解为一组**线性、可执行、彼此解耦**的子步骤。

请遵循以下原则进行拆解：
1. 每个步骤应当是**可以独立完成**的具体任务，而不是抽象阶段
2. 步骤之间应具有**明确的先后依赖关系**
3. 每个步骤的目标应当**清晰、可验证、无歧义**
4. 不要在步骤中提前完成后续步骤的内容

请按照以下 JSON 格式输出执行计划：

```json
{
  "<步骤1名称>": "<该步骤需要完成的具体目标>",
  "<步骤2名称>": "<该步骤需要完成的具体目标>",
  "...": "...",
  "<步骤n名称>": "<该步骤需要完成的具体目标>"
}

"""     

    def process(self, messages):
        task = messages[-1]["content"]
        current_messages = [
            {"role": "system", "content": self.plan_prompt},
            {"role": "user", "content": f"<task>{task}</task>"}
        ]
        for chunk in self.llm_module(current_messages):
            if chunk.content:
                yield chunk.content
            if chunk.finished:
                current_messages = chunk.messages
                plan_text = chunk.messages[-1]["content"]
                self._metadata = {
                    'plan': extract_json_codeblock(plan_text)[0]
                }
        return current_messages
    

class PlanAndSolveModule(BaseModule):
    """
    规划与解决模块：先生成计划，再逐步执行计划中的每一步
    
    适用于需要多步推理和操作的任务
    
    核心改进：
    1. 每个step独立上下文，避免历史膨胀
    2. solver不感知完整plan，避免提前执行
    3. 步骤间通过结构化handoff传递，而非完整对话
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
            max_iterations: solver 模块最大迭代次数
            context_processor: 上下文处理器函数
            name: 模块名称
        """
        super().__init__(name, context_processor)

        self.planner_module = PlanModule(
            model_name=model_name,
            config_path=config_path
        )
        self.solver_module = ToolUseLoopModule(
            model_name=model_name,
            config_path=config_path,
            mcp_config=mcp_config,
            local_tools_folder=local_tools_folder,
            local_files_selector=local_files_selector,
            max_iterations=max_iterations,
        )
    
    def _extract_global_brief(self, messages: List[Dict[str, Any]]) -> str:
        """
        从用户输入中抽取全局目标简要描述
        
        Args:
            messages: 输入消息列表
            
        Returns:
            全局目标描述字符串
        """
        # 简化实现：取最后一条user message
        # 更稳健的做法：可以调用summarizer模块生成摘要
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
        return last_user["content"] if last_user else ""
    
    def _build_step_messages(
        self, 
        global_brief: str, 
        step: str, 
        goal: str, 
        handoff: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        为单个步骤构造最小执行上下文
        
        Args:
            global_brief: 全局任务目标（精简版）
            step: 当前步骤名称
            goal: 当前步骤目标
            handoff: 前序步骤的结构化交接物
            
        Returns:
            最小化的消息列表，仅包含执行本步骤所需的信息
        """
        step_messages = [
            {
                "role": "system", 
                "content": "你是一个执行专家。只执行当前给定的步骤，不要规划后续步骤，不要复述完整计划。专注于完成当前步骤的目标并产出明确的结果。"
            },
        ]
        
        # 添加全局目标
        if global_brief:
            step_messages.append({
                "role": "user", 
                "content": f"<global_goal>\n{global_brief}\n</global_goal>"
            })
        
        # 添加前序步骤的交接物（结构化摘要）
        if handoff:
            facts = handoff.get("facts", [])
            decisions = handoff.get("decisions", [])
            open_questions = handoff.get("open_questions", [])
            
            handoff_parts = ["<handoff>"]
            if facts:
                handoff_parts.append("**前序步骤产出的事实：**")
                for fact in facts:
                    handoff_parts.append(f"- {fact}")
            if decisions:
                handoff_parts.append("\n**前序步骤做出的决策：**")
                for decision in decisions:
                    handoff_parts.append(f"- {decision}")
            if open_questions:
                handoff_parts.append("\n**前序步骤遗留的问题：**")
                for question in open_questions:
                    handoff_parts.append(f"- {question}")
            handoff_parts.append("</handoff>")
            
            step_messages.append({
                "role": "user",
                "content": "\n".join(handoff_parts)
            })
        
        # 添加当前步骤指令
        step_messages.append({
            "role": "user",
            "content": f"现在请单独执行以下步骤，并产出该步骤的结果：\n<step>\n{step}: {goal}\n</step>"
        })
        
        return step_messages
    
    def _summarize_step_result(self, step: str, result_text: str) -> Dict[str, Any]:
        """
        将步骤执行结果摘要化为结构化的handoff
        
        Args:
            step: 步骤名称
            result_text: 步骤的完整输出文本
            
        Returns:
            结构化的交接物 {facts, decisions, open_questions}
        """
        # 简化实现：基于规则的摘要（零额外模型调用）
        # 更稳健的做法：调用summarizer模块让模型输出结构化JSON
        
        # 限制文本长度，避免后续步骤上下文过长
        truncated_text = result_text.strip()[:800] if result_text else ""
        
        # 提取关键信息（简化规则）
        facts = []
        if truncated_text:
            # 如果结果中包含明确的结论性语句，提取之
            facts.append(f"{step} 完成，结果摘要：{truncated_text}")
        
        return {
            "facts": facts,
            "decisions": [],
            "open_questions": []
        }
    
    def process(self, messages: List[Dict[str, Any]]) -> Generator[str, None, List[Dict[str, Any]]]:
        # 1. 生成计划
        for chunk in self.planner_module(messages):
            if chunk.content:
                yield chunk.content
            if chunk.finished:
                steps = chunk.metadata.get('plan')
        
        self.log("info", f"Generated plan steps: {steps}")
        
        plan_display = "\n\n<plan>\n" + "\n".join([f"{s}: {g}" for s, g in steps.items()]) + "\n</plan>\n\n"
        yield plan_display
        
        # 2. 逐步执行计划（每步独立上下文）
        global_brief = self._extract_global_brief(messages)
        handoff: Dict[str, Any] = {}  # 前序步骤的交接物
        final_messages = list(messages)  # 用于记录最终结果
        
        for step, goal in steps.items():
            # 为当前步骤构造最小化执行上下文
            step_messages = self._build_step_messages(global_brief, step, goal, handoff)
            
            # 执行当前步骤
            step_result_text = ""
            for chunk in self.solver_module(step_messages):
                if chunk.content:
                    yield chunk.content
                    step_result_text += chunk.content
                if chunk.finished:
                    pass

            # 将本步骤结果摘要化，供下一步使用
            handoff = self._summarize_step_result(step, step_result_text)
            
            # 将步骤结果记录到最终消息列表（可选，用于完整记录）
            final_messages.append({
                "role": "assistant",
                "content": f"<step_result step='{step}'>\n{step_result_text}\n</step_result>"
            })
        
        return final_messages


# TODO: need fully test
class CompositeModule(BaseModule):
    """
    组合模块：将多个模块组合成一个模块
    
    支持顺序执行和循环执行
    """
    
    def __init__(
        self,
        modules: List[BaseModule],
        execution_mode: str = "sequential",
        stop_condition: Optional[Callable] = None,
        max_iterations: int = 10,
        context_processor: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
        name: Optional[str] = None
    ):
        """
        Args:
            modules: 子模块列表
            execution_mode: 执行模式 ('sequential', 'loop')
            stop_condition: 停止条件函数,接收 messages 返回 bool
            max_iterations: 循环模式下的最大迭代次数
            context_processor: 上下文处理器函数
            name: 模块名称
        """
        super().__init__(name, context_processor=context_processor)
        self.modules = modules
        self.execution_mode = execution_mode
        self.stop_condition = stop_condition
        self.max_iterations = max_iterations
    
    def process(self, messages: List[Dict[str, Any]]) -> Generator[str, None, List[Dict[str, Any]]]:
        """
        根据执行模式处理消息
        
        Args:
            messages: 输入消息列表
            
        Yields:
            str: 流式输出的文本片段
            
        Returns:
            List[Dict[str, Any]]: 最终的完整消息列表
        """
        if self.execution_mode == "sequential":
            return (yield from self._sequential_process(messages))
        elif self.execution_mode == "loop":
            return (yield from self._loop_process(messages))
        else:
            self.log("error", f"Unknown execution mode: {self.execution_mode}")
            return messages
    
    def _sequential_process(
        self,
        messages: List[Dict[str, Any]]
    ) -> Generator[str, None, List[Dict[str, Any]]]:
        """顺序执行所有子模块"""
        current_messages = messages
        
        for module in self.modules:
            for chunk in module(current_messages):
                if chunk.content:
                    # 流式输出文本
                    yield chunk.content
                if chunk.finished:
                    # 更新消息列表
                    current_messages = chunk.messages
        
        return current_messages
    
    def _loop_process(
        self,
        messages: List[Dict[str, Any]]
    ) -> Generator[str, None, List[Dict[str, Any]]]:
        """循环执行直到满足停止条件"""
        current_messages = messages
        iterations = 0
        
        while iterations < self.max_iterations:
            # 检查停止条件
            if self.stop_condition and self.stop_condition(current_messages):
                break
            
            # 执行所有子模块
            for module in self.modules:
                for chunk in module(current_messages):
                    if chunk.content:
                        yield chunk.content
                    if chunk.finished:
                        current_messages = chunk.messages
            
            iterations += 1
        
        if iterations >= self.max_iterations:
            self.log("warning", f"Reached max iterations ({self.max_iterations})")
        
        return current_messages
    

def module_output_printer(generator: Generator[ModuleChunk, None, None]):
    """
    辅助函数：打印模块的流式输出结果
    """
    for chunk in generator:
        # 打印流式内容
        if chunk.content:
            print(chunk.content, end='', flush=True)
        
        # 最后打印完整的消息列表
        if chunk.finished:
            print("\n\n=== Final Context ===")
            for i, msg in enumerate(chunk.messages, 1):
                print(f"\n[{i}] {msg['role'].upper()}:")
                print(json.dumps(msg, indent=2, ensure_ascii=False))


def main():
    # llm_module = LLMModule(
    #     model_name="Qwen3-14B",
    #     name="TestLLMModule"
    # )

    # module_output_printer(llm_module([
    #     {"role": "user", "content": "你好"}
    # ]))

    # tooluse_module = ToolUseModule(
    #     local_tools_folder="toolbox",
    #     mcp_config={
    #         "mcpServers": {
    #             "browser_use": {
    #                 "command": "uvx",
    #                 "args": ["--from", "browser-use[cli]", "browser-use", "--mcp"],
    #                 "transport": "stdio"
    #             }
    #         }
    #     },
    #     name="TestToolUseModule"
    # )
    # print(tooluse_module.get_tools_description())
    # module_output_printer(tooluse_module([
    #     {"role": "assistant", "content": "<tool_call>\n{\"name\": \"greet\", \"arguments\": {\"content\": \"你好，朋友！\"}}\n</tool_call>"}
    # ]))

    tool_use_loop_module = ToolUseLoopModule(
        model_name="Qwen3-14B",
        local_tools_folder="toolbox",
        mcp_config={
            "mcpServers": {
                "baidu-map": {
                  "command": "npx",
                  "args": ["-y", "@baidumap/mcp-server-baidu-map"
                  ],
                  "env": {
                    "BAIDU_MAP_API_KEY": "rakR78VnBT9ibC98jwppeUL8M94VnjEG"
                  }
                },
                "math": {
                  "url": "http://localhost:8000/sse",
                  "transport": "sse"
                },
                "text": {
                  "url": "http://localhost:8001/sse",
                  "transport": "sse"
                },
                "data": {
                  "url": "http://localhost:8002/sse",
                  "transport": "sse"
                }
            }
        },
        name="TestToolUseLoopModule"
    )

    module_output_printer(tool_use_loop_module([
        {"role": "system", "content": 
            f"你是一个Agent，可以通过<tool_call>\n{{'name': ..., 'arguments': ...}}\n</tool_call>格式调用工具\n你可以使用的工具如下：\n{tool_use_loop_module.tool_module.get_tools_description()}\n需要时使用工具。"
        },
        {"role": "user", "content": "获取长沙今天的天气"}
    ]))

    # plan_and_solve_module = PlanAndSolveModule(
    #     model_name="Qwen3-14B",
    #     local_tools_folder="toolbox",
    #     mcp_config={
    #         "mcpServers": {
    #             "browser_use": {
    #                 "command": "uvx",
    #                 "args": ["--from", "browser-use[cli]", "browser-use", "--mcp"],
    #                 "transport": "stdio"
    #             }
    #         }
    #     },
    #     name="TestPlanAndSolveModule"
    # )
    # module_output_printer(plan_and_solve_module([
    #     {"role": "system", "content": 
    #         f"你是一个Agent，可以通过<tool_call>\n{{'name': ..., 'arguments': ...\}}\n</tool_call>格式调用工具\n你可以使用的工具如下：\n{tooluse_module.get_tools_description()}\n请你完成用户的任务，需要时使用工具。"
    #     },
    #     {"role": "user", "content": "你好Agent，你可以向我打招呼并介绍自己吗？"}
    # ]))


if __name__ == "__main__":
    main()
