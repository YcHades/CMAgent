"""
ReACT Agent Module - Reasoning and Acting
基于 ReACT 模式实现的标准 Agent：Thought -> Action -> Observation 循环

设计原则：
- 组合优于继承：复用 LLMModule 和 ToolUseModule
- 单一职责：只负责 ReACT 循环编排和提示词管理
- 避免代码重复：工具管理逻辑由 ToolUseModule 处理
"""

from typing import List, Dict, Any, Generator, Optional, Callable

from .base import BaseModule
from .llm_module import LLMModule
from .tool_use_module import ToolUseModule


class ReActModule(BaseModule):
    """
    ReACT Agent 模块：实现标准的 Thought-Action-Observation 循环
    
    工作流程：
    1. Thought: LLM 分析当前情况，决定下一步行动
    2. Action: 解析并执行工具调用
    3. Observation: 将工具结果反馈给 LLM
    4. 重复直到 LLM 给出最终答案（Final Answer）
    
    设计特点：
    - 组合 LLMModule 和 ToolUseModule，避免重复代码
    - 只负责 ReACT 循环编排和提示词管理
    - 显式的思考-行动-观察结构
    """
    
    # ReACT 提示模板
    REACT_SYSTEM_PROMPT = """You are a helpful assistant that uses the ReACT framework.

STRICT FORMAT:
Thought: ...
Action:
<tool_call>
{{"name": "...", "arguments": {{...}}}}
</tool_call>
[Optionally more <tool_call> blocks...]
Observation: ...

HARD RULES:
1) Action MUST contain one or more <tool_call>...</tool_call> blocks and NOTHING ELSE.
2) Each <tool_call> must be valid JSON with keys: name, arguments.
3) Tool calls are executed in the order written.
4) Wait for Observation before the next Thought.
5) If Action formatting is invalid, rewrite the Action immediately, no explanation.

Available tools:
{tools_description}
"""

    
    def __init__(
        self,
        model_name: str,
        config_path: Optional[str] = None,
        mcp_config: Optional[Dict[str, Any]] = None,
        local_tools_folder: Optional[str] = None,
        local_files_selector: Optional[List[str]] = None,
        max_iterations: int = 10,
        enable_thinking: bool = False,
        context_processor: Optional[Callable] = None,
        name: Optional[str] = None
    ):
        """
        Args:
            model_name: LLM 模型名称
            config_path: LLM 配置文件路径
            mcp_config: MCP 工具配置
            local_tools_folder: 本地工具文件夹路径
            local_files_selector: 要加载的本地工具文件列表
            max_iterations: 最大迭代次数（防止无限循环）
            enable_thinking: 是否启用 LLM 思考模式
            context_processor: 上下文处理器函数
            name: 模块名称
        """
        super().__init__(name or "ReActModule", context_processor)
        
        self.max_iterations = max_iterations
        
        # 组合 LLMModule
        self.llm_module = LLMModule(
            model_name=model_name,
            config_path=config_path,
            enable_thinking=enable_thinking,
            stop_sequences=["Observation"],
            name=f"{self.name}.LLM"
        )
        
        # 组合 ToolUseModule
        self.tool_module = ToolUseModule(
            mcp_config=mcp_config,
            local_tools_folder=local_tools_folder,
            local_files_selector=local_files_selector,
            name=f"{self.name}.Tool"
        )
    
    def get_tools_description(self) -> str:
        """获取工具描述（委托给 ToolUseModule）"""
        return self.tool_module.get_tools_description()
    
    def is_final_answer(self, text: str) -> bool:
        """检查是否包含最终答案"""
        return "Final Answer:" in text or "最终答案：" in text
    
    def process(self, messages: List[Dict[str, Any]]) -> Generator[str, None, List[Dict[str, Any]]]:
        """
        ReACT 主循环处理逻辑
        
        流程：
        1. 添加系统提示（包含工具描述）
        2. 循环：
           - LLM 生成 Thought + Action
           - 检查是否有工具调用
           - 执行工具并添加 Observation
           - 检查是否到达最终答案
        3. 返回完整对话历史
        """
        # 准备系统提示
        tools_description = self.get_tools_description()
        system_prompt = self.REACT_SYSTEM_PROMPT.format(tools_description=tools_description)
        
        # 添加系统消息（如果不存在）
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            # 更新现有系统消息
            messages[0]["content"] = system_prompt
        
        self.log("info", f"Starting ReACT loop (max {self.max_iterations} iterations)")
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            self.log("info", f"--- Iteration {iteration}/{self.max_iterations} ---")
            
            # 使用 LLMModule 生成响应
            for chunk in self.llm_module(messages):
                if chunk.content:
                    yield chunk.content
                
                if chunk.finished:
                    messages = chunk.messages
                    break
            
            # 获取最后一条 assistant 消息
            last_content = messages[-1].get("content", "")
            
            # Step 2: 检查是否是最终答案
            if self.is_final_answer(last_content):
                self.log("info", "Final answer detected, stopping ReACT loop")
                break
            
            # Step 3: 检查并执行工具调用
            # 使用 ToolUseModule 处理工具调用
            yield "\n<tool_result>"
            for chunk in self.tool_module(messages):
                if chunk.content:
                    yield chunk.content
                
                if chunk.finished:
                    messages = chunk.messages
                    # 检查是否真的有工具调用
                    has_tool_call = chunk.metadata and chunk.metadata.get('has_tool_call', False)
                    
                    if not has_tool_call:
                        # 没有工具调用，但也没有最终答案
                        self.log("warning", "No tool call found, but no Final Answer either")
                        # 添加提示消息，引导 LLM 继续
                        messages.append({
                            "role": "user",
                            "content": "Please provide either a tool call or a Final Answer."
                        })
                    else:
                        # 有工具调用，添加 Observation 提示
                        self.log("info", f"Executed {chunk.metadata.get('tool_calls_count', 0)} tool(s)")
                    break
            yield "</tool_result>\n"
        
        # 检查是否达到最大迭代次数
        if iteration >= self.max_iterations:
            warning_msg = f"\n[Warning: Reached maximum iterations ({self.max_iterations})]\n"
            self.log("warning", warning_msg)
        
        return messages

ReActAgent = ReActModule
