"""
Planning-related agent modules.
"""

from typing import List, Dict, Any, Generator, Optional, Callable

from .base import BaseModule
from .llm_module import LLMModule
from .tool_use_module import ToolUseLoopModule
from ..utils import extract_json_codeblock


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
