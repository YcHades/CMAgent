"""
工具管理器 - 统一管理本地工具和 MCP 工具

核心特性：
    - 本地工具：从 Python 模块动态加载函数
    - MCP 工具：支持 HTTP/STDIO 传输，延迟连接策略
    - 命名空间：自动添加前缀避免工具名冲突
    - Schema 生成：自动从函数签名和文档字符串生成 OpenAI 格式 schema
    - 参数校验：基于 JSON Schema 的运行时验证
    - 连接池：MCP 客户端延迟创建并复用连接

架构设计：
    - 工具路由：统一的工具查找和调度机制
    - 异步优先：MCP 工具使用异步调用，本地工具支持同步/异步
    - 错误隔离：工具加载/调用失败不影响其他工具
"""

import asyncio
import importlib
import inspect
import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_args, get_origin

from fastmcp import Client


# ==============================================================================
# 类型映射表
# ==============================================================================

TYPE_MAPPING = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    tuple: "array",
    dict: "object",
    type(None): "null",
}


# ==============================================================================
# 工具管理器主类
# ==============================================================================

class ToolManager:
    """统一的工具管理和调度中心
    
    工具路由格式：
        tool_name -> (tool_type, handler, schema)
        
        - 本地工具: ('local', function, schema)
        - MCP 工具: ('mcp', (server_alias, client_ref), schema)
    
    使用示例：
        >>> tm = ToolManager()
        >>> tm.load_from_folder('toolbox')
        >>> tm.load_from_mcp_config(config)
        >>> result = tm.call_tool('hello', {'name': 'World'})
    """

    def __init__(self):
        # 核心数据结构
        self.tool_routes: Dict[str, Tuple[str, Any, Any]] = {}
        self.tool_details: List[Dict[str, Any]] = []
        
        # MCP 连接管理
        self.mcp_configs: Dict[str, Dict[str, Any]] = {}
        self.mcp_client_instances: Dict[str, Any] = {}
        self.mcp_locks: Dict[str, asyncio.Lock] = {}
        
        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)

    # ==========================================================================
    # 生命周期管理
    # ==========================================================================

    async def cleanup(self):
        """清理所有 MCP 客户端连接"""
        for server_alias, client in list(self.mcp_client_instances.items()):
            try:
                await client.__aexit__(None, None, None)
                self.logger.info(f"✓ Closed MCP server '{server_alias}'")
            except Exception as e:
                self.logger.error(f"✗ Error closing '{server_alias}': {e}")
        
        self.mcp_client_instances.clear()
        self.mcp_locks.clear()

    # ==========================================================================
    # 查询接口 - 工具发现和统计
    # ==========================================================================

    def get_tool(self, tool_name: str) -> Optional[Tuple[str, Any, Any]]:
        """获取工具的完整信息
        
        Returns:
            (tool_type, handler, schema) 或 None
        """
        return self.tool_routes.get(tool_name)

    def get_tool_count(self) -> Dict[str, int]:
        """统计各类型工具的数量
        
        Returns:
            {"local": count, "mcp": count}
        """
        count = {"local": 0, "mcp": 0}
        for tool_type, _, _ in self.tool_routes.values():
            count[tool_type] = count.get(tool_type, 0) + 1
        return count

    def list_tools(self) -> str:
        """列出所有工具的名称和描述（人类可读格式）"""
        return "\n".join(
            f"- {tool['name']} ({tool['type']}): {tool['description']}"
            for tool in self.tool_details
        )

    def get_tools_description(self) -> str:
        """获取所有工具的 OpenAI function calling schema（LLM 格式）"""
        return "\n".join(
            json.dumps(tool["schema"], ensure_ascii=False)
            for tool in self.tool_details
        )

    # ==========================================================================
    # 加载接口 - 工具注册
    # ==========================================================================

    def load_from_folder(
        self, 
        tools_folder: str, 
        local_files_selector: Optional[List[str]] = None
    ):
        """从文件夹批量加载本地 Python 工具
        
        Args:
            tools_folder: 工具文件夹路径（相对或绝对）
            local_files_selector: 文件名列表（不含.py），None 表示加载全部
        """
        if local_files_selector is None:
            local_files_selector = [
                filename[:-3]
                for filename in os.listdir(tools_folder)
                if filename.endswith(".py") and filename != "__init__.py"
            ]

        for file_name in local_files_selector:
            self._load_pyfile(file_name, tools_folder)

    def load_from_mcp_config(self, mcp_config: Dict[str, Any]):
        """从配置加载 MCP 工具（延迟连接策略）
        
        统一支持 stdio 和 http(se) 两种传输方式
        
        Args:
            mcp_config: MCP 配置字典，格式为：
                {
                    "mcpServers": {
                        "server_alias": {
                            # stdio 方式
                            "command": "uvx",
                            "args": ["mcp-server-time"],
                            "env": {"KEY": "value"},  # 可选
                            "transport": "stdio"  # 可选，默认stdio
                        },
                        # 或 http(sse) 方式
                        "server_alias2": {
                            "url": "http://localhost:8000/sse",
                            "transport": "sse"
                        },
                        # 简化的 http 配置（直接传URL字符串）
                        "server_alias3": "http://localhost:8001/sse"
                    }
                }
        """
        servers_config = mcp_config.get("mcpServers", {})
        asyncio.run(self._load_mcp_tools_async(servers_config))

    async def _load_mcp_tools_async(self, mcp_config: Dict[str, Any]):
        """异步加载 MCP 工具（内部方法）
        
        Args:
            mcp_config: 服务器配置字典，key为服务器别名，value为配置
        """
        for server_alias, server_config in mcp_config.items():
            await self._register_mcp_server(server_alias, server_config)

    async def _register_mcp_server(self, server_alias: str, server_config: Any):
        """注册单个 MCP 服务器的所有工具
        
        Args:
            server_alias: 服务器别名
            server_config: 服务器配置，支持以下格式：
                - 字符串: URL (默认使用sse传输)
                - 字典: 完整配置 (command+args 用于stdio, url用于http/sse)
        """
        try:
            # 规范化配置格式
            normalized_config = self._normalize_server_config(server_alias, server_config)
            if not normalized_config:
                return
            
            # 保存配置供后续连接使用
            self.mcp_configs[server_alias] = normalized_config

            # 临时连接获取工具列表（连接后立即关闭）
            wrapped_config = {"mcpServers": {server_alias: normalized_config}}
            async with Client(wrapped_config) as client:
                tools = await client.list_tools()
                
                # 注册所有工具
                for tool in tools:
                    self._register_mcp_tool(tool.name, tool, server_alias)
                
                self.logger.info(
                    f"✓ Loaded {len(tools)} tools from MCP server '{server_alias}'"
                )

        except Exception as e:
            self.logger.error(f"✗ Failed to load MCP server '{server_alias}': {e}")
    
    def _normalize_server_config(self, server_alias: str, server_config: Any) -> Optional[Dict[str, Any]]:
        """规范化服务器配置格式
        
        Args:
            server_alias: 服务器别名
            server_config: 原始配置
            
        Returns:
            规范化后的配置字典，失败返回None
        """
        if isinstance(server_config, str):
            # 字符串格式: 视为 HTTP URL
            return {
                "url": server_config,
                "transport": "sse"
            }
        
        elif isinstance(server_config, dict):
            # 字典格式: 检测传输方式
            if "url" in server_config:
                # HTTP/SSE 方式
                config = server_config.copy()
                if "transport" not in config:
                    config["transport"] = "sse"
                return config
            
            elif "command" in server_config:
                # STDIO 方式
                config = server_config.copy()
                if "transport" not in config:
                    config["transport"] = "stdio"
                return config
            
            else:
                self.logger.error(
                    f"Invalid config for '{server_alias}': "
                    "must contain 'url' (for http/sse) or 'command' (for stdio)"
                )
                return None
        
        else:
            self.logger.error(
                f"Invalid config type for '{server_alias}': "
                f"expected str or dict, got {type(server_config).__name__}"
            )
            return None

    # ==========================================================================
    # 调用接口 - 工具执行
    # ==========================================================================

    def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """调用工具（统一入口）
        
        Args:
            tool_name: 工具名称（可能带命名空间前缀）
            tool_args: 工具参数字典
            
        Returns:
            工具执行结果（字符串格式）
            
        Raises:
            ValueError: 工具不存在或参数不合法
            RuntimeError: 工具执行失败
        """
        # 1. 查找工具
        if tool_name not in self.tool_routes:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool_type, handler, schema = self.tool_routes[tool_name]

        # 2. 校验参数
        self._validate_args(tool_name, tool_args, schema)

        # 3. 分发调用
        if tool_type == "local":
            return self._call_local_tool(tool_name, handler, tool_args)
        elif tool_type == "mcp":
            return self._call_mcp_tool(tool_name, handler, tool_args)
        else:
            raise ValueError(f"Unknown tool type: {tool_type}")

    # ==========================================================================
    # 内部方法 - 工具注册
    # ==========================================================================

    def _register_local_tool(self, tool_name: str, tool_func: Callable):
        """注册本地 Python 函数为工具"""
        if tool_name in self.tool_routes:
            self.logger.warning(f"⚠ Overwriting existing tool '{tool_name}'")

        schema = ToolManager.generate_local_schema(tool_func)
        self.tool_routes[tool_name] = ("local", tool_func, schema)
        self.tool_details.append({
            "name": tool_name,
            "description": schema["function"]["description"],
            "schema": schema,
            "type": "local",
        })

    def _register_mcp_tool(self, tool_name: str, tool_info: Any, server_alias: str):
        """注册 MCP 工具（带命名空间前缀）
        
        Args:
            tool_name: 原始工具名
            tool_info: MCP 工具信息对象
            server_alias: 服务器别名（用作命名空间）
        """
        namespaced_name = f"{server_alias}_{tool_name}"

        if namespaced_name in self.tool_routes:
            self.logger.warning(f"⚠ Overwriting existing tool '{namespaced_name}'")

        # 提取 schema（client 引用稍后创建）
        input_schema = getattr(tool_info, "inputSchema", {})
        self.tool_routes[namespaced_name] = ("mcp", (server_alias, None), input_schema)

        schema = ToolManager.format_mcp_schema(namespaced_name, tool_info.description, input_schema)
        self.tool_details.append({
            "name": namespaced_name,
            "description": schema["function"]["description"],
            "schema": schema,
            "type": "mcp",
            "server": server_alias,
            "original_name": tool_name,
        })

    def _load_pyfile(self, file_name: str, folder_name: Optional[str] = None):
        """从 Python 文件加载所有公开函数"""
        try:
            module_name = f"{folder_name}.{file_name}" if folder_name else file_name
            module = importlib.import_module(module_name)

            loaded_count = 0
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                # 过滤：可调用、非私有、属于该模块
                if (callable(attr) 
                    and not attr_name.startswith("_")
                    and getattr(attr, "__module__", None) == module.__name__):
                    self._register_local_tool(attr_name, attr)
                    loaded_count += 1

            if loaded_count > 0:
                self.logger.info(f"✓ Loaded {loaded_count} tools from {module_name}")
        
        except Exception as e:
            self.logger.error(f"✗ Error loading module '{module_name}': {e}")

    # ==========================================================================
    # 内部方法 - 工具执行
    # ==========================================================================

    def _call_local_tool(
        self, tool_name: str, tool_func: Callable, tool_args: Dict[str, Any]
    ) -> str:
        """执行本地 Python 函数"""
        try:
            result = tool_func(**tool_args)
            return ToolManager.serialize_result(result)
        except Exception as e:
            self.logger.error(f"✗ Local tool '{tool_name}' failed: {e}")
            raise

    def _call_mcp_tool(
        self, tool_name: str, handler: Tuple[str, Any], tool_args: Dict[str, Any]
    ) -> str:
        """执行 MCP 工具（同步包装）"""
        return asyncio.run(self._acall_mcp_tool(tool_name, handler, tool_args))

    async def _acall_mcp_tool(
        self, tool_name: str, handler: Tuple[str, Any], tool_args: Dict[str, Any]
    ) -> str:
        """异步执行 MCP 工具（带并发控制）"""
        server_alias, _ = handler

        # 提取原始工具名（去除命名空间前缀）
        original_name = (
            tool_name[len(server_alias) + 1:]
            if tool_name.startswith(f"{server_alias}_")
            else tool_name
        )

        # 获取或创建并发锁
        if server_alias not in self.mcp_locks:
            self.mcp_locks[server_alias] = asyncio.Lock()

        async with self.mcp_locks[server_alias]:
            try:
                client = await self._ensure_mcp_client(server_alias)
                result = await client.call_tool(original_name, tool_args)
                return ToolManager.serialize_mcp_result(result)
            
            except Exception as e:
                self.logger.error(f"✗ MCP tool '{tool_name}' failed: {e}")
                raise

    async def _ensure_mcp_client(self, server_alias: str) -> Any:
        """确保 MCP 客户端已连接（延迟初始化）"""
        # 如果已存在连接，直接返回
        if server_alias in self.mcp_client_instances:
            return self.mcp_client_instances[server_alias]
        
        # 检查配置是否存在
        if server_alias not in self.mcp_configs:
            raise ValueError(
                f"MCP server '{server_alias}' not configured. "
                "Call load_from_mcp_config first."
            )
        
        # 创建新连接
        config = self.mcp_configs[server_alias]
        wrapped_config = {"mcpServers": {server_alias: config}}
        client = Client(wrapped_config)
        
        await client.__aenter__()
        self.mcp_client_instances[server_alias] = client
        
        self.logger.info(f"✓ Connected to MCP server '{server_alias}'")
        return client

    # ==========================================================================
    # 内部方法 - 参数校验
    # ==========================================================================

    def _validate_args(
        self, tool_name: str, tool_args: Dict[str, Any], schema: Dict[str, Any]
    ):
        """基于 JSON Schema 校验工具参数
        
        Raises:
            ValueError: 必需参数缺失
        """
        if not isinstance(schema, dict):
            return
        
        # 提取 parameters（支持多种 schema 格式）
        if "function" in schema and "parameters" in schema["function"]:
            parameters = schema["function"]["parameters"]
        elif "parameters" in schema:
            parameters = schema["parameters"]
        else:
            parameters = schema
        
        if not parameters:
            return
        
        required = parameters.get("required", [])
        properties = parameters.get("properties", {})

        # 检查必需参数
        missing = [p for p in required if p not in tool_args]
        if missing:
            raise ValueError(
                f"Tool '{tool_name}' missing required parameters: {missing}"
            )

        # 警告未知参数
        unknown = [p for p in tool_args if p not in properties]
        if unknown:
            self.logger.warning(
                f"Tool '{tool_name}' received unknown parameters: {unknown}"
            )

    # ==========================================================================
    # 静态工具方法 - 结果序列化（可复用）
    # ==========================================================================

    @staticmethod
    def serialize_result(result: Any) -> str:
        """序列化工具结果为字符串（通用方法）
        
        支持类型：
            - str: 直接返回
            - dict/list: JSON 格式化
            - 其他: 转为字符串
        """
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    @staticmethod
    def serialize_mcp_result(result: Any) -> str:
        """序列化 MCP 工具的 CallToolResult（专用方法）
        
        MCP 返回结构（优先级从高到低）：
            1. data: 解析后的结构化数据
            2. structured_content: 原始结构化内容
            3. content: ContentBlock 列表（文本/图片等）
            4. 兜底: 直接转字符串
        """
        # 优先使用解析后的 data
        if hasattr(result, 'data') and result.data is not None:
            return ToolManager.serialize_result(result.data)
        
        # 其次使用 structured_content
        if hasattr(result, 'structured_content') and result.structured_content:
            return json.dumps(result.structured_content, ensure_ascii=False, indent=2)
        
        # 从 content 列表提取文本
        if hasattr(result, 'content') and result.content:
            text_parts = []
            for block in result.content:
                if hasattr(block, 'type') and block.type == 'text':
                    text_parts.append(getattr(block, 'text', str(block)))
                else:
                    text_parts.append(str(block))
            
            if text_parts:
                return '\n'.join(text_parts)
        
        # 兜底方案
        return str(result)

    # ==========================================================================
    # 静态工具方法 - Schema 生成（可复用）
    # ==========================================================================

    @staticmethod
    def format_mcp_schema(
        tool_name: str, description: str, input_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将 MCP JSON Schema 转换为 OpenAI function calling 格式"""
        parameters = input_schema or {}
        
        # 确保基本字段存在
        if "type" not in parameters:
            parameters["type"] = "object"
        if "properties" not in parameters:
            parameters["properties"] = {}

        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description or "[No description]",
                "parameters": parameters,
            },
        }

    @staticmethod
    def generate_local_schema(func: Callable) -> Dict[str, Any]:
        """从 Python 函数生成 OpenAI function calling schema
        
        解析来源：
            - 函数签名：参数名、类型注解、默认值
            - 文档字符串：函数描述、参数描述
        """
        func_name = func.__name__
        signature = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        # 提取函数描述和参数描述
        func_description = doc.split("\nArgs:")[0].strip() or "[No description]"
        param_descriptions = {}
        
        # 解析 Args 部分
        match = re.search(r"Args:\s*(.*?)(?=\s*(?:Returns:|$))", doc, re.DOTALL)
        if match:
            args_section = match.group(1)
            for line in args_section.strip().splitlines():
                param_match = re.match(r"\s*(\w+)\s*:\s*(.*?)\s*$", line.strip())
                if param_match:
                    param_name, param_desc = param_match.groups()
                    param_descriptions[param_name] = param_desc.strip()

        # 生成参数 schema
        parameters = {"type": "object", "properties": {}, "required": []}

        for param_name, param in signature.parameters.items():
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            
            # 处理类型注解
            if get_origin(param_type) is Union:
                # Union 类型
                param_schema = {"oneOf": []}
                for possible_type in get_args(param_type):
                    if get_origin(possible_type) is list:
                        item_type = get_args(possible_type)[0] if get_args(possible_type) else str
                        param_schema["oneOf"].append({
                            "type": "array",
                            "items": {"type": TYPE_MAPPING.get(item_type, "string")}
                        })
                    else:
                        param_schema["oneOf"].append({"type": TYPE_MAPPING.get(possible_type, "string")})
            
            elif get_origin(param_type) is list:
                # List 类型
                item_type = get_args(param_type)[0] if get_args(param_type) else str
                param_schema = {
                    "type": "array",
                    "items": {"type": TYPE_MAPPING.get(item_type, "string")}
                }
            
            else:
                # 基础类型
                param_schema = {"type": TYPE_MAPPING.get(param_type, "string")}
            
            # 添加描述和默认值
            param_schema["description"] = param_descriptions.get(
                param_name, f"[No description for '{param_name}']"
            )
            
            if param.default != inspect.Parameter.empty:
                param_schema["default"] = param.default
            
            parameters["properties"][param_name] = param_schema
            
            # 标记必需参数
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        return {
            "type": "function",
            "function": {
                "name": func_name,
                "description": func_description,
                "parameters": parameters,
            },
        }


# ==============================================================================
# 测试代码
# ==============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tm = ToolManager()

    # 加载本地工具
    tm.load_from_folder(tools_folder="toolbox")
    print("=" * 60)
    print("Registered tools:\n", tm.list_tools())
    print("Tool counts:", tm.get_tool_count())
