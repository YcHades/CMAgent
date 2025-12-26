"""
工具管理器 - 统一管理本地工具和 MCP 工具

核心职责：
    - 工具注册：从本地 Python 模块或 MCP 服务器加载工具
    - 工具调度：统一的查找、校验和调用接口
    - 连接管理：MCP 客户端的延迟连接和复用

设计原则：
    - 单一职责：ToolManager 只负责工具的生命周期管理
    - 延迟加载：MCP 连接在首次调用时才建立
    - 错误隔离：单个工具的失败不影响其他工具
"""

import asyncio
import importlib
import inspect
import json
import logging
import os
import re
import sys
from pathlib import Path
from fastmcp import Client
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_args, get_origin


# ==============================================================================
# 类型常量
# ==============================================================================

# Python 类型到 JSON Schema 类型的映射
TYPE_MAPPING: Dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    tuple: "array",
    dict: "object",
    type(None): "null",
}

# 工具类型常量
TOOL_TYPE_LOCAL = "local"
TOOL_TYPE_MCP = "mcp"


# ==============================================================================
# Schema 生成工具函数
# ==============================================================================

def generate_function_schema(func: Callable) -> Dict[str, Any]:
    """从 Python 函数生成 OpenAI function calling schema
    
    解析来源：
        - 函数签名：参数名、类型注解、默认值
        - 文档字符串：函数描述、参数描述（支持 Google 风格）
    
    Args:
        func: Python 函数对象
        
    Returns:
        OpenAI function calling 格式的 schema 字典
    """
    func_name = func.__name__
    signature = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # 解析文档字符串
    func_description, param_descriptions = _parse_docstring(doc)

    # 生成参数 schema
    parameters = _build_parameters_schema(signature, param_descriptions)

    return {
        "type": "function",
        "function": {
            "name": func_name,
            "description": func_description,
            "parameters": parameters,
        },
    }


def format_mcp_schema(
    tool_name: str, 
    description: str, 
    input_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """将 MCP 工具的 JSON Schema 转换为 OpenAI function calling 格式
    
    Args:
        tool_name: 工具名称
        description: 工具描述
        input_schema: MCP 工具的输入 schema
        
    Returns:
        OpenAI function calling 格式的 schema 字典
    """
    parameters = input_schema.copy() if input_schema else {}
    
    # 确保基本字段存在
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description or "[No description]",
            "parameters": parameters,
        },
    }


def _parse_docstring(doc: str) -> Tuple[str, Dict[str, str]]:
    """解析 Google 风格的文档字符串
    
    Returns:
        (函数描述, {参数名: 参数描述})
    """
    func_description = doc.split("\nArgs:")[0].strip() or "[No description]"
    param_descriptions = {}
    
    match = re.search(r"Args:\s*(.*?)(?=\s*(?:Returns:|Raises:|$))", doc, re.DOTALL)
    if match:
        for line in match.group(1).strip().splitlines():
            param_match = re.match(r"\s*(\w+)\s*:\s*(.+?)\s*$", line.strip())
            if param_match:
                param_descriptions[param_match.group(1)] = param_match.group(2).strip()
    
    return func_description, param_descriptions


def _build_parameters_schema(
    signature: inspect.Signature, 
    param_descriptions: Dict[str, str]
) -> Dict[str, Any]:
    """从函数签名构建参数 schema"""
    parameters: Dict[str, Any] = {
        "type": "object", 
        "properties": {}, 
        "required": []
    }

    for param_name, param in signature.parameters.items():
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        param_schema = _type_to_json_schema(param_type)
        
        # 添加描述
        param_schema["description"] = param_descriptions.get(
            param_name, f"[No description for '{param_name}']"
        )
        
        # 添加默认值
        if param.default != inspect.Parameter.empty:
            param_schema["default"] = param.default
        else:
            parameters["required"].append(param_name)
        
        parameters["properties"][param_name] = param_schema

    return parameters


def _type_to_json_schema(param_type: type) -> Dict[str, Any]:
    """将 Python 类型转换为 JSON Schema"""
    origin = get_origin(param_type)
    
    if origin is Union:
        return {
            "oneOf": [_type_to_json_schema(t) for t in get_args(param_type)]
        }
    
    if origin is list:
        item_type = get_args(param_type)[0] if get_args(param_type) else str
        return {
            "type": "array",
            "items": {"type": TYPE_MAPPING.get(item_type, "string")}
        }
    
    return {"type": TYPE_MAPPING.get(param_type, "string")}


# ==============================================================================
# 结果序列化工具函数
# ==============================================================================

def serialize_result(result: Any) -> str:
    """序列化工具执行结果为字符串
    
    Args:
        result: 任意类型的执行结果
        
    Returns:
        字符串格式的结果
    """
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False, indent=2)
    return str(result)


def serialize_mcp_result(result: Any) -> str:
    """序列化 MCP 工具的 CallToolResult
    
    MCP 返回结构优先级：
        1. data - 解析后的结构化数据
        2. structured_content - 原始结构化内容
        3. content - ContentBlock 列表
        4. 兜底 - 直接转字符串
    """
    # 优先使用解析后的 data
    if hasattr(result, 'data') and result.data is not None:
        return serialize_result(result.data)
    
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
    
    return str(result)


# ==============================================================================
# 工具管理器主类
# ==============================================================================

class ToolManager:
    """统一的工具管理和调度中心
    
    工具路由表结构：
        tool_name -> (tool_type, handler, schema)
        
        - 本地工具: (TOOL_TYPE_LOCAL, callable, schema)
        - MCP 工具: (TOOL_TYPE_MCP, server_alias, schema)
    
    使用示例：
        >>> tm = ToolManager()
        >>> tm.load_local_tools('toolbox')
        >>> tm.load_mcp_tools(mcp_config)
        >>> result = tm.call_tool('hello', {'name': 'World'})
        >>> await tm.cleanup()
    """

    def __init__(self):
        # 工具注册表
        self._routes: Dict[str, Tuple[str, Any, Dict]] = {}
        self._details: List[Dict[str, Any]] = []
        
        # MCP 连接管理
        self._mcp_configs: Dict[str, Dict[str, Any]] = {}
        self._mcp_clients: Dict[str, Client] = {}
        self._mcp_locks: Dict[str, asyncio.Lock] = {}
        
        self._logger = logging.getLogger(self.__class__.__name__)

    # ==========================================================================
    # 公共接口 - 工具查询
    # ==========================================================================

    def get_tool(self, name: str) -> Optional[Tuple[str, Any, Dict]]:
        """获取工具的完整路由信息
        
        Args:
            name: 工具名称
            
        Returns:
            (tool_type, handler, schema) 或 None
        """
        return self._routes.get(name)

    def get_tool_count(self) -> Dict[str, int]:
        """获取各类型工具的数量统计"""
        counts = {TOOL_TYPE_LOCAL: 0, TOOL_TYPE_MCP: 0}
        for tool_type, _, _ in self._routes.values():
            counts[tool_type] = counts.get(tool_type, 0) + 1
        return counts

    def list_tools(self) -> str:
        """列出所有工具（人类可读格式）"""
        return "\n".join(
            f"- {d['name']} ({d['type']}): {d['description']}"
            for d in self._details
        )

    def get_tools_schema_object(self) -> List[Dict[str, Any]]:
        """获取所有工具的 OpenAI function calling schema 列表"""
        return [d["schema"] for d in self._details]

    def get_tools_description(self) -> str:
        """获取所有工具 schema 的 JSON 字符串（用于 LLM Prompt）"""
        return "\n".join(
            json.dumps(d["schema"], ensure_ascii=False)
            for d in self._details
        )

    # ==========================================================================
    # 公共接口 - 工具加载
    # ==========================================================================

    def load_local_tools(
        self, 
        folder: str, 
        files: Optional[List[str]] = None
    ) -> None:
        """从文件夹加载本地 Python 工具
        
        Args:
            folder: 工具文件夹路径或包名（如 'toolbox' 或 'cmagent.toolbox'）
            files: 要加载的文件名列表（不含 .py），None 表示加载全部
        """
        module_base, folder_path = self._resolve_folder(folder)
        if not folder_path:
            return

        # 默认加载所有 Python 文件
        if files is None:
            files = [
                f[:-3] for f in os.listdir(folder_path)
                if f.endswith(".py") and f != "__init__.py"
            ]

        for filename in files:
            self._load_module(filename, module_base)

    def load_mcp_tools(self, config: Dict[str, Any]) -> None:
        """从配置加载 MCP 工具
        
        支持的配置格式：
            {
                "mcpServers": {
                    "server_name": {
                        "command": "uvx", "args": ["..."],  # stdio 方式
                        # 或
                        "url": "http://...",  # http/sse 方式
                    }
                }
            }
        """
        servers = config.get("mcpServers", {})
        asyncio.run(self._load_mcp_servers(servers))

    # ==========================================================================
    # 公共接口 - 工具调用
    # ==========================================================================

    def call_tool(self, name: str, args: Dict[str, Any]) -> str:
        """调用工具（统一入口）
        
        Args:
            name: 工具名称
            args: 工具参数字典
            
        Returns:
            字符串格式的执行结果
            
        Raises:
            ValueError: 工具不存在或参数无效
            RuntimeError: 工具执行失败
        """
        if name not in self._routes:
            raise ValueError(f"Tool '{name}' not found")

        tool_type, handler, schema = self._routes[name]
        self._validate_args(name, args, schema)

        if tool_type == TOOL_TYPE_LOCAL:
            return self._call_local(name, handler, args)
        else:
            return self._call_mcp(name, handler, args)

    # ==========================================================================
    # 公共接口 - 生命周期管理
    # ==========================================================================

    async def cleanup(self) -> None:
        """清理所有 MCP 客户端连接"""
        for alias, client in list(self._mcp_clients.items()):
            try:
                await client.__aexit__(None, None, None)
                self._logger.info(f"✓ Closed MCP server '{alias}'")
            except Exception as e:
                self._logger.error(f"✗ Error closing '{alias}': {e}")
        
        self._mcp_clients.clear()
        self._mcp_locks.clear()

    # ==========================================================================
    # 私有方法 - 本地工具
    # ==========================================================================

    def _resolve_folder(self, folder: str) -> Tuple[Optional[str], Optional[str]]:
        """解析工具目录，返回 (模块基础名, 文件系统路径)"""
        # 尝试作为文件路径
        path = Path(folder)
        if path.exists():
            module_base = folder if "/" not in folder and "\\" not in folder else None
            if module_base is None and str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return module_base, str(path)

        # 尝试作为模块导入
        for module_name in [folder, f"cmagent.{folder}"]:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "__path__"):
                    return module.__name__, str(next(iter(module.__path__)))
            except ModuleNotFoundError:
                continue
        
        self._logger.error(f"✗ Tools folder not found: {folder}")
        return None, None

    def _load_module(self, filename: str, module_base: Optional[str]) -> None:
        """从 Python 模块加载所有公开函数"""
        module_name = f"{module_base}.{filename}" if module_base else filename
        
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            self._logger.error(f"✗ Error importing '{module_name}': {e}")
            return

        count = 0
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if callable(obj) and getattr(obj, "__module__", None) == module.__name__:
                self._register_local(name, obj)
                count += 1

        if count > 0:
            self._logger.info(f"✓ Loaded {count} tools from {module_name}")

    def _register_local(self, name: str, func: Callable) -> None:
        """注册本地工具"""
        if name in self._routes:
            self._logger.warning(f"⚠ Overwriting tool '{name}'")
            self._details = [d for d in self._details if d["name"] != name]

        schema = generate_function_schema(func)
        self._routes[name] = (TOOL_TYPE_LOCAL, func, schema)
        self._details.append({
            "name": name,
            "type": TOOL_TYPE_LOCAL,
            "description": schema["function"]["description"],
            "schema": schema,
        })

    def _call_local(self, name: str, func: Callable, args: Dict[str, Any]) -> str:
        """执行本地工具"""
        self._logger.info(f"Calling '{name}' [local]")
        try:
            result = func(**args)
            return serialize_result(result)
        except Exception as e:
            self._logger.error(f"✗ Local tool '{name}' failed: {e}")
            raise

    # ==========================================================================
    # 私有方法 - MCP 工具
    # ==========================================================================

    async def _load_mcp_servers(self, servers: Dict[str, Any]) -> None:
        """异步加载所有 MCP 服务器的工具"""
        for alias, config in servers.items():
            await self._register_mcp_server(alias, config)

    async def _register_mcp_server(self, alias: str, config: Any) -> None:
        """注册单个 MCP 服务器的所有工具"""
        normalized = self._normalize_mcp_config(alias, config)
        if not normalized:
            return
        
        self._mcp_configs[alias] = normalized
        
        try:
            # 临时连接获取工具列表
            wrapped = {"mcpServers": {alias: normalized}}
            async with Client(wrapped) as client:
                tools = await client.list_tools()
                for tool in tools:
                    self._register_mcp_tool(alias, tool.name, tool.description, 
                                           getattr(tool, "inputSchema", {}))
                self._logger.info(f"✓ Loaded {len(tools)} tools from MCP '{alias}'")
        except Exception as e:
            self._logger.error(f"✗ Failed to load MCP '{alias}': {e}")

    def _normalize_mcp_config(self, alias: str, config: Any) -> Optional[Dict[str, Any]]:
        """规范化 MCP 服务器配置"""
        if isinstance(config, str):
            return {"url": config, "transport": "sse"}
        
        if isinstance(config, dict):
            result = config.copy()
            if "url" in config:
                result.setdefault("transport", "sse")
            elif "command" in config:
                result.setdefault("transport", "stdio")
            else:
                self._logger.error(f"Invalid MCP config for '{alias}': need 'url' or 'command'")
                return None
            return result
        
        self._logger.error(f"Invalid config type for '{alias}': {type(config).__name__}")
        return None

    def _register_mcp_tool(
        self, 
        server_alias: str, 
        tool_name: str, 
        description: str,
        input_schema: Dict[str, Any]
    ) -> None:
        """注册 MCP 工具（冲突时添加服务器前缀）"""
        final_name = tool_name
        
        # 处理命名冲突
        if tool_name in self._routes:
            existing_type, existing_handler, _ = self._routes[tool_name]
            
            # 同服务器重复注册 -> 覆盖
            if existing_type == TOOL_TYPE_MCP and existing_handler == server_alias:
                self._details = [d for d in self._details if d["name"] != tool_name]
            else:
                # 不同来源冲突 -> 重命名
                if existing_type == TOOL_TYPE_MCP and not tool_name.startswith(f"{existing_handler}_"):
                    self._rename_tool(tool_name, f"{existing_handler}_{tool_name}")
                final_name = f"{server_alias}_{tool_name}"
                self._logger.warning(f"⚠ Name conflict: '{tool_name}' -> '{final_name}'")

        schema = format_mcp_schema(final_name, description, input_schema)
        self._routes[final_name] = (TOOL_TYPE_MCP, server_alias, input_schema)
        self._details.append({
            "name": final_name,
            "type": TOOL_TYPE_MCP,
            "description": schema["function"]["description"],
            "schema": schema,
            "server": server_alias,
            "original_name": tool_name,
        })

    def _rename_tool(self, old_name: str, new_name: str) -> None:
        """重命名已注册的工具"""
        if old_name not in self._routes:
            return
            
        self._routes[new_name] = self._routes.pop(old_name)
        
        for detail in self._details:
            if detail["name"] == old_name:
                detail["name"] = new_name
                if "schema" in detail:
                    detail["schema"]["function"]["name"] = new_name
                break
        
        self._logger.info(f"⚠ Renamed '{old_name}' -> '{new_name}'")

    def _call_mcp(self, name: str, server_alias: str, args: Dict[str, Any]) -> str:
        """执行 MCP 工具（同步包装）"""
        return asyncio.run(self._call_mcp_async(name, server_alias, args))

    async def _call_mcp_async(
        self, name: str, server_alias: str, args: Dict[str, Any]
    ) -> str:
        """异步执行 MCP 工具"""
        # 获取原始工具名
        original_name = name
        for d in self._details:
            if d["name"] == name and d.get("original_name"):
                original_name = d["original_name"]
                break

        self._logger.info(f"Calling '{name}' [mcp:{server_alias}]")

        # 确保并发锁存在
        if server_alias not in self._mcp_locks:
            self._mcp_locks[server_alias] = asyncio.Lock()

        async with self._mcp_locks[server_alias]:
            try:
                client = await self._get_mcp_client(server_alias)
                result = await client.call_tool(original_name, args)
                return serialize_mcp_result(result)
            except Exception as e:
                self._logger.error(f"✗ MCP tool '{name}' failed: {e}")
                raise

    async def _get_mcp_client(self, alias: str) -> Client:
        """获取或创建 MCP 客户端连接"""
        if alias not in self._mcp_configs:
            raise ValueError(f"MCP server '{alias}' not configured")
        
        # 检查现有连接
        if alias in self._mcp_clients:
            client = self._mcp_clients[alias]
            if hasattr(client, '_session') and client._session is not None:
                return client
            # 连接无效，清理
            try:
                await client.__aexit__(None, None, None)
            except:
                pass
            del self._mcp_clients[alias]
        
        # 创建新连接
        config = self._mcp_configs[alias]
        client = Client({"mcpServers": {alias: config}})
        
        try:
            await client.__aenter__()
            self._mcp_clients[alias] = client
            self._logger.info(f"✓ Connected to MCP '{alias}'")
            return client
        except Exception as e:
            raise RuntimeError(f"Failed to connect to MCP '{alias}': {e}")

    # ==========================================================================
    # 私有方法 - 参数校验
    # ==========================================================================

    def _validate_args(
        self, name: str, args: Dict[str, Any], schema: Dict[str, Any]
    ) -> None:
        """基于 JSON Schema 校验工具参数"""
        if not isinstance(schema, dict):
            return
        
        # 提取 parameters
        params = schema.get("function", {}).get("parameters", schema.get("parameters", schema))
        if not params:
            return
        
        required = params.get("required", [])
        properties = params.get("properties", {})

        # 检查必需参数
        missing = [p for p in required if p not in args]
        if missing:
            raise ValueError(f"Tool '{name}' missing required parameters: {missing}")

        # 警告未知参数
        unknown = [p for p in args if p not in properties]
        if unknown:
            self._logger.warning(f"Tool '{name}' received unknown parameters: {unknown}")