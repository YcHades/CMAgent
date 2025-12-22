import importlib
import inspect
import json
import os
import re
import logging
import asyncio
from typing import Callable, List, Union, get_origin, get_args, Dict, Any, Tuple, Optional
from fastmcp import Client

TYPE_MAPPING = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    tuple: "array",
    dict: "object",
    type(None): "null"
}

class ToolManager:
    """
    统一管理本地工具和 MCP 工具
    
    工具路由格式：
    - 本地工具：tool_name -> ('local', function_object)
    - MCP 工具：tool_name -> ('mcp', (url, client))
    """

    def __init__(self):
        # 统一的工具路由：tool_name -> (type, handler)
        # 本地工具的 handler 是函数对象，MCP 工具的 handler 是 (url, client) 元组
        self.tool_routes = {}  
        self.tool_details = []  # 所有工具的详细信息
        self.mcp_clients = []  # MCP 客户端列表 (url, client)
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_tool(self, tool_name: str) -> Optional[Tuple[str, Any]]:
        """获取工具的类型和处理器"""
        return self.tool_routes.get(tool_name)

    def list_tools(self) -> str:
        """列出所有工具名称和描述，常用于调试或前端展示"""
        return '\n'.join([
            f"-Tool: {tool['name']}, Type: {tool['type']}, Description: {tool['description']}"
            for tool in self.tool_details
        ])

    def get_tool_count(self) -> Dict[str, int]:
        """统计各类型工具的数量"""
        count = {'local': 0, 'mcp': 0}
        for tool_type, _ in self.tool_routes.values():
            count[tool_type] = count.get(tool_type, 0) + 1
        return count
        
    def get_tools_description(self) -> str:
        """获取所有工具的schema形式描述信息，常用于agent的system prompt"""
        descriptions = []
        for tool in self.tool_details:
            tool_schema = tool.get('schema', '{}')
            descriptions.append(
                json.dumps(tool_schema, ensure_ascii=False)
            )
        return '\n'.join(descriptions)

    def register_local_tool(self, tool_name: str, tool_func: Callable):
        """注册本地工具"""
        if tool_name in self.tool_routes:
            self.logger.warning(f"Tool {tool_name} already exists, overwriting")
        
        self.tool_routes[tool_name] = ('local', tool_func)
        tool_schema = self.generate_local_tool_schema(tool_func)
        
        self.tool_details.append({
            'name': tool_name,
            'description': tool_schema.get('function').get('description'),
            'schema': tool_schema,
            'type': 'local'
        })

    def register_mcp_tool(self, tool_name: str, tool_info: Any, url: str, client: Any):
        """注册 MCP 工具"""
        if tool_name in self.tool_routes:
            self.logger.warning(f"Tool {tool_name} already exists, overwriting")
        
        self.tool_routes[tool_name] = ('mcp', (url, client))

        # 将 MCP schema 转换为 OpenAI function calling 格式
        tool_schema = self.format_mcp_tool_schema(
            tool_name=tool_name,
            description=tool_info.description,
            input_schema=tool_info.inputSchema if hasattr(tool_info, 'inputSchema') else {}
        )
        
        self.tool_details.append({
            'name': tool_name,
            'description': tool_schema.get('function').get('description'),
            'schema': tool_schema,
            'type': 'mcp',
            'url': url
        })

    def load_from_pyfile(self, file_name: str, folder_name: str = None):
        """从 Python 文件加载本地工具"""
        try:
            module_name = f"{folder_name}.{file_name}" if folder_name else file_name
            module = importlib.import_module(module_name)

            loaded_count = 0
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                        callable(attr)
                        and not attr_name.startswith('_')
                        and getattr(attr, '__module__', None) == module.__name__
                ):
                    self.register_local_tool(attr_name, attr)
                    loaded_count += 1
            
            if loaded_count > 0:
                self.logger.info(f"Loaded {loaded_count} tools from {module_name}")
        except Exception as e:
            self.logger.error(f"Error loading module '{module_name}': {e}")

    def load_from_folder(self, tools_folder: str = None, local_files_selector: List[str] = None):
        """从文件夹批量加载本地工具"""
        if local_files_selector is None:
            local_files_selector = [
                filename[:-3] for filename in os.listdir(tools_folder)
                if filename.endswith('.py') and filename != '__init__.py'
            ]

        for file_name in local_files_selector:
            self.load_from_pyfile(file_name, tools_folder)
    
    def load_from_mcp_servers(self, mcp_servers: List[str]):
        """从 MCP 服务器批量加载工具"""
        async def _async_load():
            for url in mcp_servers:
                try:
                    client = Client(url)
                    self.mcp_clients.append((url, client))
                    
                    async with client:
                        tools = await client.list_tools()
                    
                    loaded_count = 0
                    for tool in tools:
                        self.register_mcp_tool(tool.name, tool, url, client)
                        loaded_count += 1
                    
                    self.logger.info(f"Loaded {loaded_count} MCP tools from {url}")
                except Exception as e:
                    self.logger.error(f"Cannot load MCP tools from {url}: {e}")
                    continue
        
        asyncio.run(_async_load())
    
    def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """统一的工具调用接口"""
        if tool_name not in self.tool_routes:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool_type, handler = self.tool_routes[tool_name]
        
        if tool_type == 'local':
            return self._call_local_tool(tool_name, handler, tool_args)
        elif tool_type == 'mcp':
            return self._call_mcp_tool(tool_name, handler, tool_args)
        else:
            raise ValueError(f"Unknown tool type: {tool_type}")
    
    def _call_local_tool(self, tool_name: str, tool_func: Callable, tool_args: Dict[str, Any]) -> str:
        """调用本地工具"""
        try:
            result = tool_func(**tool_args)
            return str(result)
        except Exception as e:
            self.logger.error(f"Local tool {tool_name} execution failed: {e}")
            raise
    
    def _call_mcp_tool(self, tool_name: str, handler: Tuple[str, Any], tool_args: Dict[str, Any]) -> str:
        """调用 MCP 工具"""
        url, client = handler
        
        async def _async_call():
            try:
                async with client:
                    result = await client.call_tool(tool_name, tool_args)
                return result.data
            except Exception as e:
                self.logger.error(f"MCP tool {tool_name} from {url} failed: {e}")
                raise
        
        return asyncio.run(_async_call())

    @staticmethod
    def format_mcp_tool_schema(tool_name: str, description: str, input_schema: Dict[str, Any]) -> str:
        """
        将 MCP 的 JSON Schema 格式转换为 OpenAI function calling schema
        
        Args:
            tool_name: 工具名称
            description: 工具描述
            input_schema: MCP 的 inputSchema (JSON Schema 格式)
        
        Returns:
            OpenAI function calling schema (JSON 字符串)
        """
        # MCP 的 inputSchema 已经是 JSON Schema 格式，可以直接用作 parameters
        # 确保有基本结构
        parameters = input_schema if input_schema else {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # 如果 input_schema 缺少必要字段，补充默认值
        if "type" not in parameters:
            parameters["type"] = "object"
        if "properties" not in parameters:
            parameters["properties"] = {}
        
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description or "[WARNING: No description provided]",
                "parameters": parameters
            }
        }
        
        return tool_schema

    @staticmethod
    def generate_local_tool_schema(func: Callable, enhance_des: str = None) -> str:
        """
        根据本地工具函数生成 OpenAI function calling schema
        """

        func_name = func.__name__
        doc = inspect.getdoc(func)
        signature = inspect.signature(func)

        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }

        param_descriptions = {}
        if doc:
            match = re.search(r"Args:\s*(.*?)(?=\s*(?:Returns:|$))", doc, re.DOTALL)
            if match:
                args_section = match.group(1)
                param_lines = args_section.strip().splitlines()
                for line in param_lines:
                    param_match = re.match(r"\s*(\w+)\s*:\s*(.*?)\s*$", line.strip())
                    if param_match:
                        param_name, param_desc = param_match.groups()
                        param_descriptions[param_name] = param_desc.strip()

        for param_name, param in signature.parameters.items():
            param_type = param.annotation
            if param_type == inspect._empty:
                param_type = str

            if get_origin(param_type) is Union:
                possible_types = get_args(param_type)
                param_info = {"oneOf": []}
                for possible_type in possible_types:
                    if get_origin(possible_type) is list:
                        param_info["oneOf"].append({
                            "type": "array",
                            "items": {
                                "type": TYPE_MAPPING.get(get_args(possible_type)[0], "string")
                            }
                        })
                    else:
                        param_info["oneOf"].append({"type": TYPE_MAPPING.get(possible_type, "string")})
            elif get_origin(param_type) is list:
                param_info = {
                    "type": "array",
                    "items": {
                        "type": TYPE_MAPPING.get(get_args(param_type)[0], "string")
                    }
                }
            else:
                param_info = {"type": TYPE_MAPPING.get(param_type, "string")}

            if param_name in param_descriptions:
                param_info["description"] = param_descriptions[param_name]
            else:
                param_info["description"] = f"[WARNING: There is currently no parameter description for `{param_name}`]"

            if param.default != inspect._empty:
                param_info["default"] = param.default

            parameters["properties"][param_name] = param_info

            if param.default == inspect._empty:
                parameters["required"].append(param_name)

        if enhance_des is not None:
            func_des = enhance_des
        elif doc:
            func_des = doc.split("\nArgs:")[0]
        else:
            func_des = "[WARNING: There is currently no tool description]"

        tool_schema = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": func_des,
                "parameters": parameters
            }
        }

        return tool_schema

if __name__ == "__main__":
    # 简单测试
    tm = ToolManager()
    tm.load_from_folder(tools_folder='toolbox')
    print("Registered tools:\n", tm.list_tools())
    print("Tool counts:", tm.get_tool_count())