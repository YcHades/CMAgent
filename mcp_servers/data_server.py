"""
数据转换 MCP 服务器
提供各种数据格式转换和编码工具
"""
import json
import base64
import hashlib
from typing import Any, Dict, List
from fastmcp import FastMCP

# 创建 MCP 服务器实例
mcp = FastMCP("Data Converter")


@mcp.tool()
def json_to_yaml(json_str: str) -> str:
    """
    将JSON字符串转换为YAML格式
    
    Args:
        json_str: JSON格式的字符串
        
    Returns:
        YAML格式的字符串
    """
    try:
        data = json.loads(json_str)
        # 简单的YAML格式化（这里使用基础方法，生产环境建议用yaml库）
        def to_yaml(obj, indent=0):
            lines = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        lines.append("  " * indent + f"{key}:")
                        lines.append(to_yaml(value, indent + 1))
                    else:
                        lines.append("  " * indent + f"{key}: {value}")
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        lines.append("  " * indent + "-")
                        lines.append(to_yaml(item, indent + 1))
                    else:
                        lines.append("  " * indent + f"- {item}")
            return "\n".join(lines)
        
        return to_yaml(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"无效的JSON格式: {str(e)}")


@mcp.tool()
def base64_encode(text: str) -> str:
    """
    Base64编码
    
    Args:
        text: 要编码的文本
        
    Returns:
        Base64编码后的字符串
    """
    text_bytes = text.encode('utf-8')
    base64_bytes = base64.b64encode(text_bytes)
    return base64_bytes.decode('utf-8')


@mcp.tool()
def base64_decode(encoded_text: str) -> str:
    """
    Base64解码
    
    Args:
        encoded_text: Base64编码的字符串
        
    Returns:
        解码后的文本
    """
    try:
        base64_bytes = encoded_text.encode('utf-8')
        text_bytes = base64.b64decode(base64_bytes)
        return text_bytes.decode('utf-8')
    except Exception as e:
        raise ValueError(f"Base64解码失败: {str(e)}")


@mcp.tool()
def md5_hash(text: str) -> str:
    """
    计算MD5哈希值
    
    Args:
        text: 要计算哈希的文本
        
    Returns:
        MD5哈希值（十六进制字符串）
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


@mcp.tool()
def sha256_hash(text: str) -> str:
    """
    计算SHA256哈希值
    
    Args:
        text: 要计算哈希的文本
        
    Returns:
        SHA256哈希值（十六进制字符串）
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@mcp.tool()
def hex_encode(text: str) -> str:
    """
    将文本转换为十六进制
    
    Args:
        text: 要转换的文本
        
    Returns:
        十六进制字符串
    """
    return text.encode('utf-8').hex()


@mcp.tool()
def hex_decode(hex_str: str) -> str:
    """
    将十六进制转换为文本
    
    Args:
        hex_str: 十六进制字符串
        
    Returns:
        解码后的文本
    """
    try:
        return bytes.fromhex(hex_str).decode('utf-8')
    except Exception as e:
        raise ValueError(f"十六进制解码失败: {str(e)}")


@mcp.tool()
def list_to_csv(data: List[Dict[str, Any]]) -> str:
    """
    将列表数据转换为CSV格式
    
    Args:
        data: 字典列表，每个字典代表一行
        
    Returns:
        CSV格式的字符串
    """
    if not data:
        return ""
    
    # 获取所有键作为表头
    headers = list(data[0].keys())
    csv_lines = [",".join(headers)]
    
    # 添加数据行
    for row in data:
        values = [str(row.get(h, "")) for h in headers]
        csv_lines.append(",".join(values))
    
    return "\n".join(csv_lines)


@mcp.tool()
def format_json(json_str: str, indent: int = 2) -> str:
    """
    格式化JSON字符串
    
    Args:
        json_str: JSON字符串
        indent: 缩进空格数，默认2
        
    Returns:
        格式化后的JSON字符串
    """
    try:
        data = json.loads(json_str)
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except json.JSONDecodeError as e:
        raise ValueError(f"无效的JSON格式: {str(e)}")


@mcp.tool()
def minify_json(json_str: str) -> str:
    """
    压缩JSON字符串（移除空格和换行）
    
    Args:
        json_str: JSON字符串
        
    Returns:
        压缩后的JSON字符串
    """
    try:
        data = json.loads(json_str)
        return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    except json.JSONDecodeError as e:
        raise ValueError(f"无效的JSON格式: {str(e)}")


@mcp.tool()
def url_encode(text: str) -> str:
    """
    URL编码
    
    Args:
        text: 要编码的文本
        
    Returns:
        URL编码后的字符串
    """
    from urllib.parse import quote
    return quote(text)


@mcp.tool()
def url_decode(encoded_text: str) -> str:
    """
    URL解码
    
    Args:
        encoded_text: URL编码的字符串
        
    Returns:
        解码后的文本
    """
    from urllib.parse import unquote
    return unquote(encoded_text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002, help="Server port")
    args = parser.parse_args()
    
    # 运行服务器
    mcp.run(transport="sse", port=args.port)
