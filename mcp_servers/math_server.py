"""
数学计算 MCP 服务器
提供基础数学运算工具
"""
import math
from fastmcp import FastMCP

# 创建 MCP 服务器实例
mcp = FastMCP("Math Calculator")


@mcp.tool()
def add(a: float, b: float) -> float:
    """
    两数相加
    
    Args:
        a: 第一个数字
        b: 第二个数字
        
    Returns:
        两数之和
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """
    两数相减
    
    Args:
        a: 被减数
        b: 减数
        
    Returns:
        差值
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    两数相乘
    
    Args:
        a: 第一个数字
        b: 第二个数字
        
    Returns:
        乘积
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """
    两数相除
    
    Args:
        a: 被除数
        b: 除数
        
    Returns:
        商
        
    Raises:
        ValueError: 当除数为0时
    """
    if b == 0:
        raise ValueError("除数不能为0")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """
    计算幂次方
    
    Args:
        base: 底数
        exponent: 指数
        
    Returns:
        幂次方结果
    """
    return math.pow(base, exponent)


@mcp.tool()
def square_root(number: float) -> float:
    """
    计算平方根
    
    Args:
        number: 要计算平方根的数字（必须非负）
        
    Returns:
        平方根
        
    Raises:
        ValueError: 当输入负数时
    """
    if number < 0:
        raise ValueError("不能计算负数的平方根")
    return math.sqrt(number)


@mcp.tool()
def factorial(n: int) -> int:
    """
    计算阶乘
    
    Args:
        n: 非负整数
        
    Returns:
        n的阶乘
        
    Raises:
        ValueError: 当输入负数或非整数时
    """
    if n < 0:
        raise ValueError("阶乘只能计算非负整数")
    return math.factorial(n)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()
    
    # 运行服务器
    mcp.run(transport="sse", port=args.port)
