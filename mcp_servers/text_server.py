"""
文本处理 MCP 服务器
提供字符串操作和文本分析工具
"""
import re
from typing import List, Dict
from fastmcp import FastMCP

# 创建 MCP 服务器实例
mcp = FastMCP("Text Processor")


@mcp.tool()
def reverse_string(text: str) -> str:
    """
    反转字符串
    
    Args:
        text: 要反转的字符串
        
    Returns:
        反转后的字符串
    """
    return text[::-1]


@mcp.tool()
def count_words(text: str) -> int:
    """
    统计文本中的单词数量
    
    Args:
        text: 要统计的文本
        
    Returns:
        单词数量
    """
    words = text.split()
    return len(words)


@mcp.tool()
def count_characters(text: str, include_spaces: bool = False) -> int:
    """
    统计字符数量
    
    Args:
        text: 要统计的文本
        include_spaces: 是否包含空格，默认False
        
    Returns:
        字符数量
    """
    if include_spaces:
        return len(text)
    else:
        return len(text.replace(" ", ""))


@mcp.tool()
def to_uppercase(text: str) -> str:
    """
    转换为大写
    
    Args:
        text: 输入文本
        
    Returns:
        大写文本
    """
    return text.upper()


@mcp.tool()
def to_lowercase(text: str) -> str:
    """
    转换为小写
    
    Args:
        text: 输入文本
        
    Returns:
        小写文本
    """
    return text.lower()


@mcp.tool()
def capitalize_words(text: str) -> str:
    """
    将每个单词首字母大写
    
    Args:
        text: 输入文本
        
    Returns:
        每个单词首字母大写的文本
    """
    return text.title()


@mcp.tool()
def remove_punctuation(text: str) -> str:
    """
    移除标点符号
    
    Args:
        text: 输入文本
        
    Returns:
        移除标点后的文本
    """
    return re.sub(r'[^\w\s]', '', text)


@mcp.tool()
def find_and_replace(text: str, find: str, replace: str, case_sensitive: bool = True) -> str:
    """
    查找并替换文本
    
    Args:
        text: 原始文本
        find: 要查找的字符串
        replace: 替换的字符串
        case_sensitive: 是否区分大小写，默认True
        
    Returns:
        替换后的文本
    """
    if case_sensitive:
        return text.replace(find, replace)
    else:
        pattern = re.compile(re.escape(find), re.IGNORECASE)
        return pattern.sub(replace, text)


@mcp.tool()
def extract_emails(text: str) -> List[str]:
    """
    提取文本中的所有邮箱地址
    
    Args:
        text: 输入文本
        
    Returns:
        邮箱地址列表
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails


@mcp.tool()
def extract_urls(text: str) -> List[str]:
    """
    提取文本中的所有URL
    
    Args:
        text: 输入文本
        
    Returns:
        URL列表
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls


@mcp.tool()
def text_statistics(text: str) -> Dict[str, int]:
    """
    获取文本的详细统计信息
    
    Args:
        text: 输入文本
        
    Returns:
        包含各种统计信息的字典
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    return {
        "total_characters": len(text),
        "characters_no_spaces": len(text.replace(" ", "")),
        "total_words": len(words),
        "total_sentences": len(sentences),
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
        "total_lines": len(text.splitlines())
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    args = parser.parse_args()
    
    # 运行服务器
    mcp.run(transport="sse", port=args.port)
