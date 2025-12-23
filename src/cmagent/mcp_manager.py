#!/usr/bin/env python3
"""
MCP 服务器管理器 - 统一管理所有 MCP 服务

功能：
    - 自动发现和启动 MCP 服务器脚本
    - 进程生命周期管理（启动、停止、重启）
    - 状态监控和日志记录

使用方法：
    python -m cmagent.mcp_manager start       # 启动所有服务器
    python -m cmagent.mcp_manager stop        # 停止所有服务器
    python -m cmagent.mcp_manager restart     # 重启所有服务器
    python -m cmagent.mcp_manager status      # 查看服务器状态
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


# ==============================================================================
# 配置常量
# ==============================================================================

SERVERS_DIR = "mcp_servers"          # MCP 服务器脚本目录
BASE_PORT = 8000                      # 起始端口号
PIDS_FILE = ".mcp_pids.json"         # 进程ID存储文件
LOGS_DIR = "logs/mcp"                 # 日志目录


def _find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start


def _safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


# ==============================================================================
# MCP 服务器管理器
# ==============================================================================

class MCPManager:
    """MCP 服务器管理器 - 自动发现并管理所有 MCP 服务"""
    
    def __init__(self, servers_dir: str = SERVERS_DIR):
        """初始化管理器
        
        Args:
            servers_dir: MCP 服务器脚本所在目录（相对于项目根目录）
        """
        self.package_root = Path(__file__).resolve().parent
        self.project_root = _find_project_root(self.package_root)
        self.servers_dir = self._resolve_servers_dir(servers_dir)
        self.pids_file = self.project_root / PIDS_FILE
        self.logs_dir = self.project_root / LOGS_DIR
        
        # 确保目录存在
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 自动发现服务器
        self.servers = self._discover_servers()
    
    def _resolve_servers_dir(self, servers_dir: str) -> Path:
        package_path = self.package_root / servers_dir
        if package_path.exists():
            return package_path
        return self.project_root / servers_dir
    
    # ==========================================================================
    # 服务器发现
    # ==========================================================================
    
    def _discover_servers(self) -> Dict[str, Dict[str, any]]:
        """自动发现所有 MCP 服务器脚本
        
        规则：
            - 扫描 SERVERS_DIR 目录
            - 匹配 *_server.py 文件
            - 按文件名分配端口（从 BASE_PORT 开始）
        
        Returns:
            服务器配置字典 {name: {script: Path, port: int}}
        """
        if not self.servers_dir.exists():
            print(f"⚠️  服务器目录不存在: {self.servers_dir}")
            return {}
        
        servers = {}
        port = BASE_PORT
        
        # 扫描服务器脚本（排序以保证端口分配稳定）
        server_files = sorted(self.servers_dir.glob("*_server.py"))
        
        for script_path in server_files:
            # 提取服务器名称（去除 _server.py 后缀）
            name = script_path.stem.replace("_server", "")
            
            servers[name] = {
                "script": script_path,
                "port": port
            }
            port += 1
        
        return servers
    
    # ==========================================================================
    # 进程管理
    # ==========================================================================
    
    def _load_pids(self) -> Dict[str, int]:
        """加载进程ID记录"""
        if not self.pids_file.exists():
            return {}
        
        try:
            with open(self.pids_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  加载PID文件失败: {e}")
            return {}
    
    def _save_pids(self, pids: Dict[str, int]):
        """保存进程ID记录"""
        try:
            with open(self.pids_file, 'w') as f:
                json.dump(pids, f, indent=2)
        except Exception as e:
            print(f"⚠️  保存PID文件失败: {e}")
    
    def _is_running(self, pid: int) -> bool:
        """检查进程是否存活
        
        Args:
            pid: 进程ID
            
        Returns:
            True 如果进程存在且可访问
        """
        try:
            os.kill(pid, 0)  # 发送空信号测试进程
            return True
        except (OSError, ProcessLookupError):
            return False
    
    # ==========================================================================
    # 服务器操作
    # ==========================================================================
    
    def start_server(self, name: str, port: Optional[int] = None) -> bool:
        """启动单个服务器
        
        Args:
            name: 服务器名称
            port: 端口号（可选，默认使用配置值）
            
        Returns:
            True 如果启动成功
        """
        # 验证服务器存在
        if name not in self.servers:
            print(f"❌ 未知的服务器: {name}")
            print(f"   可用服务器: {', '.join(self.servers.keys())}")
            return False
        
        config = self.servers[name]
        script = config["script"]
        port = port or config["port"]
        
        # 检查是否已运行
        pids = self._load_pids()
        if name in pids and self._is_running(pids[name]):
            print(f"⚠️  {name} 已在运行 (PID: {pids[name]}, 端口: {port})")
            return True
        
        # 启动服务器
        log_file = self.logs_dir / f"{name}.log"
        print(f"🚀 启动 {name} 服务器...")
        print(f"   端口: {port}")
        print(f"   脚本: {_safe_relpath(script, self.project_root)}")
        print(f"   日志: {_safe_relpath(log_file, self.project_root)}")
        
        try:
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    ["uv", "run", str(script), "--port", str(port)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.project_root)
                )
            
            # 保存进程ID
            pids[name] = process.pid
            self._save_pids(pids)
            
            # 等待启动
            time.sleep(1)
            
            # 验证启动状态
            if self._is_running(process.pid):
                print(f"✅ {name} 启动成功 (PID: {process.pid})")
                return True
            else:
                print(f"❌ {name} 启动失败，请查看日志")
                return False
        
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return False
    
    def stop_server(self, name: str) -> bool:
        """停止单个服务器
        
        Args:
            name: 服务器名称
            
        Returns:
            True 如果停止成功
        """
        pids = self._load_pids()
        
        if name not in pids:
            print(f"⚠️  {name} 未运行")
            return False
        
        pid = pids[name]
        
        # 尝试优雅停止
        if self._is_running(pid):
            try:
                print(f"🛑 停止 {name} (PID: {pid})...")
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.5)
                
                # 如果仍在运行，强制终止
                if self._is_running(pid):
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.5)
                
                print(f"✅ {name} 已停止")
            except Exception as e:
                print(f"❌ 停止失败: {e}")
                return False
        else:
            print(f"⚠️  {name} 进程不存在 (PID: {pid})")
        
        # 删除PID记录
        del pids[name]
        self._save_pids(pids)
        return True
    
    def start_all(self):
        """启动所有服务器"""
        if not self.servers:
            print("⚠️  未发现任何服务器脚本")
            return
        
        print(f"🚀 启动所有 MCP 服务器 (共 {len(self.servers)} 个)")
        print("=" * 60)
        
        success_count = 0
        for name in self.servers:
            if self.start_server(name):
                success_count += 1
            print()  # 空行分隔
            time.sleep(0.5)  # 避免端口冲突
        
        print("=" * 60)
        print(f"✅ 启动完成: {success_count}/{len(self.servers)} 个服务器运行中")
    
    def stop_all(self):
        """停止所有服务器"""
        pids = self._load_pids()
        
        if not pids:
            print("⚠️  没有运行中的服务器")
            return
        
        print(f"🛑 停止所有 MCP 服务器 (共 {len(pids)} 个)")
        print("=" * 60)
        
        for name in list(pids.keys()):
            self.stop_server(name)
            print()
        
        print("=" * 60)
        print("✅ 所有服务器已停止")
    
    def restart_all(self):
        """重启所有服务器"""
        print("🔄 重启所有 MCP 服务器")
        print("=" * 60)
        
        self.stop_all()
        print()
        time.sleep(2)
        self.start_all()
    
    def show_status(self):
        """显示所有服务器状态"""
        print("📊 MCP 服务器状态")
        print("=" * 60)
        
        if not self.servers:
            print("⚠️  未发现任何服务器脚本")
            return
        
        pids = self._load_pids()
        
        # 表头
        print(f"{'服务器':<12} {'状态':<10} {'PID':<8} {'端口':<6} {'脚本'}")
        print("-" * 60)
        
        # 服务器列表
        for name, config in self.servers.items():
            port = config["port"]
            script = config["script"].name
            
            if name in pids and self._is_running(pids[name]):
                status = "✅ 运行中"
                pid = str(pids[name])
            else:
                status = "⚪ 已停止"
                pid = "-"
            
            print(f"{name:<12} {status:<10} {pid:<8} {port:<6} {script}")
        
        print("=" * 60)
        print(f"总计: {len(self.servers)} 个服务器, {len(pids)} 个运行中")


# ==============================================================================
# 命令行入口
# ==============================================================================

def main():
    """命令行入口"""
    manager = MCPManager()
    
    # 解析命令
    if len(sys.argv) < 2:
        command = "status"
    else:
        command = sys.argv[1].lower()
    
    # 执行命令
    if command == "start":
        manager.start_all()
    
    elif command == "stop":
        manager.stop_all()
    
    elif command == "restart":
        manager.restart_all()
    
    elif command == "status":
        manager.show_status()
    
    else:
        print("❌ 未知命令")
        print("\n使用方法:")
        print("  python -m cmagent.mcp_manager start       # 启动所有服务器")
        print("  python -m cmagent.mcp_manager stop        # 停止所有服务器")
        print("  python -m cmagent.mcp_manager restart     # 重启所有服务器")
        print("  python -m cmagent.mcp_manager status      # 查看服务器状态")
        sys.exit(1)


if __name__ == "__main__":
    main()
