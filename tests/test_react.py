import logging
from datetime import datetime
from pathlib import Path

from cmagent.modules import ReActAgent
from cmagent.utils import module_output_printer

# 配置日志输出到文件
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"test_react_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8')  # 只输出到文件
    ]
)

print(f"Logs will be saved to: {log_file}")

react_agent = ReActAgent(
    model_name="qwen3-14b",
    mcp_config={
        "mcpServers": {
            "baidu-map": {
              "command": "npx",
              "args": ["-y", "@baidumap/mcp-server-baidu-map"],
              "env": {
                "BAIDU_MAP_API_KEY": "rakR78VnBT9ibC98jwppeUL8M94VnjEG"
              }
            },
            "math": {
              "url": "http://localhost:8001/sse",
              "transport": "sse"
            }
        }
    },
    local_tools_folder="./toolbox"
)

messages = [
    {"role": "user", "content": "计算这周长沙的平均温度"}
]

module_output_printer(
    react_agent(messages)
)
