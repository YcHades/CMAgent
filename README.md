
> A modular Agent framework built on the **Msg In / Msg Out** paradigm.

This project is a **modular and extensible Agent framework** designed around a unified **Message In / Message Out** abstraction.
Agents, tools, and execution nodes communicate exclusively through standardized messages, enabling clean decoupling and flexible composition.

The framework provides a **unified tool management system** that supports both:

* **MCP (Model Context Protocol) tools**
* **Local Python tools**

By abstracting tool invocation and execution flow, the framework enables **model-agnostic, tool-agnostic, and composable Agent architectures**, making it suitable for building complex multi-agent systems.

**Project Layout (src-based):**

* Core agent modules live in `src/cmagent/modules/`.
* Tooling and runtime utilities are under `src/cmagent/` (e.g., `llm.py`, `tool_manager.py`, `mcp_manager.py`).
* `toolbox/` hosts local Python tools; `mcp_servers/` includes sample MCP servers.
* Top-level scripts (`mcp_manager.py`, `tool_manager.py`, `llm.py`) are repo shims for the package.

**Key Features:**

* 🧩 Msg In / Msg Out–driven modular architecture
* 🔌 Unified tool abstraction for MCP and local Python tools
* 🧠 Decoupled Agents, tools, and execution logic
* 🔄 Composable and extensible execution pipelines
* 🛠️ Designed for rapid prototyping and production-ready Agent systems

## Quickstart

### 1) Install dependencies

```bash
uv sync
```

### 2) Configure models

Edit `config.yaml` to set your `llm` model configuration (e.g., `model`, `api_key`, `base_url`).

### 3) Run MCP servers (optional)

```bash
uv run python mcp_manager.py start
```

Use `stop`, `restart`, and `status` as needed.

### 4) Try the module loop

```bash
uv run python -m cmagent.modules.modules
```

This runs the demo loop defined in `src/cmagent/modules/modules.py`.

## Basic Usage

### Use the LLM manager

```python
from cmagent.llm import LLMManager

manager = LLMManager("config.yaml")
print(manager.chat("YourModelName", "你好"))
```

### Use tool manager with local tools

```python
from cmagent.tool_manager import ToolManager

tm = ToolManager()
tm.load_from_folder("toolbox")
print(tm.call_tool("hello", {"name": "World"}))
```

### Compose agent modules

```python
from cmagent.modules import ToolUseLoopModule, module_output_printer

tool_loop = ToolUseLoopModule(
    model_name="YourModelName",
    local_tools_folder="toolbox"
)

module_output_printer(tool_loop([
    {"role": "system", "content": "你是一个Agent，需要时使用工具。"},
    {"role": "user", "content": "请用工具打招呼"}
]))
```

## Adding Tools

- Add local tools under `toolbox/` as simple Python functions.
- MCP servers live in `mcp_servers/` and are managed via `mcp_manager.py`.

## Development Tips

- Prefer `uv run` for scripts so dependencies are isolated.
- Tests go under `tests/` and run with `pytest`.
