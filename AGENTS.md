# Repository Guidelines

## Project Structure & Module Organization
- `agents/` contains core agent modules and orchestration logic (see `agents/modules.py`).
- `mcp_servers/` provides example MCP servers (`*_server.py`) used by the tool layer.
- `toolbox/` hosts local Python tools loaded by the tool manager.
- Top-level Python entry points include `mcp_manager.py`, `tool_manager.py`, and `llm.py`.
- Configuration is in `config.yaml`; dependency metadata lives in `pyproject.toml` and `uv.lock`.

## Build, Test, and Development Commands
- `uv sync` installs dependencies from `pyproject.toml`/`uv.lock`.
- `uv run python mcp_manager.py start` starts all MCP servers; `stop`, `restart`, and `status` are also supported.
- `python mcp_manager.py status` works without `uv` if dependencies are already installed.
- `pytest` runs the test suite (add tests first; none are committed yet).

## Coding Style & Naming Conventions
- Python 3.12+ only (see `pyproject.toml`).
- Use 4-space indentation, type hints where helpful, and descriptive snake_case names for functions/variables.
- Public tool functions should be in `toolbox/*.py` and remain importable without side effects.

## Testing Guidelines
- Testing framework: `pytest` (listed in dependencies).
- Place tests under `tests/` and name files `test_*.py`.
- Prefer unit tests for tools and managers; add integration tests for MCP server behavior as needed.

## Commit & Pull Request Guidelines
- Commit history shows short, imperative messages (English or Chinese), e.g., "add readme" / "添加统一的工具管理类".
- Keep PRs focused; include a brief description, key changes, and any usage notes or screenshots if behavior is user-facing.

## Security & Configuration Tips
- Store secrets in environment variables (use `python-dotenv` if needed) rather than committing them to `config.yaml`.
- When adding new MCP servers, update the manager config and document ports/transport choices.
