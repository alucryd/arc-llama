# User Skills

arc-llama supports user-defined skills: plain Python files that register custom
tools for the agent. Skills are loaded once at server startup.

## Skills directory

By default the server loads skills from:

```
~/.config/arc-llama/skills
```

You can change this path with the `paths.skills_dir` option in `config.toml`:

```toml
[paths]
skills_dir = "~/.config/arc-llama/skills"
```

## Writing a skill

Create a file ending in `.py` inside the skills directory. Import the `tool`
decorator from `arc_llama.agent.tools` and decorate a function with the tool
metadata:

```python
from arc_llama.agent.tools import tool, ToolResult

@tool(
    name="get_timestamp",
    description="Return the current UTC time in ISO 8601 format.",
    parameters={"type": "object", "properties": {}},
)
def _get_timestamp(arguments, ctx):
    from datetime import datetime, timezone
    return ToolResult(datetime.now(timezone.utc).isoformat())
```

The function receives two arguments:

* `arguments`: a dict of arguments passed by the LLM.
* `ctx`: a `ToolContext` with `root` (project `Path`), `client`
  (`httpx.AsyncClient`), and `extra` (shared state such as `chat_store`).

A tool may return a `ToolResult`, a plain string, or (if `async`) an awaitable
of either.

## Safety

* Broken skills are logged and skipped; they do not prevent the server from
  starting.
* Skills can register tools with the same name as built-in tools, replacing the
  built-in implementation.
* Tools that perform destructive actions should set
  `requires_confirmation=True` so the UI can prompt for approval.

## Example

See `skills/example_skill.py` in this repository for a minimal working skill.
It is not loaded automatically; copy it into your configured skills directory
and restart the server to try it.
