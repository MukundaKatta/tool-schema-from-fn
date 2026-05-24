# tool-schema-from-fn

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tool-schema-from-fn.svg)](https://pypi.org/project/tool-schema-from-fn/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Turn a Python function into an Anthropic / OpenAI tool schema.** Zero deps. Reads the signature, reads the docstring, gives you back JSON your LLM can use.

```python
from tool_schema_from_fn import tool_schema, openai_function

def search_web(q: str, *, max_results: int = 10, timeout: int = 30) -> list[str]:
    """Search the web for a query.

    Args:
        q: The search query.
        max_results: How many results to return.
        timeout: Seconds before giving up.
    """
    ...

anthropic_tool = tool_schema(search_web)
# {
#   "name": "search_web",
#   "description": "Search the web for a query.",
#   "input_schema": {
#     "type": "object",
#     "properties": {
#       "q":           {"type": "string",  "description": "The search query."},
#       "max_results": {"type": "integer", "default": 10, "description": "How many results to return."},
#       "timeout":     {"type": "integer", "default": 30, "description": "Seconds before giving up."}
#     },
#     "required": ["q"]
#   }
# }

openai_tool = openai_function(search_web)
# {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
```

## Why

The first thing every agent framework does is turn function signatures into JSON Schema. Everyone writes it. Most rely on a heavy dependency (pydantic, attrs, msgspec) to do it.

`tool-schema-from-fn` does the small honest version of this work with no runtime deps. Uses `inspect` for the signature, `typing.get_origin/args` for generics, and a tiny regex pass for Google-style docstrings.

Supported types: `str`, `int`, `float`, `bool`, `bytes`, `list[T]`, `dict[str, T]`, `Optional[T]`, `T | None`, `Literal["a", "b"]`. Unknown types degrade to `{"type": "string"}` rather than crashing.

Supported docstring style: Google (first paragraph → description; `Args:` section → property descriptions). Variants `Arguments:` and `Parameters:` are also recognized.

## Install

```bash
pip install tool-schema-from-fn
```

## API

```python
from tool_schema_from_fn import (
    tool_schema,           # Anthropic shape
    openai_function,       # OpenAI shape
    type_to_json_schema,   # exposed for custom builders
    parse_google_docstring,
)

tool_schema(fn, name=None, description=None, skip_params=None) -> dict
openai_function(fn, name=None, description=None, skip_params=None) -> dict
```

`skip_params` is the most useful keyword — it lets you drop framework-injected args (`ctx`, `request`, `session`) before exposing the tool to the LLM. `self` and `cls` are dropped automatically.

## Companion libraries

- [`agentvet`](https://github.com/MukundaKatta/agentvet) — validate the LLM's args at runtime against the generated schema.
- [`tool-arg-defaults`](https://github.com/MukundaKatta/tool-arg-defaults) — fill in defaults before validation.
- [`agentcast`](https://github.com/MukundaKatta/agentcast) — validate the *output* of the tool / LLM response.

## License

MIT
