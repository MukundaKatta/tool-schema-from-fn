"""tool-schema-from-fn - turn a Python function into a tool schema.

Anthropic's Tool Use API, OpenAI's Function Calling, Bedrock Converse, and
every other major LLM provider take a JSON Schema description of each
tool. Writing those by hand is tedious and tends to drift from the
underlying function signature.

`tool_schema()` walks a function's signature and Google-style docstring
to produce an `{name, description, input_schema}` block ready for
Anthropic's `tools=[...]`. `openai_function()` returns the same content in
OpenAI's `{type, function: {name, description, parameters}}` shape.

    from tool_schema_from_fn import tool_schema, openai_function

    def search_web(q: str, *, max_results: int = 10, timeout: int = 30) -> list[str]:
        '''Search the web for a query.

        Args:
            q: The search query.
            max_results: How many results to return.
            timeout: Seconds before giving up.
        '''
        ...

    anthropic_tool = tool_schema(search_web)
    openai_tool    = openai_function(search_web)

Supported types out of the box: str, int, float, bool, list, dict, plus
`list[T]` / `dict[str, T]` for primitive T. Optional[T] / T | None marks
the param as not required. Default values are reflected as `default` in
the schema.

Docstring parsing is best-effort Google-style. The first paragraph is the
function description; an `Args:` section maps `param_name: description`
lines onto schema properties.
"""

from __future__ import annotations

import inspect
import re
import typing
from typing import Any, Callable, Iterable, get_type_hints

__version__ = "0.1.0"
__all__ = [
    "tool_schema",
    "openai_function",
    "type_to_json_schema",
    "parse_google_docstring",
]


# ---- type → JSON Schema ---------------------------------------------------


_PRIM_MAP: dict[type, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    type(None): {"type": "null"},
    list: {"type": "array"},
    dict: {"type": "object"},
    bytes: {"type": "string", "contentEncoding": "base64"},
}


def type_to_json_schema(tp: Any) -> dict[str, Any]:
    """Map a Python type annotation to a JSON Schema fragment.

    Supports: primitives, `list[T]`, `dict[str, T]`, `Optional[T]`, `T | None`,
    `Literal[...]`. Unknown types degrade to `{"type": "string"}` rather than
    raise — we'd rather produce a usable (if loose) schema than crash on
    a complex annotation.
    """
    if tp is inspect.Parameter.empty:
        return {"type": "string"}

    if tp in _PRIM_MAP:
        return dict(_PRIM_MAP[tp])

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin in (list, typing.List):  # type: ignore[attr-defined]
        if args:
            return {"type": "array", "items": type_to_json_schema(args[0])}
        return {"type": "array"}

    if origin in (dict, typing.Dict):  # type: ignore[attr-defined]
        if len(args) == 2:
            return {
                "type": "object",
                "additionalProperties": type_to_json_schema(args[1]),
            }
        return {"type": "object"}

    if origin is typing.Union or _is_union_pep604(tp):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return type_to_json_schema(non_none[0])
        return {"anyOf": [type_to_json_schema(a) for a in non_none]}

    if origin is typing.Literal:
        return {"enum": list(args)}

    return {"type": "string"}


def _is_union_pep604(tp: Any) -> bool:
    """Detect `X | Y` (PEP 604) unions on Python 3.10+."""
    # Python 3.10+ exposes `types.UnionType`, but it's import-tied to
    # the runtime; we sniff for "__or__" + an `__args__` instead.
    return hasattr(tp, "__args__") and type(tp).__name__ == "UnionType"


# ---- docstring parsing ---------------------------------------------------


_DESCRIPTION_HEADERS = re.compile(
    r"^(Args|Arguments|Parameters|Returns|Return|Raises|Yields|Example|Examples|Note|Notes):\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_ARGS_HEADER = re.compile(
    r"^(?:Args|Arguments|Parameters):\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_ARG_LINE = re.compile(
    r"^(?P<indent>\s+)(?P<name>\w+)(?:\s*\(.*?\))?\s*:\s*(?P<desc>.*)$"
)


def parse_google_docstring(docstring: str | None) -> tuple[str, dict[str, str]]:
    """Return (top-level description, {param_name: description}).

    The top-level description is whatever appears before the first
    section header. The Args section maps each `name: description` line
    (with optional indented continuation) to its parameter.
    """
    if not docstring:
        return "", {}

    cleaned = inspect.cleandoc(docstring)

    # find the first section header (Args / Returns / etc.)
    first_header = _DESCRIPTION_HEADERS.search(cleaned)
    if first_header:
        top = cleaned[: first_header.start()].strip()
        rest = cleaned[first_header.start():]
    else:
        top = cleaned.strip()
        rest = ""

    arg_descs: dict[str, str] = {}
    args_match = _ARGS_HEADER.search(rest)
    if not args_match:
        return top, arg_descs

    body = rest[args_match.end():]
    # stop at the next header
    next_section = _DESCRIPTION_HEADERS.search(body)
    if next_section:
        body = body[: next_section.start()]

    current_name: str | None = None
    current_lines: list[str] = []
    base_indent: int | None = None

    for line in body.splitlines():
        match = _ARG_LINE.match(line)
        if match and (base_indent is None or len(match.group("indent")) <= base_indent):
            # flush previous
            if current_name is not None:
                arg_descs[current_name] = " ".join(current_lines).strip()
            current_name = match.group("name")
            current_lines = [match.group("desc").strip()]
            base_indent = len(match.group("indent"))
        elif current_name is not None and line.strip():
            # continuation
            current_lines.append(line.strip())

    if current_name is not None:
        arg_descs[current_name] = " ".join(current_lines).strip()

    return top, arg_descs


# ---- main entrypoint ------------------------------------------------------


def tool_schema(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    skip_params: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return an Anthropic-style tool schema for `fn`.

    Result shape: `{"name", "description", "input_schema"}` where
    `input_schema` is a JSON Schema object describing the parameters.

    Args:
        fn: the function to introspect.
        name: optional override for the tool name (defaults to fn.__name__).
        description: optional override (defaults to the first paragraph of
            the docstring).
        skip_params: parameter names to omit from the schema (useful for
            framework-injected args like `ctx` or `request`).
    """
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn)
    top_desc, arg_descs = parse_google_docstring(doc)
    # Resolve stringified annotations (PEP 563 / from __future__ import annotations).
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    skip = set(skip_params or ())

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in skip or pname == "self" or pname == "cls":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        ann = hints.get(pname, param.annotation)
        schema = type_to_json_schema(ann)
        desc = arg_descs.get(pname)
        if desc:
            schema["description"] = desc

        if param.default is not inspect.Parameter.empty:
            schema["default"] = param.default
        else:
            # required if no default AND not optional in the type
            if not _is_optional(ann):
                required.append(pname)

        properties[pname] = schema

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        input_schema["required"] = required

    return {
        "name": name or fn.__name__,
        "description": description if description is not None else top_desc,
        "input_schema": input_schema,
    }


def openai_function(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    skip_params: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return an OpenAI function-calling schema for `fn`.

    Shape: `{"type": "function", "function": {"name", "description", "parameters"}}`.
    """
    inner = tool_schema(fn, name=name, description=description, skip_params=skip_params)
    return {
        "type": "function",
        "function": {
            "name": inner["name"],
            "description": inner["description"],
            "parameters": inner["input_schema"],
        },
    }


def _is_optional(ann: Any) -> bool:
    if ann is inspect.Parameter.empty:
        return False
    if isinstance(ann, str):
        # forward-ref string that we couldn't resolve; play it safe
        return False
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is typing.Union or _is_union_pep604(ann):
        return type(None) in args
    return False
