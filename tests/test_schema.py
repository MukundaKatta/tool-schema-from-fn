"""Tests for tool_schema_from_fn."""

from __future__ import annotations

from typing import Literal, Optional


from tool_schema_from_fn import (
    openai_function,
    parse_google_docstring,
    tool_schema,
    type_to_json_schema,
)


# ---- type_to_json_schema -------------------------------------------------


def test_primitives():
    assert type_to_json_schema(str) == {"type": "string"}
    assert type_to_json_schema(int) == {"type": "integer"}
    assert type_to_json_schema(float) == {"type": "number"}
    assert type_to_json_schema(bool) == {"type": "boolean"}


def test_list_of_str():
    assert type_to_json_schema(list[str]) == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_dict_str_int():
    assert type_to_json_schema(dict[str, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


def test_optional_strips_none():
    assert type_to_json_schema(Optional[int]) == {"type": "integer"}


def test_pep604_union_with_none():
    assert type_to_json_schema(int | None) == {"type": "integer"}


def test_pep604_union_multi():
    out = type_to_json_schema(int | str)
    assert out == {"anyOf": [{"type": "integer"}, {"type": "string"}]}


def test_literal_becomes_enum():
    out = type_to_json_schema(Literal["a", "b", "c"])
    assert out == {"enum": ["a", "b", "c"]}


def test_unknown_type_falls_back_to_string():
    class _Custom:
        pass

    assert type_to_json_schema(_Custom) == {"type": "string"}


def test_bare_list_no_items():
    assert type_to_json_schema(list) == {"type": "array"}


# ---- parse_google_docstring ----------------------------------------------


def test_parse_empty_docstring():
    assert parse_google_docstring(None) == ("", {})
    assert parse_google_docstring("") == ("", {})


def test_parse_top_description_only():
    top, args = parse_google_docstring("Do a thing.\n\nMore details here.")
    assert "Do a thing." in top
    assert args == {}


def test_parse_args_section():
    doc = """Search the web.

    Args:
        q: The query.
        max_results: How many to return.
    """
    top, args = parse_google_docstring(doc)
    assert "Search the web" in top
    assert args == {
        "q": "The query.",
        "max_results": "How many to return.",
    }


def test_parse_args_with_type_annotation_in_parens():
    doc = """Stuff.

    Args:
        q (str): The query.
        n (int): How many.
    """
    _, args = parse_google_docstring(doc)
    assert args == {"q": "The query.", "n": "How many."}


def test_parse_stops_at_returns():
    doc = """Top.

    Args:
        q: query

    Returns:
        list of stuff
    """
    _, args = parse_google_docstring(doc)
    assert args == {"q": "query"}


def test_parse_arguments_alias():
    doc = """Top.

    Arguments:
        q: query
    """
    _, args = parse_google_docstring(doc)
    assert args == {"q": "query"}


# ---- tool_schema ---------------------------------------------------------


def test_tool_schema_basic():
    def search_web(q: str, *, max_results: int = 10) -> list[str]:
        """Search the web for a query.

        Args:
            q: The search query.
            max_results: How many results to return.
        """
        ...

    s = tool_schema(search_web)
    assert s["name"] == "search_web"
    assert "Search the web" in s["description"]
    props = s["input_schema"]["properties"]
    assert props["q"] == {"type": "string", "description": "The search query."}
    assert props["max_results"]["type"] == "integer"
    assert props["max_results"]["default"] == 10
    assert s["input_schema"]["required"] == ["q"]


def test_tool_schema_optional_param_not_required():
    def f(q: str, page: Optional[int] = None):
        """f.

        Args:
            q: query
            page: pagination
        """
        ...

    s = tool_schema(f)
    assert s["input_schema"]["required"] == ["q"]
    # `page` is Optional → schema is plain int (None stripped)
    assert s["input_schema"]["properties"]["page"]["type"] == "integer"
    # A `None` default is the "unset" sentinel — it should NOT surface as
    # `"default": null`, which would mislead the LLM into passing null.
    assert "default" not in s["input_schema"]["properties"]["page"]


def test_tool_schema_none_default_omitted_but_real_default_kept():
    def f(a: int = 5, b: str = None):
        """f.

        Args:
            a: a number with a real default
            b: an optional string defaulting to None
        """
        ...

    s = tool_schema(f)
    props = s["input_schema"]["properties"]
    # A real (non-None) default is reflected.
    assert props["a"]["default"] == 5
    # A None default is omitted entirely.
    assert "default" not in props["b"]
    # Neither is required (both have defaults).
    assert "required" not in s["input_schema"]


def test_tool_schema_explicit_name_and_description():
    def f(q: str):
        """blah."""
        ...

    s = tool_schema(f, name="other_name", description="overridden")
    assert s["name"] == "other_name"
    assert s["description"] == "overridden"


def test_tool_schema_skip_params():
    def f(ctx, q: str):
        """f.

        Args:
            q: query
        """
        ...

    s = tool_schema(f, skip_params=["ctx"])
    assert "ctx" not in s["input_schema"]["properties"]
    assert "q" in s["input_schema"]["properties"]


def test_tool_schema_skips_self_and_cls():
    class Tools:
        def method(self, q: str): ...

        @classmethod
        def klass(cls, q: str): ...

    s_m = tool_schema(Tools.method)
    assert "self" not in s_m["input_schema"]["properties"]
    s_k = tool_schema(Tools.klass)
    assert "cls" not in s_k["input_schema"]["properties"]


def test_tool_schema_skips_var_args_and_kwargs():
    def f(q: str, *args, **kwargs): ...

    s = tool_schema(f)
    assert set(s["input_schema"]["properties"]) == {"q"}


def test_tool_schema_with_list_and_dict_annotations():
    def f(items: list[str], meta: dict[str, int] = None): ...

    s = tool_schema(f)
    props = s["input_schema"]["properties"]
    assert props["items"] == {"type": "array", "items": {"type": "string"}}
    assert props["meta"]["type"] == "object"


def test_tool_schema_required_only_when_no_default_and_not_optional():
    def f(a: str, b: int = 5, c: Optional[float] = None, d: str = "x"): ...

    s = tool_schema(f)
    assert s["input_schema"]["required"] == ["a"]


def test_tool_schema_no_required_section_when_empty():
    def f(a: str = "x"): ...

    s = tool_schema(f)
    assert "required" not in s["input_schema"]


# ---- openai_function -----------------------------------------------------


def test_openai_function_shape():
    def search(q: str):
        """Search.

        Args:
            q: query
        """
        ...

    s = openai_function(search)
    assert s["type"] == "function"
    assert s["function"]["name"] == "search"
    assert s["function"]["parameters"]["properties"]["q"]["type"] == "string"
