import ast
import json
import sys
import traceback
from io import StringIO

import numpy as np
import pandas as pd


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
}

SAFE_CALLS = set(SAFE_BUILTINS) | {
    "to_datetime",
    "to_numeric",
    "where",
    "select",
    "concat",
}
SAFE_METHODS = {
    "agg",
    "astype",
    "clip",
    "copy",
    "drop",
    "drop_duplicates",
    "dropna",
    "fillna",
    "groupby",
    "head",
    "isna",
    "lower",
    "notna",
    "replace",
    "reset_index",
    "rename",
    "round",
    "sort_values",
    "strip",
    "to_numeric",
    "to_datetime",
    "value_counts",
    "upper",
    # added: stats, string and mapping helpers commonly needed for cleaning
    "abs", "apply", "between", "contains", "count", "endswith", "extract",
    "idxmax", "idxmin", "isin", "isnull", "map", "max", "mean", "median",
    "min", "mode", "notnull", "nunique", "quantile", "split", "startswith",
    "std", "sum", "title", "transform",
}
BLOCKED_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


class CodePolicy(ast.NodeVisitor):
    def visit_Import(self, node):
        raise ValueError("Imports are disabled; use the provided pd, np, and df objects.")

    def visit_ImportFrom(self, node):
        raise ValueError("Imports are disabled; use the provided pd, np, and df objects.")

    def visit_Name(self, node):
        if node.id.startswith("_") or node.id in BLOCKED_NAMES:
            raise ValueError(f"Use of {node.id!r} is not allowed.")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("_"):
            raise ValueError("Private and dunder attributes are not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node):
        function = node.func
        if isinstance(function, ast.Name):
            allowed = function.id in SAFE_CALLS
        elif isinstance(function, ast.Attribute):
            allowed = function.attr in SAFE_METHODS | {"to_datetime", "to_numeric", "where", "select", "concat"}
        else:
            allowed = False
        if not allowed:
            name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", "this call")
            raise ValueError(
                f"'{name}' is not available in the restricted Python tool. "
                "Use another approach, e.g. built-in pandas methods like fillna, replace, "
                "astype, to_numeric, clip, str.strip/lower/title/extract."
            )
        self.generic_visit(node)


def main():
    request = json.load(sys.stdin)
    frame = pd.read_json(StringIO(request["dataframe_json"]), orient="table")
    tree = ast.parse(request["code"], mode="exec")
    CodePolicy().visit(tree)
    namespace = {"__builtins__": SAFE_BUILTINS, "df": frame, "pd": pd, "np": np}
    output = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = output
        exec(compile(tree, "<agent-code>", "exec"), namespace, namespace)
    finally:
        sys.stdout = old_stdout
    frame = namespace.get("df")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Your code must leave a pandas DataFrame named df.")
    result = {
        "dataframe_json": frame.to_json(orient="table", date_format="iso"),
        "stdout": output.getvalue()[:4000],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": str(error), "traceback": traceback.format_exc()[-4000:]}))
        sys.exit(1)