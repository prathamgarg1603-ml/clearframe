import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from io import StringIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI


BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
MAX_TOOL_ROUNDS = 10
WORKER_PATH = Path(__file__).with_name("sandbox_worker.py")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "describe_data",
            "description": "Summarize the current dataframe, its columns, types, missingness, and sample rows.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run pandas code in a restricted subprocess with a timeout. Use df for the dataframe; leave the cleaned result in df. pd and np are provided, so do not import them.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python statements that transform df."}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": "Create a chart from the current dataframe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["bar", "line", "scatter", "hist"]},
                    "x": {"type": "string", "description": "Column for x axis; not used for hist."},
                    "y": {"type": "string", "description": "Column for y axis; for hist this is the measured column."},
                    "title": {"type": "string"},
                },
                "required": ["kind", "y", "title"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = """You are a careful data-cleaning analyst. Inspect the uploaded dataframe with describe_data before changing it. Use run_python for auditable transformations, and make_chart when a useful chart is requested. The current dataframe is named df; pd and np are already available, and imports are disabled. Keep the cleaned result in df. Check your work after transformations. If a tool returns an error, diagnose it and try corrected code. Never claim a change succeeded unless a tool confirms it. Finish with a concise report of changes, caveats, and output columns."""


class AgentError(RuntimeError):
    pass


class DataAgent:
    def __init__(self, api_key, model=DEFAULT_MODEL, client=None, timeout_seconds=8):
        if not api_key:
            raise ValueError("Set NEBIUS_API_KEY before starting the agent.")
        self.client = client or OpenAI(api_key=api_key, base_url=BASE_URL)
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.frame = None
        self.chart_path = None

    def smoke_test(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Reply with exactly: Token Factory connection works."}],
            temperature=0,
            max_tokens=30,
        )
        return response.choices[0].message.content

    def describe_data(self):
        frame = self.frame
        columns = []
        for column in frame.columns:
            series = frame[column]
            info = {
                "name": str(column),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 2),
                "unique": int(series.nunique(dropna=True)),
            }
            non_null = series.dropna()
            if pd.api.types.is_numeric_dtype(series) and len(non_null):
                info.update(
                    min=float(non_null.min()), max=float(non_null.max()),
                    mean=round(float(non_null.mean()), 3), median=float(non_null.median()),
                )
            elif len(non_null):
                text = non_null.astype(str)
                numeric = pd.to_numeric(text, errors="coerce")
                if 0.8 < numeric.notna().mean() < 1:
                    info["non_numeric_values_in_mostly_numeric_column"] = text[numeric.isna()].unique().tolist()[:5]
                if numeric.notna().mean() < 0.5:
                    variants = text.groupby(text.str.strip().str.lower()).agg(lambda x: sorted(set(x)))
                    mixed = [v[:4] for v in variants if len(v) > 1][:5]
                    if mixed:
                        info["spelling_or_case_variants"] = mixed
                info["top_values"] = text.value_counts().head(5).to_dict()
            columns.append(info)
        summary = {
            "shape": {"rows": int(len(frame)), "columns": int(len(frame.columns))},
            "exact_duplicate_rows": int(frame.duplicated().sum()),
            "repeated_id_values": {
                str(c): int(frame[c].dropna().duplicated().sum())
                for c in frame.columns if str(c).lower().endswith("id")
            },
            "columns": columns,
            "sample_rows": frame.head(5).astype(object).where(pd.notna(frame.head(5)), None).to_dict(orient="records"),
        }
        return json.dumps(summary, default=str)

    def run_python(self, code):
        payload = json.dumps({"code": code, "dataframe_json": self.frame.to_json(orient="table", date_format="iso")})
        with tempfile.TemporaryDirectory(prefix="data-agent-") as workdir:
            env = {"PATH": os.environ.get("PATH", "")}
            if os.name == "nt" and os.environ.get("SYSTEMROOT"):
                env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(WORKER_PATH)],
                    input=payload,
                    capture_output=True,
                    text=True,
                    cwd=workdir,
                    env=env,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"Python tool exceeded its {self.timeout_seconds}-second limit.")
        try:
            result = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            detail = completed.stderr[-2000:] or completed.stdout[-2000:] or "Worker exited without a result."
            raise RuntimeError(detail)
        if completed.returncode != 0 or "error" in result:
            raise RuntimeError(result.get("traceback", result.get("error", completed.stderr[-2000:])))
        self.frame = pd.read_json(StringIO(result["dataframe_json"]), orient="table")
        message = "Transformation applied."
        if result.get("stdout"):
            message += " Output: " + result["stdout"]
        return message

    def make_chart(self, kind, y, title, x=None):
        if y not in self.frame.columns or (x and x not in self.frame.columns):
            raise ValueError("Chart columns must exist in the current dataframe.")
        if kind != "hist" and not x:
            raise ValueError("Provide x for bar, line, and scatter charts.")
        figure, axis = plt.subplots(figsize=(8, 4.5))
        if kind == "hist":
            self.frame[y].dropna().plot(kind="hist", ax=axis, bins=20)
        elif kind == "scatter":
            axis.scatter(self.frame[x], self.frame[y], alpha=0.7)
        else:
            self.frame.plot(kind=kind, x=x, y=y, ax=axis, legend=False)
        axis.set_title(title)
        figure.tight_layout()
        handle = tempfile.NamedTemporaryFile(prefix="data-agent-chart-", suffix=".png", delete=False)
        self.chart_path = handle.name
        handle.close()
        figure.savefig(self.chart_path, dpi=140)
        plt.close(figure)
        return "Chart created."

    def _execute_tool(self, name, arguments):
        if name == "describe_data":
            return self.describe_data()
        if name == "run_python":
            return self.run_python(arguments["code"])
        if name == "make_chart":
            return self.make_chart(**arguments)
        raise ValueError(f"Unknown tool: {name}")

    def run(self, frame, task, on_step=None):
        self.frame = frame.copy()
        self.chart_path = None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        for round_number in range(1, MAX_TOOL_ROUNDS + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump(exclude_none=True))
            if not assistant_message.tool_calls:
                return assistant_message.content or "The model returned an empty response.", self.frame.copy(), self.chart_path
            for call in assistant_message.tool_calls:
                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    result = self._execute_tool(name, arguments)
                    tool_content = str(result)
                    status = "completed"
                except Exception as error:
                    tool_content = f"Tool error: {type(error).__name__}: {error}"
                    status = "error"
                if on_step:
                    on_step({"round": round_number, "tool": name, "status": status, "detail": tool_content[:1200]})
                messages.append({"role": "tool", "tool_call_id": call.id, "name": name, "content": tool_content})
        raise AgentError(f"Agent reached the {MAX_TOOL_ROUNDS}-round limit without a final answer.")