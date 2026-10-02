import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from agent import DataAgent


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return next(self.responses)


def test_run_python_transforms_dataframe():
    agent = DataAgent("test-key", client=object())
    agent.frame = pd.DataFrame({"name": [" Ada ", "Lin "], "score": [1, None]})

    result = agent.run_python("df['name'] = df['name'].str.strip(); df = df.dropna(subset=['score'])")

    assert result == "Transformation applied."
    assert agent.frame["name"].tolist() == ["Ada"]


@pytest.mark.parametrize(
    "code",
    ["import os", "df = pd.read_csv('secret.csv')", "print(open('secret.txt').read())"],
)
def test_run_python_rejects_unsafe_operations(code):
    agent = DataAgent("test-key", client=object())
    agent.frame = pd.DataFrame({"value": [1]})

    with pytest.raises(RuntimeError):
        agent.run_python(code)


def test_run_python_times_out():
    agent = DataAgent("test-key", client=object(), timeout_seconds=0.01)
    agent.frame = pd.DataFrame({"value": [1]})

    with pytest.raises(TimeoutError):
        agent.run_python("while True: pass")


def test_tool_error_is_returned_to_model_and_repair_succeeds():
    class Message:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

        def model_dump(self, exclude_none=True):
            return {"role": "assistant", "content": self.content}

    class Call:
        id = "call-1"

        class Function:
            name = "run_python"
            arguments = '{"code":"df = df.drop_duplicates()"}'

        function = Function()

    responses = [
        type("Response", (), {"choices": [type("Choice", (), {"message": Message(tool_calls=[Call()])})]})(),
        type("Response", (), {"choices": [type("Choice", (), {"message": Message(content="Removed duplicates.")})]})(),
    ]
    agent = DataAgent("test-key", client=FakeClient(responses))
    agent._execute_tool = lambda name, arguments: (_ for _ in ()).throw(ValueError("bad generated code"))

    seen = []
    report, _, _ = agent.run(pd.DataFrame({"x": [1]}), "Clean the data", on_step=seen.append)

    assert report == "Removed duplicates."
    assert seen[0]["status"] == "error"