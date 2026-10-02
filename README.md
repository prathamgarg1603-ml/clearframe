# Clearframe

Clearframe is a small data-cleaning agent prototype built around Nebius Token Factory and NVIDIA Nemotron. Upload a CSV, describe a cleaning task, inspect the agent's tool calls, and download the cleaned CSV and Markdown report.

## Start

1. Create a Nebius Token Factory account at [tokenfactory.nebius.com](https://tokenfactory.nebius.com/) and join the hackathon through its official registration page.
2. Create an API key in the Token Factory project settings. Keep it private.
3. In the Token Factory model catalog, copy the exact API model ID for an NVIDIA Nemotron model available to your project. The app's model field and `NEBIUS_MODEL` environment variable are editable because model availability is project-dependent; the seeded Nemotron-3 Super name may need replacing with the catalog's exact ID.
4. Create an environment and install dependencies:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   $env:NEBIUS_API_KEY = "your-key"
   $env:NEBIUS_MODEL = "paste-the-catalog-model-id"
   streamlit run app.py
   ```

5. Use **Test connection** to send a single prompt and check for a reply, then upload a CSV and run a cleaning task.

## Token Factory Integration

The app uses the OpenAI Python client with Nebius's documented base URL, `https://api.tokenfactory.nebius.com/v1/`, and the Chat Completions API. Nebius documents OpenAI-compatible function calling using `tools`, `tool_choice`, assistant `tool_calls`, and `role: tool` results. The three registered tools are `describe_data`, `run_python`, and `make_chart`; their outputs are fed back to the model until it returns a normal assistant response.

The exact model ID is not hard-coded as authoritative: Token Factory's `/v1/models` catalog requires project authentication. Copy the Nemotron ID from the model catalog and verify the selected model supports function calling in the Token Factory Playground before the demo. The application starts with a Nemotron-3-Super candidate ID, but you should replace it if the authenticated catalog displays a different exact string.

Documentation:

- [Token Factory quickstart](https://docs.tokenfactory.nebius.com/quickstart)
- [API introduction and authentication](https://docs.tokenfactory.nebius.com/api-reference/introduction)
- [List models](https://docs.tokenfactory.nebius.com/api-reference/examples/list-of-models)
- [Function calling and tools](https://docs.tokenfactory.nebius.com/ai-models-inference/function-calling)

## Reliability and Safety

Tool errors are sent back as tool results, allowing the model to inspect the failure and retry. The loop has a 10-round cap. Model-generated code runs in a separate Python subprocess with isolated mode, a restricted AST policy, a temporary working directory, and an 8-second timeout; it is not executed in the Streamlit process. This lightweight worker is a prototype guardrail, not a production security boundary. Before accepting untrusted uploads or deploying publicly, run code execution in a disposable container or remote sandbox with network disabled, resource limits, and filesystem isolation.

Only CSV upload and local demonstration are implemented in this MVP. No public deployment or dataset benchmark has been run yet, so there are no success-rate results to report. Add 3-5 public datasets and record completion, repair attempts, and output quality before making claims in the submission README.

## Checks

```powershell
pytest -q
```

## Hackathon Submission Checklist

- Join the hackathon and create the Token Factory account/API key (requires your account sign-in).
- Confirm the model ID from the authenticated catalog and make a real smoke-test request.
- Benchmark 3-5 messy public datasets and summarize results here.
- Deploy the app publicly and record a demo video of 3 minutes or less.
- Publish the repository and complete the project description and feedback form.