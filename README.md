# Chameleon Code - Use any AI model with Claude Code

Use any AI model with Claude Code.

## What is this?

Claude Chameleon is a proxy that translates Claude Code requests to any AI model. Uses OpenRouter by default.

## Quick Start

```bash
CAM_API_KEY="<YOUR_OPENROUTER_API_KEY>" ./cam.py
```

## Options

```
--model, -m MODEL              Model to use
--small-model SMALL_MODEL      Small model for haiku/subagent tasks (defaults to --model)
--max-tokens MAX_TOKENS        Maximum tokens per request
--url URL                      API URL to route to (defaults to OpenRouter)
--api-key API_KEY              API key (or set CAM_API_KEY env var)
--server                       Run server only, connect externally
--port PORT                    Port to bind to
```

## Examples

```bash
# Use with OpenRouter (default)
CAM_API_KEY="sk-or-v1-..." ./cam.py --model deepseek/deepseek-chat

# Use with different API
CAM_API_KEY="sk-..." ./cam.py --url https://api.openai.com/v1 --model gpt-4

# Run server on custom port
CAM_API_KEY="sk-..." ./cam.py --server --port 8080
```
