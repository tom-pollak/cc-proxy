# Chameleon Code - Use any AI model with Claude Code

Use any AI model with Claude Code.

## What is this?

Chameleon Code is a proxy that translates Claude Code requests to any AI model. Uses OpenRouter by default.

## Quick Start

```bash
uv tool install chameleon-code
CHAM_API_KEY="<YOUR_OPENROUTER_API_KEY>" cham
```

Or run standalone (it's single Python file, requires `uv`):

```bash
./cham.py
```

## Options

```
--model, -m MODEL              Model to use
--small-model SMALL_MODEL      Small model for haiku/subagent tasks (defaults to --model)
--max-tokens MAX_TOKENS        Maximum tokens per request
--url URL                      API URL to route to (defaults to OpenRouter)
--api-key API_KEY              API key (or set CHAM_API_KEY env var)
--server                       Run server only, connect externally
--port PORT                    Port to bind to
```

## Examples

```bash
# Use with OpenRouter (default)
CHAM_API_KEY="sk-or-v1-..." cham --model deepseek/deepseek-chat

# Use with different API
CHAM_API_KEY="sk-..." cham --url https://api.openai.com/v1 --model gpt-4

# Run server on custom port
CHAM_API_KEY="sk-..." cham --server --port 8080
```
