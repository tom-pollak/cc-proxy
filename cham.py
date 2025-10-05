#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydantic",
#   "fastapi",
#   "uvicorn",
#   "openai",
# ]
# ///
# fmt: off
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import threading
import uuid
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")


class ToolCall(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict

    def to_openai(self) -> dict:
        return {
            "id": self.id, "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.input or {})},
        }

class ToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Any

    def to_openai(self) -> dict:
        return {"role": "tool", "tool_call_id": self.tool_use_id, "content": str(self.content)}

class Content(BaseModel):
    type: Literal["text"]
    text: str


# ██████████████████████████████████  Request  ███████████████████████████████████


class ToolSpec(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description or "", "parameters": self.input_schema},
        }

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[Content | ToolCall | ToolResult]

    def to_openai(self) -> list[dict]:
        if isinstance(self.content, str):
            return [{"role": self.role, "content": self.content}]
        out, texts, calls = [], [], []
        for b in (self.content or []):
            if isinstance(b, Content):
                texts.append(b.text)
            elif isinstance(b, ToolCall):
                calls.append(b.to_openai())
            elif isinstance(b, ToolResult):
                out.append(b.to_openai())
        if texts or calls:
            out.append({
                "role": self.role, "content": "".join(texts),
                **({"tool_calls": calls} if calls else {}),
            })
        return out

class MessageRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = 0.7
    max_tokens: int | None = 1024
    stream: bool | None = False
    tools: list[ToolSpec] | None = None
    tool_choice: str | dict[str, str] | None = "auto"

    def to_openai(self, max_tokens: int | None) -> dict:
        msgs = [m for msg in self.messages for m in msg.to_openai()]
        out = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.tools:
            out |= {"tools": [t.to_openai() for t in self.tools], "tool_choice": "auto"}
        return out


# ██████████████████████████████████  Response  ██████████████████████████████████


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

class MessageResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"]
    type: Literal["message"]
    content: list[Content | ToolCall | ToolResult]
    stop_reason: Literal["tool_use", "end_turn"]
    stop_sequence: str | None = None
    usage: Usage

    @classmethod
    def from_completion(cls, completion) -> MessageResponse:
        choice = completion.choices[0]
        msg = choice.message

        blocks = []
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                blocks.append(
                    ToolCall(type="tool_use", id=call.id, name=call.function.name, input=json.loads(call.function.arguments or "{}"))
                )
            stop_reason = "tool_use"
        else:
            blocks.append(Content(type="text", text=msg.content or ""))
            stop_reason = "end_turn"

        return cls(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            model=completion.model,
            role="assistant",
            type="message",
            content=blocks,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
            ),
        )


# ███████████████████████████████████  Server  ███████████████████████████████████


def mk_app(max_tokens: int | None, url: str, api_key: str) -> FastAPI:
    app = FastAPI()
    client = OpenAI(base_url=url, api_key=api_key)

    @app.post("/v1/messages")
    async def messages(request: MessageRequest) -> MessageResponse:
        payload = request.to_openai(max_tokens)
        try:
            completion = client.chat.completions.create(**payload)
        except Exception as e:
            logger.error(f"Error calling {request.model}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        return MessageResponse.from_completion(completion)

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chameleon Code - Use any AI model with Claude Code",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30)
    )
    parser.add_argument("--model", "-m", default="deepseek/deepseek-chat-v3.1:free", help="Model to use")
    parser.add_argument("--small-model", help="Small model for haiku/subagent tasks (defaults to --model)")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens per request")
    parser.add_argument("--url", default="https://openrouter.ai/api/v1", help="API URL to route to (defaults to OpenRouter)")
    parser.add_argument("--api-key", default=os.getenv("CHAM_API_KEY"), help="API key (or set CHAM_API_KEY env var)")
    parser.add_argument("--server", action="store_true", help="Run server only, connect externally")
    parser.add_argument("--port", type=int, default=8654, help="Port to bind to")
    args = parser.parse_args()
    small_model = args.small_model or args.model

    if not args.api_key:
        parser.error("--api-key is required (or set CHAM_API_KEY environment variable)")

    app = mk_app(args.max_tokens, args.url, args.api_key)

    anthropic_env = {
        "ANTHROPIC_BASE_URL": f"http://localhost:{args.port}/",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": args.model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": args.model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": small_model,
        "CLAUDE_CODE_SUBAGENT_MODEL": small_model,
    }

    if args.server:
        env_str = " \\\n".join(f"{k}={v}" for k, v in anthropic_env.items())
        print(f"Connect with:\n{env_str} \\\nclaude\n")
        uvicorn.run(app, port=args.port)
    else:
        threading.Thread(target=lambda: uvicorn.run(app, port=args.port, log_level="critical"), daemon=True).start()
        subprocess.run(["claude"], env={**os.environ, **anthropic_env})
