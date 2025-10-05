from typing import Any, Literal

from pydantic import BaseModel


class ToolCall(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict


class ToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Any


class ToolSpec(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict


class Content(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[Content | ToolCall | ToolResult]


class MessagesRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float | None = 0.7
    max_tokens: int | None = 1024
    stream: bool | None = False
    tools: list[ToolSpec] | None = None
    tool_choice: str | dict[str, str] | None = "auto"
