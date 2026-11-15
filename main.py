# free-model-aggregator-02: one server, two wires, live free models.
#
# A single-file Python 3.13 standard-library server. It listens on a port it
# picks, registers itself and its catalog with the trial env, and answers
# OpenAI Chat Completions and Anthropic Messages requests by dialling a real
# free OpenRouter model through the grader's gateway. Every completion comes
# from the gateway; this server never writes an answer of its own.
#
# Routing, in one sentence: for each request, rank the free models the brief
# lists by their house rating among the ones that meet the request's needs
# (JSON mode, tools, vision), dial the best first, and on a refusal fail over
# to the next one, reporting every refusal in the x-makefun-upstream-failures
# trail so the failover column can see it.
#
# Determinism: every choice is a function of the brief, the request and the
# gateway's answers. Response ids are sequential counters, created is always
# 0, and there is no wall clock or unseeded randomness anywhere in an answer.
# The only clock use is a per-request deadline that bounds how long a prompt
# may spend on failover; it never leaks into a response.
#
# Usage:
#   python3 main.py --list   print the catalog document and exit 0
#   python3 main.py          run the server for one trial (env vars set by
#                            the trial: MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE,
#                            MAKEFUN_TRIAL_SEED, OPENAI_BASE_URL,
#                            ANTHROPIC_BASE_URL, OPENAI_API_KEY,
#                            ANTHROPIC_API_KEY)

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

SCHEMA_PREFIX = "free-model-aggregator-02/"
PROTOCOL_VERSION = 1
OPENROUTER_PREFIX = "openrouter/"
CATALOG_PROVIDER_ID = "openrouter"
CATALOG_BASE_URL = "https://openrouter.ai/api/v1"
CATALOG_KEY_ENV = "OPENROUTER_API_KEY"
FAILURES_HEADER = "x-makefun-upstream-failures"
MAX_TRAIL_ROWS = 8
MAX_ATTEMPTS_PER_PROMPT = 6
CALL_TIMEOUT_SEC = 6.0
PROMPT_DEADLINE_SEC = 25.0
POLL_INTERVAL_SEC = 0.1

Json = dict | list | str | int | float | bool | None


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ""


def number_or_none(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def bool_of(value: object) -> bool:
    return value is True


def find_documents(text: str) -> list[dict]:
    """Every JSON object in `text` that carries a string `schema` member.

    The brief is a text file with two JSON documents embedded in it, so a
    plain json.loads of the whole file would fail. A decoder is walked over
    each '{' in the text; objects whose schema starts with this contest's
    prefix are collected in the order they appear.
    """
    decoder = json.JSONDecoder()
    found: list[dict] = []
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        pos = end
        if is_record(value):
            schema = value.get("schema")
            if isinstance(schema, str) and schema.startswith(SCHEMA_PREFIX):
                found.append(value)
    return found


def read_brief(path: str) -> dict:
    """The brief at `path`: its free-model list and its prompt manifest."""
    brief: dict = {"models": [], "prompts": []}
    if not path:
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return brief
    for doc in find_documents(text):
        if isinstance(doc.get("models"), list):
            brief["models"] = [m for m in doc["models"] if is_record(m)]
        if isinstance(doc.get("prompts"), list):
            brief["prompts"] = [p for p in doc["prompts"] if is_record(p)]
    return brief


def model_id_of(model: dict) -> str:
    return string_of(model.get("id"))


def rating_of(model: dict) -> int:
    value = number_or_none(model.get("rating_milli"))
    return int(value) if value is not None else 0


def context_of(model: dict) -> int | None:
    value = number_or_none(model.get("context_length"))
    return int(value) if value is not None else None


def capabilities_of(model: dict) -> dict:
    return {
        "tools": bool_of(model.get("tools")),
        "vision": bool_of(model.get("vision")),
        "json": bool_of(model.get("json")),
    }


def meets_requirements(model: dict, requires: dict) -> bool:
    caps = capabilities_of(model)
    for key in ("tools", "vision", "json"):
        if bool_of(requires.get(key)) and not caps[key]:
            return False
    return True


def candidates_for(models: list[dict], requires: dict) -> list[dict]:
    """Free models that meet `requires`, best rating first, id as tie-break."""
    able = [m for m in models if bool_of(m.get("free")) and meets_requirements(m, requires)]
    able.sort(key=lambda m: (-rating_of(m), model_id_of(m)))
    return able


def build_catalog(models: list[dict]) -> dict:
    """The catalog document: one provider, exactly the brief's free models."""
    rows = []
    for model in models:
        if not bool_of(model.get("free")):
            continue
        row: dict = {"id": model_id_of(model), "free": True}
        ctx = context_of(model)
        row["context_length"] = ctx if ctx is not None else None
        rows.append(row)
    return {
        "providers": [
            {
                "id": CATALOG_PROVIDER_ID,
                "base_url": CATALOG_BASE_URL,
                "key_env": CATALOG_KEY_ENV,
                "wire": "openai",
                "models": rows,
            }
        ]
    }


def catalog_bytes(models: list[dict]) -> bytes:
    """The catalog as one deterministic JSON document (the --list output)."""
    return (json.dumps(build_catalog(models), separators=(",", ":")) + "\n").encode("utf-8")


def openai_model_list(models: list[dict]) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id_of(m),
                "object": "model",
                "created": 0,
                "owned_by": CATALOG_PROVIDER_ID,
            }
            for m in models
            if bool_of(m.get("free"))
        ],
    }


def anthropic_model_list(models: list[dict]) -> dict:
    data = []
    for model in models:
        if not bool_of(model.get("free")):
            continue
        data.append(
            {
                "id": model_id_of(model),
                "type": "model",
                "display_name": string_of(model.get("display_name")) or model_id_of(model),
                "created_at": "2026-11-15T00:00:00Z",
            }
        )
    first = data[0]["id"] if data else ""
    last = data[-1]["id"] if data else ""
    return {"data": data, "has_more": False, "first_id": first, "last_id": last}


def text_of_content(content: object) -> str:
    """Plain text carried by a message's content, string or blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def anthropic_text_of(content: object) -> str:
    return text_of_content(content)


def openai_parts_of(content: object) -> list[dict]:
    """An Anthropic message's content blocks as OpenAI chat parts."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": ""}]
    parts: list[dict] = []
    for block in content:
        if not is_record(block):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif kind == "image" and is_record(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                media = string_of(source.get("media_type")) or "image/png"
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{source['data']}"},
                    }
                )
        elif kind == "tool_result":
            # Tool results travel as their own OpenAI tool message; a text
            # fallback keeps the content visible when no call id is known.
            parts.append({"type": "text", "text": anthropic_text_of(block.get("content"))})
    return parts if parts else [{"type": "text", "text": ""}]


def anthropic_tools_to_openai(tools: object) -> list[dict]:
    out: list[dict] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not is_record(tool):
            continue
        name = string_of(tool.get("name"))
        if not name:
            continue
        schema = tool.get("input_schema")
        parameters = schema if is_record(schema) else {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": string_of(tool.get("description")),
                    "parameters": parameters,
                },
            }
        )
    return out


def anthropic_to_openai(body: dict, model: str) -> dict:
    """Translate an Anthropic Messages body into a Chat Completions body."""
    messages: list[dict] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = anthropic_text_of(system)
        if text:
            messages.append({"role": "system", "content": text})
    raw_messages = body.get("messages")
    if isinstance(raw_messages, list):
        for message in raw_messages:
            if not is_record(message):
                continue
            role = string_of(message.get("role")) or "user"
            parts = openai_parts_of(message.get("content"))
            messages.append({"role": role, "content": parts})
    out: dict = {"model": model, "messages": messages}
    max_tokens = number_or_none(body.get("max_tokens"))
    if max_tokens is not None:
        out["max_tokens"] = int(max_tokens)
    temperature = number_or_none(body.get("temperature"))
    if temperature is not None:
        out["temperature"] = temperature
    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        out["tools"] = tools
    if bool_of(body.get("stream")):
        out["stream"] = True
    stop = body.get("stop_sequences")
    if isinstance(stop, list) and stop:
        out["stop"] = [string_of(s) for s in stop if isinstance(s, str)]
    return out


def requirements_of_openai(body: dict) -> dict:
    """What a model must support to serve this OpenAI-shaped request."""
    response_format = body.get("response_format")
    json_mode = is_record(response_format) and string_of(response_format.get("type")) in (
        "json_object",
        "json_schema",
    )
    tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    vision = False
    raw_messages = body.get("messages")
    if isinstance(raw_messages, list):
        for message in raw_messages:
            if not is_record(message):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if is_record(part) and part.get("type") == "image_url":
                        vision = True
    return {"tools": tools, "vision": vision, "json": json_mode}


def requirements_of_anthropic(body: dict) -> dict:
    tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    vision = False
    raw_messages = body.get("messages")
    if isinstance(raw_messages, list):
        for message in raw_messages:
            if not is_record(message):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if is_record(block) and block.get("type") == "image":
                        vision = True
    return {"tools": tools, "vision": vision, "json": False}


def openai_stop_to_anthropic(finish: str) -> str:
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "refusal",
    }.get(finish, "end_turn")


def openai_message_to_anthropic(message: dict, model: str, reply_id: str, usage: dict, finish: str) -> dict:
    """An OpenAI choice message as an Anthropic message object."""
    content: list[dict] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    elif isinstance(text, list):
        joined = text_of_content(text)
        if joined:
            content.append({"type": "text", "text": joined})
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not is_record(call) or not is_record(call.get("function")):
                continue
            function = call["function"]
            name = string_of(function.get("name"))
            if not name:
                continue
            raw_args = function.get("arguments")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else {}
            except json.JSONDecodeError:
                parsed_args = {}
            if not is_record(parsed_args):
                parsed_args = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": string_of(call.get("id")) or f"toolu-{reply_id}",
                    "name": name,
                    "input": parsed_args,
                }
            )
    if not content:
        content.append({"type": "text", "text": ""})
    return {
        "type": "message",
        "id": reply_id,
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": openai_stop_to_anthropic(finish),
        "usage": usage,
    }


def usage_from_openai(usage: object, text: str) -> dict:
    """Whole usage counts: the provider's when present, else a counted one."""
    prompt_tokens = 0
    completion_tokens = 0
    if is_record(usage):
        prompt_value = number_or_none(usage.get("prompt_tokens"))
        completion_value = number_or_none(usage.get("completion_tokens"))
        if prompt_value is not None:
            prompt_tokens = int(prompt_value)
        if completion_value is not None:
            completion_tokens = int(completion_value)
    if completion_tokens < 1:
        completion_tokens = max(1, (len(text) + 3) // 4)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def sse_frame(data: str, event: str | None = None) -> str:
    if event is not None:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"


def openai_stream_body(reply_id: str, model: str, text: str, finish: str, usage: dict) -> str:
    """A complete OpenAI event stream for one finished completion."""
    base = {"id": reply_id, "object": "chat.completion.chunk", "created": 0, "model": model}
    frames: list[str] = []
    first = dict(base)
    first["choices"] = [
        {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
    ]
    frames.append(sse_frame(json.dumps(first, separators=(",", ":"))))
    # One delta carrying the whole text: splitting it could cut a token in
    # half across two frames, and a reader that does not join deltas would
    # then miss it.
    if text:
        chunk = dict(base)
        chunk["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": None}]
        frames.append(sse_frame(json.dumps(chunk, separators=(",", ":"))))
    final = dict(base)
    final["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish}]
    final["usage"] = {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
    }
    frames.append(sse_frame(json.dumps(final, separators=(",", ":"))))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_stream_body(reply_id: str, model: str, text: str, finish: str, usage: dict) -> str:
    """A complete Anthropic event stream for one finished message."""
    frames: list[str] = []
    start = {
        "type": "message_start",
        "message": {
            "id": reply_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": 0},
        },
    }
    frames.append(sse_frame(json.dumps(start, separators=(",", ":")), "message_start"))
    block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    frames.append(sse_frame(json.dumps(block_start, separators=(",", ":")), "content_block_start"))
    # One delta carrying the whole text: splitting it could cut a token in
    # half across two frames, and a reader that does not join deltas would
    # then miss it.
    if text:
        delta = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }
        frames.append(sse_frame(json.dumps(delta, separators=(",", ":")), "content_block_delta"))
    block_stop = {"type": "content_block_stop", "index": 0}
    frames.append(sse_frame(json.dumps(block_stop, separators=(",", ":")), "content_block_stop"))
    stop_reason = openai_stop_to_anthropic(finish)
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": usage["completion_tokens"]},
    }
    frames.append(sse_frame(json.dumps(message_delta, separators=(",", ":")), "message_delta"))
    frames.append(sse_frame(json.dumps({"type": "message_stop"}, separators=(",", ":")), "message_stop"))
    return "".join(frames)


class UpstreamRefusal(Exception):
    """A gateway call the provider refused, with its status and code."""

    def __init__(self, status: int, code: str):
        super().__init__(f"upstream refused: {status} {code}")
        self.status = status
        self.code = code


def error_code_of(body: bytes) -> str:
    try:
        value = json.loads(body.decode("utf-8", "replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    if is_record(value) and is_record(value.get("error")):
        error = value["error"]
        code = string_of(error.get("code")) or string_of(error.get("type"))
        return code[:64]
    return ""


def gateway_url() -> str:
    base = os.environ.get("OPENAI_BASE_URL", "")
    return base.rstrip("/") + "/chat/completions"


def dial(model: str, body: dict) -> tuple[dict, dict]:
    """One Chat Completions call through the gateway; raises on refusal.

    Returns (answer, usage) where answer is the provider's parsed JSON
    (non-streaming) or a dict with text/finish/usage assembled from its
    streamed chunks. A 4xx/5xx becomes an UpstreamRefusal; a timeout or a
    network error becomes one with status 0 and code "timeout".
    """
    payload = dict(body)
    payload["model"] = model
    payload["stream"] = False
    if number_or_none(payload.get("max_tokens")) is None:
        payload["max_tokens"] = 512
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        gateway_url(),
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", ""),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=CALL_TIMEOUT_SEC) as response:
            raw = response.read()
    except urllib.error.HTTPError as err:
        raise UpstreamRefusal(err.code, error_code_of(err.read()) or "http_error") from err
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise UpstreamRefusal(0, "timeout") from err
    try:
        answer = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as err:
        raise UpstreamRefusal(0, "bad_json") from err
    if not is_record(answer):
        raise UpstreamRefusal(0, "bad_json")
    choices = answer.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamRefusal(0, "no_choices")
    choice = choices[0] if is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    text = text_of_content(message.get("content"))
    finish = string_of(choice.get("finish_reason")) or "stop"
    usage = usage_from_openai(answer.get("usage"), text)
    return (
        {"message": message, "text": text, "finish": finish, "tool_calls": message.get("tool_calls")},
        usage,
    )


class Server:
    """One trial's state: the brief, the refused models, the id counter."""

    def __init__(self, brief: dict):
        self.models = [m for m in brief.get("models", []) if is_record(m)]
        self.counter = 0
        self.lock = threading.Lock()

    def next_id(self, prefix: str) -> str:
        with self.lock:
            self.counter += 1
            return f"{prefix}-{self.counter}"

    def serve(self, wire: str, body: dict) -> tuple[int, dict, bytes]:
        """Answer one prompt; returns (status, headers, body bytes)."""
        if wire == "anthropic":
            requires = requirements_of_anthropic(body)
        else:
            requires = requirements_of_openai(body)
        stream = bool_of(body.get("stream"))
        candidates = candidates_for(self.models, requires)
        if not candidates:
            # No free model meets the request: relax in a fixed order —
            # streaming never matters, then JSON, then tools, then vision —
            # so the prompt still gets an answer from the best available.
            for drop in ("vision", "json", "tools"):
                relaxed = dict(requires)
                relaxed[drop] = False
                candidates = candidates_for(self.models, relaxed)
                if candidates:
                    requires = relaxed
                    break
        failures: list[dict] = []
        deadline = time.monotonic() + PROMPT_DEADLINE_SEC
        tried: list[str] = []
        answer = None
        usage = {"prompt_tokens": 0, "completion_tokens": 1}
        served_model = ""
        for candidate in candidates:
            model = model_id_of(candidate)
            if model in tried:
                continue
            if len(tried) >= MAX_ATTEMPTS_PER_PROMPT:
                break
            if time.monotonic() > deadline:
                break
            tried.append(model)
            if wire == "anthropic":
                upstream_body = anthropic_to_openai(body, model)
            else:
                upstream_body = dict(body)
                upstream_body["model"] = model
                upstream_body["stream"] = False
                if number_or_none(upstream_body.get("max_tokens")) is None:
                    upstream_body["max_tokens"] = 512
            try:
                answer, usage = dial(model, upstream_body)
            except UpstreamRefusal as refusal:
                failures.append(
                    {"status": refusal.status, "code": refusal.code, "model": model}
                )
                continue
            served_model = model
            break
        headers: dict = {}
        if failures:
            trail = failures[:MAX_TRAIL_ROWS]
            headers[FAILURES_HEADER] = json.dumps(trail, separators=(",", ":"))
        if answer is None or not served_model:
            return 502, headers, self.error_body(wire, "all upstream models refused the prompt")
        reply_id = self.next_id("chatcmpl-agg" if wire == "openai" else "msg-agg")
        if wire == "openai":
            if stream:
                payload = openai_stream_body(
                    reply_id, served_model, answer["text"], answer["finish"], usage
                )
                headers["Content-Type"] = "text/event-stream"
                return 200, headers, payload.encode("utf-8")
            message = dict(answer["message"])
            message["role"] = "assistant"
            out = {
                "id": reply_id,
                "object": "chat.completion",
                "created": 0,
                "model": served_model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": answer["finish"],
                    }
                ],
                "usage": {
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
                },
            }
            return 200, headers, json.dumps(out).encode("utf-8")
        anthropic_usage = {
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
        }
        if stream:
            payload = anthropic_stream_body(
                reply_id, served_model, answer["text"], answer["finish"], usage
            )
            headers["Content-Type"] = "text/event-stream"
            return 200, headers, payload.encode("utf-8")
        out = openai_message_to_anthropic(
            answer["message"], served_model, reply_id, anthropic_usage, answer["finish"]
        )
        return 200, headers, json.dumps(out).encode("utf-8")

    @staticmethod
    def error_body(wire: str, message: str) -> bytes:
        if wire == "anthropic":
            return json.dumps(
                {"type": "error", "error": {"type": "overloaded_error", "message": message}}
            ).encode("utf-8")
        return json.dumps(
            {"error": {"message": message, "type": "server_error", "code": "upstream_refused"}}
        ).encode("utf-8")


SERVER: Server | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: int, headers: dict, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in headers.items():
            if key.lower() != "content-type":
                self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length else 0
        except ValueError:
            size = 0
        return self.rfile.read(size) if size > 0 else b""

    def do_GET(self) -> None:  # noqa: N802 - the base class's name
        path = urlsplit(self.path).path
        if path != "/v1/models" or SERVER is None:
            self._send(404, {}, b'{"error":{"message":"not found","type":"not_found"}}')
            return
        auth = self.headers.get("Authorization", "")
        api_key = self.headers.get("x-api-key", "")
        if api_key:
            body = anthropic_model_list(SERVER.models)
        elif auth.startswith("Bearer "):
            body = openai_model_list(SERVER.models)
        else:
            body = openai_model_list(SERVER.models)
        self._send(200, {}, json.dumps(body).encode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802 - the base class's name
        path = urlsplit(self.path).path
        if path not in ("/v1/chat/completions", "/v1/messages") or SERVER is None:
            self._send(404, {}, b'{"error":{"message":"not found","type":"not_found"}}')
            return
        wire = "anthropic" if path == "/v1/messages" else "openai"
        raw = self._read_body()
        try:
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send(400, {}, SERVER.error_body(wire, "request body is not JSON"))
            return
        if not is_record(body):
            self._send(400, {}, SERVER.error_body(wire, "request body is not an object"))
            return
        status, headers, payload = SERVER.serve(wire, body)
        self._send(status, headers, payload)


def http_post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {}


def http_get_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {}


def register(port: int, models: list[dict]) -> None:
    """Register with the env, retrying until it answers ok.

    The env can still be starting when this server comes up; a single
    fire-and-forget register that misses it would leave the trial
    un-driven forever (the env only starts on the first register, and
    only the first register counts). So retry for up to 30 seconds.
    """
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env_url:
        return
    payload = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": build_catalog(models)}
    deadline = time.monotonic() + 30.0
    while True:
        answer = http_post(f"{env_url}/register", payload)
        if bool_of(answer.get("ok")):
            return
        if time.monotonic() > deadline:
            return
        time.sleep(POLL_INTERVAL_SEC)


def poll_work() -> None:
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env_url:
        return
    while True:
        state = http_get_json(f"{env_url}/work")
        if bool_of(state.get("done")):
            return
        time.sleep(POLL_INTERVAL_SEC)


def pick_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def start() -> int:
    global SERVER
    brief = read_brief(os.environ.get("MAKEFUN_PROMPT_FILE", ""))
    SERVER = Server(brief)
    port = pick_port()
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    register(port, SERVER.models)
    poll_work()
    server.shutdown()
    server.server_close()
    return 0


def main() -> int:
    if "--list" in sys.argv[1:]:
        brief = read_brief(os.environ.get("MAKEFUN_PROMPT_FILE", ""))
        sys.stdout.buffer.write(catalog_bytes(brief["models"]))
        return 0
    return start()


if __name__ == "__main__":
    raise SystemExit(main())
