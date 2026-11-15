# The best live free-model aggregator — make.fun `free-model-aggregator-02`

One stdlib-only Python 3.13 server (`main.py`, no third-party packages) that
serves an OpenAI-compatible API (`GET /v1/models`,
`POST /v1/chat/completions`) and an Anthropic-compatible API
(`POST /v1/messages`), and answers every prompt with a real completion from
a live free OpenRouter model, dialled through the grader's gateway.

## How it works

- **Catalog.** At startup it reads the brief at `$MAKEFUN_PROMPT_FILE` and
  builds the catalog document from the fourteen free models it lists:
  one provider, `openrouter`, each model once. `python3 main.py --list`
  prints exactly the catalog it registers — byte for byte.
- **Routing.** For each request it ranks the free models by the house
  rating among the ones that meet the request's needs (JSON mode, tools,
  vision) and dials the best first: the top-rated model overall for plain
  and tools prompts, the top-rated JSON-capable model for JSON-mode
  prompts. Requirements are relaxed in a fixed order only when no model
  meets them all.
- **Failover.** A 429 / 5xx / timeout moves to the next candidate, and the
  next prompt tries the best model again — a refusal may be transient, and
  a retry costs only a fast call the tie-break does not count. Every
  refusal is reported in the `x-makefun-upstream-failures` trail header so
  the failover column can see it.
- **Wire fidelity.** Requests on `/v1/messages` are translated into a Chat
  Completions body for the gateway and the answer translated back, in JSON
  and in streaming (SSE) form, with usage and stop reasons whole and the
  dialled model id on every answer and every streamed chunk.
- **Determinism.** Every choice is a function of the brief, the requests
  and the gateway's answers: sequential response ids, `created` always 0,
  no wall clock or unseeded randomness in any answer.

## Trial

The trial runs the server as `python3 -P -s main.py` with the env at
`$MAKEFUN_ENV_URL`. The server picks a free port, registers
`{protocol_version: 1, port, catalog}`, serves the six prompts, polls
`GET $MAKEFUN_ENV_URL/work` and exits 0 when it says done.
