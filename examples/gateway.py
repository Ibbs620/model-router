# gateway.py
import asyncio, hashlib, json, os, threading, time, traceback, uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)          # before anything reads os.environ

from openjiuwen.x_router.service import build_service, parse_reasoning            # noqa: E402
from openjiuwen.x_router.complexity import conversation_preview   # noqa: E402

try:
    from openjiuwen.x_router.classifier.capture import CAPTURE        # your Laya backend writes its output here
except ImportError:
    CAPTURE = threading.local()
    print("WARNING: capture.py not found; tier/confidence/probs will be null", flush=True)

# ---------------------------------------------------------------- setup
app = FastAPI()
svc = build_service(str(HERE.parent / "config" / "x-router-example.toml"), workers=2)

OR = "https://openrouter.ai/api/v1"
KEY = os.environ["OPENROUTER_KEY"]
LOCAL_BASE = os.environ.get("LOCAL_BASE", "http://127.0.0.1:8001/v1")   # vLLM
LOCAL_KEY = os.environ.get("LOCAL_KEY", "EMPTY")      # vLLM ignores the key unless you set --api-key

UPSTREAMS = {
    "SIMPLE":    {"base": LOCAL_BASE, "key": LOCAL_KEY, "model": "local-simple",
                  "in": 0.0, "out": 0.0,                       # local GPU, no per-token cost
                  "temperature": 0.2},                         # no reasoning key: vLLM doesn't take OpenRouter's object
    "MEDIUM":    {"base": OR, "key": KEY, "model": "google/gemma-4-26b-a4b-it",
                  "in": 0.0, "out": 0.0,                       # TODO: fill in from openrouter.ai/models
                  "temperature": 0.2,
                  "reasoning": {"enabled": True}},             # reasoning on, model's default effort
    "COMPLEX":   {"base": OR, "key": KEY, "model": "deepseek/deepseek-v3.2",
                  "in": 0.2088e-6, "out": 0.3096e-6,
                  "temperature": 0.2,
                  "reasoning": {"enabled": True}},
    "RESEARCH":  {"base": OR, "key": KEY, "model": "openai/gpt-6-luna",
                  "in": 0.0, "out": 0.0,                       # TODO: fill in from openrouter.ai/models
                  "temperature" : 0.2,
                  "reasoning": {"effort": "medium"}},
    "REASONING": {"base": OR, "key": KEY, "model": "deepseek/deepseek-v4.1-flash",
                  "in": 0.0243e-6, "out": 0.60e-6,                      
                  "temperature" : 0.2,
                  "reasoning": {"effort": "max"}},
}
LOG_DIR = Path(os.environ.get("ROUTER_LOG_DIR", HERE / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
try:
    os.chmod(LOG_DIR, 0o700)           # prompts can contain sensitive data on a shared box
except OSError:
    pass
LOG_FULL_PROMPTS = os.environ.get("LOG_FULL_PROMPTS", "1") == "1"
FORCE_ENGLISH = os.environ.get("FORCE_ENGLISH") == "1"   # leave unset for benchmark runs

def _env_float(name):
    v = os.environ.get(name)
    return float(v) if v else None

# global overrides; None = not forced. Can be changed at runtime via POST /settings
SETTINGS = {
    "temperature": _env_float("FORCE_TEMPERATURE"),
    "reasoning_effort": os.environ.get("FORCE_REASONING_EFFORT") or None,
}

EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}

def apply_params(body, up):
    """Mutates the upstream request body. Precedence: forced global > client > tier default."""
    # temperature
    if SETTINGS["temperature"] is not None:
        body["temperature"] = SETTINGS["temperature"]
    elif "temperature" not in body and up.get("temperature") is not None:
        body["temperature"] = up["temperature"]

    # reasoning: OpenRouter-only; strip it for the local upstream
    if up["base"] != OR:
        body.pop("reasoning", None)
        body.pop("reasoning_effort", None)
        return
    if SETTINGS["reasoning_effort"]:
        body.pop("reasoning_effort", None)
        body["reasoning"] = {"effort": SETTINGS["reasoning_effort"]}
    elif "reasoning" not in body and "reasoning_effort" not in body and up.get("reasoning"):
        body["reasoning"] = dict(up["reasoning"])

_log_lock = threading.Lock()
SESSION_COUNT = {}                     # session id -> requests seen
CURRENT = {"label": ""}                # set via POST /mark/{label}


import re

SOURCE_RE = re.compile(r"\bsource=([^\s,;]+)")
# only the known-bad source is fatal; other sources (e.g. a bandit override) are logged, not blocked
BAD_SOURCES = {s for s in os.environ.get("BAD_ROUTE_SOURCES", "heuristic_fallback").split(",") if s}
HALT = {"reason": None, "since": None, "rejected": 0}


def route_source(sel):
    m = SOURCE_RE.search(str(getattr(sel, "reasoning", "") or ""))
    return m.group(1) if m else None


def halt(reason):
    if HALT["reason"] is not None:
        return
    HALT.update(reason=reason, since=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (LOG_DIR / "HALTED.txt").write_text(f"{HALT['since']}\n{reason}\n", encoding="utf-8")
    print("\n" + "!" * 70 + f"\nROUTER HALTED: {reason}\nAll requests now return 503 "
          "until POST /resume\n" + "!" * 70 + "\n", flush=True)


def halted_response():
    HALT["rejected"] += 1
    return JSONResponse({"error": {
        "type": "router_unhealthy",
        "message": f"Router halted since {HALT['since']}: {HALT['reason']}"}}, status_code=503)

# ---------------------------------------------------------------- helpers
def write_jsonl(name, rec):
    line = json.dumps(rec, ensure_ascii=False, default=str)
    with _log_lock, open(LOG_DIR / name, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def text_of(m):
    c = (m or {}).get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):            # multimodal content parts
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def conv_stats(msgs):
    roles = [m.get("role") for m in msgs]
    last_user = max((i for i, r in enumerate(roles) if r == "user"), default=-1)
    return {
        "n_messages": len(msgs),
        "user_turns": roles.count("user"),                  # user<->agent exchanges so far
        "assistant_msgs": roles.count("assistant"),
        "tool_msgs": roles.count("tool"),
        "agent_steps_this_turn": sum(1 for r in roles[last_user + 1:] if r == "assistant"),
        "last_role": roles[-1] if roles else None,          # "tool" = mid-loop continuation
    }


def route_and_capture(msgs, sid):
    """Runs in a worker thread; the Laya backend fills CAPTURE.data during route."""
    CAPTURE.data = None
    sel = svc.route(msgs, session_id=sid, agent_id="gateway")
    return sel, getattr(CAPTURE, "data", None)


def record(rec, up, ok, t0, msgs, text, tool_calls, usage, error=None):
    """Synchronous: write both log files. Returns the cost in USD."""
    usage = usage or {}
    pt, ct = usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0
    cost = pt * up["in"] + ct * up["out"]
    rec.update({
        "ok": ok,
        "error": error,
        "upstream_ms": round((time.time() - t0) * 1000),
        "input_tokens": pt,
        "output_tokens": ct,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "usage_missing": not usage,
        "cost_usd": round(cost, 6),
        "n_tool_calls_out": len(tool_calls or []),
    })
    write_jsonl("requests.jsonl", rec)
    if LOG_FULL_PROMPTS:
        write_jsonl("prompts.jsonl", {"request_id": rec["request_id"], "messages": msgs,
                                      "response_text": text, "tool_calls": tool_calls})
    return cost


async def report(sel, ok, t0, msgs, body, text, tool_calls, cost, sid):
    """Feed the outcome back to the router. Never lets a failure break the response."""
    try:
        await asyncio.to_thread(
            svc.report, sel,
            outcome="ok" if ok else "unavailable",
            latency_ms=(time.time() - t0) * 1000,
            messages=msgs, response_text=text,
            tool_calls=tool_calls or (), tools=body.get("tools") or (),
            cost_usd=cost, session_id=sid, agent_id="gateway",
        )
    except Exception:
        traceback.print_exc()


@app.on_event("shutdown")
def _close():
    svc.close(timeout=10)

# ---------------------------------------------------------------- small endpoints
@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "auto", "object": "model"}]}


@app.get("/stats")
def stats():
    return svc.stats


@app.post("/mark/{label}")
def mark(label: str):
    """Tag every following request with a label (e.g. the PinchBench task id)."""
    CURRENT["label"] = "" if label == "none" else label
    return {"label": CURRENT["label"]}

@app.get("/settings")
def get_settings():
    return SETTINGS


@app.post("/settings")
async def set_settings(req: Request):
    """e.g. {"temperature": 0.0, "reasoning_effort": "low"}; null clears an override."""
    data = await req.json()
    if "temperature" in data:
        t = data["temperature"]
        if t is not None and not (0 <= float(t) <= 2):
            return JSONResponse({"error": "temperature must be in [0, 2]"}, status_code=400)
        SETTINGS["temperature"] = None if t is None else float(t)
    if "reasoning_effort" in data:
        e = data["reasoning_effort"]
        if e is not None and e not in EFFORTS:
            return JSONResponse({"error": f"reasoning_effort must be one of {sorted(EFFORTS)}"},
                                status_code=400)
        SETTINGS["reasoning_effort"] = e
    return SETTINGS

# ---------------------------------------------------------------- main handler
@app.post("/v1/chat/completions")
async def chat(req: Request):
    if HALT["reason"]:
        return halted_response()          # no routing, no GPU, no upstream call
    rec = None
    try:
        body = await req.json()
        msgs = body["messages"]
        if FORCE_ENGLISH:
            msgs = msgs + [{"role": "system",
                            "content": "Respond in English unless the user writes in another language."}]
            body["messages"] = msgs

        label = CURRENT["label"]
        first_user = next((m for m in msgs if m.get("role") == "user"), msgs[0])
        sid = req.headers.get("x-session-id") or hashlib.md5(
            (label + json.dumps(first_user, sort_keys=True, default=str)).encode()).hexdigest()
        SESSION_COUNT[sid] = SESSION_COUNT.get(sid, 0) + 1

        last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), {})
        sys_msg = next((m for m in msgs if m.get("role") == "system"), {})
        rec = {
            "request_id": uuid.uuid4().hex[:12],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "label": label,
            "session": sid,
            "request_no_in_session": SESSION_COUNT[sid],
            **conv_stats(msgs),
            "last_user_prompt": text_of(last_user)[:2000],
            "last_user_prompt_chars": len(text_of(last_user)),
            "system_prompt_hash": hashlib.md5(text_of(sys_msg).encode()).hexdigest()[:10],
            "classifier_view": conversation_preview(msgs, svc.params.classifier_preview_chars),
            "stream": bool(body.get("stream")),
            "has_tools": bool(body.get("tools")),
        }

        # ---- route
        t_route = time.perf_counter()
        sel, cap = await asyncio.to_thread(route_and_capture, msgs, sid)
        route_ms = (time.perf_counter() - t_route) * 1000
        cap = cap or {}

        source = route_source(sel)
        rec["route_source"] = source
        if source in BAD_SOURCES:
            halt(f"route source={source} (classifier failed; reasoning: {getattr(sel, 'reasoning', None)})")
            rec.update({"ok": False, "error": f"router_fallback:{source}",
                        "tier": cap.get("tier"), "router_model": sel.selected_model_id,
                        "reasoning": getattr(sel, "reasoning", None),
                        "route_ms": round(route_ms, 1)})
            write_jsonl("requests.jsonl", rec)
            return halted_response()

        up = UPSTREAMS[sel.selected_model_id]
        rec.update({
            "tier": cap.get("tier"),
            "confidence": cap.get("confidence"),
            "probs": cap.get("probs"),
            "router_model": sel.selected_model_id,
            "upstream_model": up["model"],
            "reasoning": getattr(sel, "reasoning", None),
            "route_ms": round(route_ms, 1),
        })
        print("ROUTED ->", sel.selected_model_id, "| tier", cap.get("tier"),
              "| conf", cap.get("confidence"), "|", label, flush=True)

        body["model"] = up["model"]
        apply_params(body, up)
        rec["req_temperature"] = body.get("temperature")
        rec["req_reasoning"] = body.get("reasoning") or body.get("reasoning_effort")
        url = f"{up['base']}/chat/completions"
        headers = {"Authorization": f"Bearer {up['key']}"}
        t0 = time.time()
        client = httpx.AsyncClient(timeout=300)

        # ---- streaming
        if body.get("stream"):
            body["stream_options"] = {"include_usage": True}
            r = await client.send(
                client.build_request("POST", url, json=body, headers=headers), stream=True)

            if r.status_code >= 400:                       # surface upstream errors as-is
                err = await r.aread()
                await r.aclose(); await client.aclose()
                print("UPSTREAM ERROR", r.status_code, err[:500], flush=True)
                cost = record(rec, up, False, t0, msgs, None, None, {},
                              error=f"HTTP {r.status_code}: {err[:300].decode('utf-8', 'replace')}")
                await report(sel, False, t0, msgs, body, None, None, cost, sid)
                return Response(err, status_code=r.status_code, media_type="application/json")

            async def gen():
                text, tc, usage = [], {}, {}
                ok, err, completed = True, None, False
                try:
                    async for line in r.aiter_lines():
                        yield line + "\n"
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                ev = json.loads(line[6:])
                            except Exception:
                                continue
                            usage = ev.get("usage") or usage
                            for ch in ev.get("choices", []):
                                d = ch.get("delta") or {}
                                text.append(d.get("content") or "")
                                for t in d.get("tool_calls") or []:
                                    slot = tc.setdefault(t.get("index", 0), {
                                        "id": None, "type": "function",
                                        "function": {"name": "", "arguments": ""}})
                                    if t.get("id"):
                                        slot["id"] = t["id"]
                                    fn = t.get("function") or {}
                                    slot["function"]["name"] += fn.get("name") or ""
                                    slot["function"]["arguments"] += fn.get("arguments") or ""
                    completed = True
                except Exception as e:
                    ok, err = False, f"{type(e).__name__}: {e}"
                    traceback.print_exc()
                finally:
                    await r.aclose(); await client.aclose()
                    if not completed and ok:               # client hung up mid-stream
                        ok, err = False, "client_disconnected"
                    tool_calls = [tc[i] for i in sorted(tc)]
                    cost = record(rec, up, ok, t0, msgs, "".join(text), tool_calls, usage, error=err)
                await report(sel, ok, t0, msgs, body, "".join(text), tool_calls, cost, sid)

            return StreamingResponse(gen(), media_type="text/event-stream")

        # ---- non-streaming
        try:
            r = await client.post(url, json=body, headers=headers)
        finally:
            await client.aclose()
        try:
            data = r.json()
        except Exception:
            print("UPSTREAM NON-JSON", r.status_code, r.text[:500], flush=True)
            cost = record(rec, up, False, t0, msgs, None, None, {},
                          error=f"non-JSON HTTP {r.status_code}: {r.text[:300]}")
            await report(sel, False, t0, msgs, body, None, None, cost, sid)
            return Response(r.content, status_code=r.status_code)

        ok = r.status_code < 400
        choice = (data.get("choices") or [{}])[0].get("message", {})
        err = None if ok else json.dumps(data.get("error", data))[:300]
        cost = record(rec, up, ok, t0, msgs, choice.get("content"), choice.get("tool_calls"),
                      data.get("usage") or {}, error=err)
        await report(sel, ok, t0, msgs, body, choice.get("content"), choice.get("tool_calls"), cost, sid)
        return JSONResponse(data, status_code=r.status_code)

    except Exception as e:
        traceback.print_exc()
        if rec is not None:                                 # failed requests still get a log line
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
            write_jsonl("requests.jsonl", rec)
        return JSONResponse({"error": {"message": f"{type(e).__name__}: {e}"}}, status_code=500)

@app.get("/health")
def health():
    return {"halted": HALT, "router": svc.stats}


@app.post("/resume")
def resume():
    """Clear the halt after you've fixed the cause (and checked that Laya is healthy)."""
    HALT.update(reason=None, since=None, rejected=0)
    try:
        (LOG_DIR / "HALTED.txt").unlink()
    except FileNotFoundError:
        pass
    return {"resumed": True}


@app.on_event("startup")
def _probe():
    """Refuse to start serving if the classifier is already broken. Also warms it up."""
    try:
        sel, _ = route_and_capture([{"role": "user", "content": "Hello"}], "startup-probe")
        src = route_source(sel)
        if src in BAD_SOURCES:
            halt(f"startup probe got source={src}")
    except Exception as e:
        halt(f"startup probe raised {type(e).__name__}: {e}")