# gateway.py
import asyncio, hashlib, json, os, threading, time, traceback, uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)          # before anything reads os.environ

from openjiuwen.x_router.service import build_service            # noqa: E402
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

# keys must match the model ids in the profile's [targets] / tier_models exactly
UPSTREAMS = {
    "SIMPLE": {"base": OR, "key": KEY, "model": "z-ai/glm-5.3-flash",            "in": 0.02e-6,   "out": 0.3e-6},
    "MEDIUM": {"base": OR, "key": KEY, "model": "deepseek/deepseek-v3.2",        "in": 0.2088e-6, "out": 0.3096e-6},
    "COMPLEX": {"base": OR, "key": KEY, "model": "deepseek/deepseek-v4.1-flash",  "in": 0.0243e-6,  "out": 0.60e-6},
    "RESEARCH": {"base": OR, "key": KEY, "model": "deepseek/deepseek-v4.1-flash",  "in": 0.0243e-6,  "out": 0.60e-6},
    "REASONING": {"base": OR, "key": KEY, "model": "anthropic/claude-sonnet-4.6",   "in": 3e-6, "out": 15e-6},
}
LOG_DIR = Path(os.environ.get("ROUTER_LOG_DIR", HERE / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
try:
    os.chmod(LOG_DIR, 0o700)           # prompts can contain sensitive data on a shared box
except OSError:
    pass
LOG_FULL_PROMPTS = os.environ.get("LOG_FULL_PROMPTS", "1") == "1"
FORCE_ENGLISH = os.environ.get("FORCE_ENGLISH") == "1"   # leave unset for benchmark runs

_log_lock = threading.Lock()
SESSION_COUNT = {}                     # session id -> requests seen
CURRENT = {"label": ""}                # set via POST /mark/{label}


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


# ---------------------------------------------------------------- main handler
@app.post("/v1/chat/completions")
async def chat(req: Request):
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