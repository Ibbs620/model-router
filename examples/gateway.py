# gateway.py
import asyncio, time, hashlib, json, os, traceback
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)          # before anything reads os.environ

from openjiuwen.x_router.service import build_service  # noqa: E402

app = FastAPI()
svc = build_service(str(HERE.parent / "config" / "x-router-example.toml"), workers=2)

OR = "https://openrouter.ai/api/v1"
KEY = os.environ["OPENROUTER_KEY"]

# keys must match the model ids in the profile's [targets] / tier_models exactly
UPSTREAMS = {
    "local":              {"base": OR, "key": KEY, "model": "qwen/qwen3.5-9b",            "in": 0.1e-6,   "out": 0.15e-6},
    "vendor-a/fast":      {"base": OR, "key": KEY, "model": "deepseek/deepseek-v3.2",     "in": 0.269e-6, "out": 0.4e-6},
    "vendor-b/deep":      {"base": OR, "key": KEY, "model": "deepseek/deepseek-v4-flash", "in": 0.04e-6,  "out": 0.64e-6},
    "vendor-c/reasoning": {"base": OR, "key": KEY, "model": "deepseek/deepseek-v4-pro",   "in": 0.462e-6, "out": 1.386e-6},
}


@app.on_event("shutdown")
def _close():
    svc.close(timeout=10)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "auto", "object": "model"}]}


@app.get("/stats")
def stats():
    return svc.stats


async def finish(sel, up, ok, t0, msgs, body, text, tool_calls, usage, sid):
    """Report the outcome. Never lets a reporting problem break the response."""
    try:
        cost = (usage.get("prompt_tokens", 0) * up["in"]
                + usage.get("completion_tokens", 0) * up["out"])
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


@app.post("/v1/chat/completions")
async def chat(req: Request):
    try:
        body = await req.json()
        msgs = body["messages"]
        sid = req.headers.get("x-session-id") or hashlib.md5(
            json.dumps(msgs[:1]).encode()).hexdigest()

        sel = await asyncio.to_thread(svc.route, msgs, session_id=sid, agent_id="gateway")
        print("ROUTED ->", sel.selected_model_id, "|", sel.reasoning, flush=True)
        up = UPSTREAMS[sel.selected_model_id]
        body["model"] = up["model"]

        url = f"{up['base']}/chat/completions"
        headers = {"Authorization": f"Bearer {up['key']}"}
        t0 = time.time()
        client = httpx.AsyncClient(timeout=300)

        # ---- streaming ----
        if body.get("stream"):
            body["stream_options"] = {"include_usage": True}
            r = await client.send(
                client.build_request("POST", url, json=body, headers=headers), stream=True)

            if r.status_code >= 400:                      # surface upstream errors as-is
                err = await r.aread()
                await r.aclose(); await client.aclose()
                print("UPSTREAM ERROR", r.status_code, err[:500], flush=True)
                await finish(sel, up, False, t0, msgs, body, None, None, {}, sid)
                return Response(err, status_code=r.status_code, media_type="application/json")

            async def gen():
                text, usage = [], {}
                try:
                    async for line in r.aiter_lines():
                        yield line + "\n"
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                ev = json.loads(line[6:])
                                usage = ev.get("usage") or usage
                                for ch in ev.get("choices", []):
                                    text.append((ch.get("delta") or {}).get("content") or "")
                            except Exception:
                                pass
                finally:
                    await r.aclose(); await client.aclose()
                await finish(sel, up, True, t0, msgs, body, "".join(text), None, usage, sid)

            return StreamingResponse(gen(), media_type="text/event-stream")

        # ---- non-streaming ----
        r = await client.post(url, json=body, headers=headers)
        await client.aclose()
        try:
            data = r.json()
        except Exception:
            print("UPSTREAM NON-JSON", r.status_code, r.text[:500], flush=True)
            return Response(r.content, status_code=r.status_code)
        choice = (data.get("choices") or [{}])[0].get("message", {})
        await finish(sel, up, r.status_code < 400, t0, msgs, body,
                     choice.get("content"), choice.get("tool_calls"),
                     data.get("usage") or {}, sid)
        return JSONResponse(data, status_code=r.status_code)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": {"message": f"{type(e).__name__}: {e}"}}, status_code=500)