from datetime import datetime

from langchain.agents.middleware import wrap_model_call, wrap_tool_call


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@wrap_model_call
def model_logging_middleware(request, handler):
    t = _ts()
    messages = request.state["messages"]
    print(f"\n📤 [{t}] MODEL REQUEST — {len(messages)} messages")
    if messages:
        last = messages[-1]
        text = last.content
        if isinstance(text, str) and len(text) > 400:
            text = text[:400] + "..."
        print(f"   Last message: {text}")
    try:
        response = handler(request)
        print(f"✅ [{t}] MODEL RESPONSE")
        result = response.result
        if result:
            msg = result[0] if isinstance(result, (list, tuple)) else result
            if msg.content is not None:
                s = str(msg.content)
                print(f"   Output: {s[:400]}{'...' if len(s) > 400 else ''}")
            for tc in msg.tool_calls or []:
                print(f"   Tool call: {tc}")
        return response
    except Exception as e:
        print(f"❌ [{t}] MODEL ERROR: {e}")
        raise


@wrap_tool_call
def tool_logging_middleware(request, handler):
    t = _ts()
    last = request.state["messages"][-1]
    print(f"\n🔧 [{t}] TOOL — tool_calls: {last.tool_calls}")
    try:
        out = handler(request)
        print(f"✅ [{t}] TOOL finished")
        return out
    except Exception as e:
        print(f"❌ [{t}] TOOL ERROR: {e}")
        raise


def get_logging_middleware() -> list:
    return [model_logging_middleware, tool_logging_middleware]


__all__ = ["model_logging_middleware", "tool_logging_middleware", "get_logging_middleware"]