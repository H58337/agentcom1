import requests
import time
import threading
import os


_USAGE_LOCK = threading.Lock()
_USAGE_LOCAL = threading.local()
_LLM_USAGE_STATS = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "estimated_cost": 0.0,
    "per_model": {},
}


def _normalize_usage(usage):
    if not isinstance(usage, dict):
        return 0, 0, 0

    prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def _estimate_cost(prompt_tokens, completion_tokens, args):
    in_price = float(getattr(args, "llm_input_price_per_mtoken", 0.0) or 0.0)
    out_price = float(getattr(args, "llm_output_price_per_mtoken", 0.0) or 0.0)
    return (prompt_tokens / 1_000_000.0) * in_price + (completion_tokens / 1_000_000.0) * out_price


def _record_llm_usage(provider, model, usage, args):
    prompt_tokens, completion_tokens, total_tokens = _normalize_usage(usage)
    est_cost = _estimate_cost(prompt_tokens, completion_tokens, args)

    model_key = f"{provider}:{model}"
    with _USAGE_LOCK:
        _LLM_USAGE_STATS["calls"] += 1
        call_id = int(_LLM_USAGE_STATS["calls"])
        _LLM_USAGE_STATS["prompt_tokens"] += prompt_tokens
        _LLM_USAGE_STATS["completion_tokens"] += completion_tokens
        _LLM_USAGE_STATS["total_tokens"] += total_tokens
        _LLM_USAGE_STATS["estimated_cost"] += est_cost

        if model_key not in _LLM_USAGE_STATS["per_model"]:
            _LLM_USAGE_STATS["per_model"][model_key] = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
            }

        row = _LLM_USAGE_STATS["per_model"][model_key]
        row["calls"] += 1
        row["prompt_tokens"] += prompt_tokens
        row["completion_tokens"] += completion_tokens
        row["total_tokens"] += total_tokens
        row["estimated_cost"] += est_cost

    call_usage = {
        "call_id": int(call_id),
        "provider": str(provider),
        "model": str(model),
        "model_key": str(model_key),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "estimated_cost": float(est_cost),
    }
    _USAGE_LOCAL.last_call_usage = call_usage
    return call_usage


def _apply_optional_max_tokens(payload, args):
    try:
        max_tokens = int(getattr(args, "max_tokens", 0) or 0)
    except Exception:
        max_tokens = 0
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    return payload


def pop_last_llm_call_usage():
    usage = getattr(_USAGE_LOCAL, "last_call_usage", None)
    _USAGE_LOCAL.last_call_usage = None
    return usage if isinstance(usage, dict) else None


def reset_llm_usage_stats():
    with _USAGE_LOCK:
        _LLM_USAGE_STATS["calls"] = 0
        _LLM_USAGE_STATS["prompt_tokens"] = 0
        _LLM_USAGE_STATS["completion_tokens"] = 0
        _LLM_USAGE_STATS["total_tokens"] = 0
        _LLM_USAGE_STATS["estimated_cost"] = 0.0
        _LLM_USAGE_STATS["per_model"] = {}


def get_llm_usage_stats(reset=False):
    with _USAGE_LOCK:
        snapshot = {
            "calls": int(_LLM_USAGE_STATS["calls"]),
            "prompt_tokens": int(_LLM_USAGE_STATS["prompt_tokens"]),
            "completion_tokens": int(_LLM_USAGE_STATS["completion_tokens"]),
            "total_tokens": int(_LLM_USAGE_STATS["total_tokens"]),
            "estimated_cost": float(_LLM_USAGE_STATS["estimated_cost"]),
            "per_model": {
                str(k): {
                    "calls": int(v.get("calls", 0)),
                    "prompt_tokens": int(v.get("prompt_tokens", 0)),
                    "completion_tokens": int(v.get("completion_tokens", 0)),
                    "total_tokens": int(v.get("total_tokens", 0)),
                    "estimated_cost": float(v.get("estimated_cost", 0.0)),
                }
                for k, v in _LLM_USAGE_STATS["per_model"].items()
            },
        }

    if reset:
        reset_llm_usage_stats()
    return snapshot


def print_llm_usage_summary(stage="stage", reset=False):
    stats = get_llm_usage_stats(reset=reset)
    calls = int(stats.get("calls", 0))
    prompt_tokens = int(stats.get("prompt_tokens", 0))
    completion_tokens = int(stats.get("completion_tokens", 0))
    total_tokens = int(stats.get("total_tokens", 0))
    est_cost = float(stats.get("estimated_cost", 0.0))
    cost_part = f", estimated_cost={est_cost:.6f}" if est_cost > 0 else ""
    print(
        f"[LLMUsage][{str(stage).upper()}] calls={calls} "
        f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
        f"total_tokens={total_tokens}{cost_part}",
        flush=True,
    )
    return stats

def api_request(system_prompt, user_prompt, args):
    _USAGE_LOCAL.last_call_usage = None
    model = (args.model or "").lower()

    if "gpt" in model:
        return gpt_api(system_prompt, user_prompt, args)
    elif "mimo" in model or "xiaomi" in model:
        return xiaomi_mimo_api(system_prompt, user_prompt, args)
    elif "glm" in model:
        return glm_api(system_prompt, user_prompt, args)
    elif "qwen" in model:
        return qwen_api(system_prompt, user_prompt, args)
    elif "deepseek" in model:
        return deepseek_api(system_prompt, user_prompt, args)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


def _join_chat_completions_url(base_url, default_base):
    base = str(base_url or default_base or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _resolve_xiaomi_api_key(args):
    env_key = str(os.environ.get("XIAOMI_API_KEY", "") or "").strip()
    arg_key = str(getattr(args, "api_key", "") or "").strip()
    if env_key:
        return env_key, "XIAOMI_API_KEY"
    return arg_key, "--api_key"


def xiaomi_mimo_api(system_prompt, user_prompt, args):
    """
    Xiaomi MiMo OpenAI-compatible Chat Completions API.

    Default endpoint:
      https://api.xiaomimimo.com/v1/chat/completions

    Use --llm_base_url to override, for example a self-hosted SGLang endpoint:
      --llm_base_url http://localhost:9001/v1
    """
    max_retry_num = args.max_retry_num
    url = _join_chat_completions_url(
        getattr(args, "llm_base_url", ""),
        "https://api.xiaomimimo.com/v1",
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_resolve_xiaomi_api_key(args)[0]}",
    }

    model_name = str(args.model or "mimo-v2-flash")
    if model_name.startswith("xiaomi/"):
        model_name = model_name.split("/", 1)[1]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": args.temperature,
        "stream": False,
    }
    _apply_optional_max_tokens(payload, args)

    while max_retry_num >= 0:
        request_result = None
        try:
            request_result = requests.post(url, headers=headers, json=payload, timeout=120)
            result_json = request_result.json()
            if "error" not in result_json:
                model_output = result_json["choices"][0]["message"]["content"]
                _record_llm_usage("xiaomi_mimo", model_name, result_json.get("usage", {}), args)
                return model_output.strip()
            else:
                print("[warning] Xiaomi MiMo error:", result_json.get("error"))
                max_retry_num -= 1
                time.sleep(2)
        except Exception as e:
            if request_result is not None:
                try:
                    print("[warning] Xiaomi MiMo request_result =", request_result.json())
                except Exception:
                    print("[warning] Xiaomi MiMo request_result (non-json) =", request_result.text)
            else:
                print("[warning] Xiaomi MiMo request_result = NULL")
            print("[warning] Xiaomi MiMo exception:", repr(e))
            max_retry_num -= 1
            time.sleep(2)
    return None


def gpt_api(system_prompt, user_prompt, args):
    max_retry_num = args.max_retry_num
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}"
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": args.model,
        "messages": messages,
        # "temperature": args.temperature,
    }
    _apply_optional_max_tokens(payload, args)

    while max_retry_num >= 0:
        request_result = None
        try:
            request_result = requests.post(url, headers=headers, json=payload, timeout=120)
            result_json = request_result.json()
            if 'error' not in result_json:
                model_output = result_json['choices'][0]['message']['content']
                _record_llm_usage("openai", str(args.model), result_json.get("usage", {}), args)
                return model_output.strip()
            else:
                print("[warning] OpenAI error:", result_json.get("error"))
                max_retry_num -= 1
                time.sleep(2)
        except Exception as e:
            if request_result is not None:
                try:
                    print("[warning] OpenAI request_result =", request_result.json())
                except Exception:
                    print("[warning] OpenAI request_result (non-json) =", request_result.text)
            else:
                print("[warning] OpenAI request_result = NULL")
            print("[warning] OpenAI exception:", repr(e))
            max_retry_num -= 1
            time.sleep(2)
    return None


def qwen_api(system_prompt, user_prompt, args):
    """
    Qwen (DashScope) OpenAI-compatible Chat Completions API.

    Endpoint (compatible-mode):
      https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions

    Auth:
      Authorization: Bearer {args.api_key}

    args.model examples:
      qwen-turbo, qwen-plus, qwen-max, ...
    """
    max_retry_num = args.max_retry_num
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        # 可选：如果你想控制最大输出
        # "max_tokens": 512,
    }
    _apply_optional_max_tokens(payload, args)

    while max_retry_num >= 0:
        request_result = None
        try:
            request_result = requests.post(url, headers=headers, json=payload, timeout=120)
            result_json = request_result.json()

            if "error" not in result_json:
                model_output = result_json["choices"][0]["message"]["content"]
                _record_llm_usage("qwen", str(args.model), result_json.get("usage", {}), args)
                return model_output.strip()
            else:
                print("[warning] Qwen error:", result_json.get("error"))
                max_retry_num -= 1
                time.sleep(2)
        except Exception as e:
            if request_result is not None:
                try:
                    print("[warning] Qwen request_result =", request_result.json())
                except Exception:
                    print("[warning] Qwen request_result (non-json) =", request_result.text)
            else:
                print("[warning] Qwen request_result = NULL")
            print("[warning] Qwen exception:", repr(e))
            max_retry_num -= 1
            time.sleep(2)
    return None


def glm_api(system_prompt, user_prompt, args):
    """
    Zhipu BigModel Chat Completions API (OpenAI-like schema, but different base URL).
    """
    max_retry_num = args.max_retry_num
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "stream": False,
        "thinking": {"type": "disabled"},
        "do_sample": True,
        "top_p": 0.95,
        "tool_stream": False,
        "response_format": {"type": "text"},
    }
    _apply_optional_max_tokens(payload, args)

    while max_retry_num >= 0:
        request_result = None
        try:
            request_result = requests.post(url, headers=headers, json=payload, timeout=120)
            result_json = request_result.json()

            if "error" not in result_json:
                model_output = result_json["choices"][0]["message"]["content"]
                _record_llm_usage("glm", str(args.model), result_json.get("usage", {}), args)
                return model_output.strip()
            else:
                print("[warning] GLM error:", result_json.get("error"))
                max_retry_num -= 1
                time.sleep(2)
        except Exception as e:
            if request_result is not None:
                try:
                    print("[warning] GLM request_result =", request_result.json())
                except Exception:
                    print("[warning] GLM request_result (non-json) =", request_result.text)
            else:
                print("[warning] GLM request_result = NULL")
            print("[warning] GLM exception:", repr(e))
            max_retry_num -= 1
            time.sleep(2)
    return None

def deepseek_api(system_prompt, user_prompt, args):
    max_retry_num = args.max_retry_num
    url = "https://api.deepseek.com/v1/chat/completions"  # 也可用 base_url=https://api.deepseek.com/v1 [web:303]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": args.model,          # 建议传 deepseek-chat 或 deepseek-reasoner [web:303]
        "messages": messages,
        "temperature": args.temperature,
    }
    _apply_optional_max_tokens(payload, args)

    while max_retry_num >= 0:
        request_result = None
        try:
            request_result = requests.post(url, headers=headers, json=payload, timeout=120)
            result_json = request_result.json()

            if "error" not in result_json:
                model_output = result_json["choices"][0]["message"]["content"]
                _record_llm_usage("deepseek", str(args.model), result_json.get("usage", {}), args)
                return model_output.strip()
            else:
                print("[warning] DeepSeek error:", result_json.get("error"))
                max_retry_num -= 1
                time.sleep(2)
        except Exception as e:
            if request_result is not None:
                try:
                    print("[warning] DeepSeek request_result =", request_result.json())
                except Exception:
                    print("[warning] DeepSeek request_result (non-json) =", request_result.text)
            else:
                print("[warning] DeepSeek request_result = NULL")
            print("[warning] DeepSeek exception:", repr(e))
            max_retry_num -= 1
            time.sleep(2)
    return None


# import requests
# import time

# def api_request(system_prompt, user_prompt, args):
#     if "gpt" in args.model:
#         return gpt_api(system_prompt, user_prompt, args)
#     else:
#         raise ValueError(f"Unsupported model: {args.model}") 



# def gpt_api(system_prompt, user_prompt, args):
#     max_retry_num = args.max_retry_num
#     url = "https://api.openai.com/v1/chat/completions"
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {args.api_key}"
#     }
    
#     messages = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt},
#     ]

#     payload = {
#         "model": args.model, 
#         "messages": messages,
#         "temperature": args.temperature,
#     }
#     while max_retry_num >= 0:
#         request_result = None
#         try:
#             request_result = requests.post(url, headers=headers, json=payload)
#             result_json = request_result.json()
#             if 'error' not in result_json: 
#                 model_output = result_json['choices'][0]['message']['content']
#                 return model_output.strip()
#             else:
#                 max_retry_num -= 1
#         except:
#             if request_result is not None:
#                 print("[warning]request_result = ", request_result.json())
#                 time.sleep(2)
#             else:
#                 print("[warning]request_result = NULL")
#             max_retry_num -= 1
#     return None




            
