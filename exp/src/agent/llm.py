"""LLM 客户端。本机只有 0.5B，所以 LocalQwen 是开发用，正式实验走 ApiClient。

关键约束（`exp/README.md` §3）：Qwen2.5-0.5B 在 2 核 4 GB 上 fp32 峰值
3.24 GB，只能打通管线，**不能出主表**——它的对话状态追踪能力太弱。
所以本模块的定位是：让 L2 的**配对**比较能跑（比"删掉 X 后答对率掉多少"，
而不是比绝对 JGA），以及在换 API key 前把管线调通。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

__all__ = ["LLMClient", "GenResult", "LocalQwen", "ApiClient", "DummyLLM", "build_llm"]


@dataclass
class GenResult:
    text: str
    n_prompt_tokens: int = 0
    n_new_tokens: int = 0
    seconds: float = 0.0
    meta: dict = field(default_factory=dict)


class LLMClient(Protocol):
    """所有 backbone 的共同接口。agent 主循环只依赖这个。"""

    name: str

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> GenResult:
        ...

    def count_tokens(self, text: str) -> int:
        ...


class LocalQwen:
    """本地 Qwen2.5-0.5B-Instruct。

    两个必须注意的点：
      1. 必须传 `attention_mask`。Qwen 的 pad_token == eos_token，
         transformers 推断不出 mask 会直接警告并按全 1 处理，
         批量时可能出错。
      2. 必须 `add_generation_prompt=True`。否则模型会接着"用户"那一轮
         往下写，输出常以 `user\\n` 开头（实测踩过），把答案污染掉。
    """

    def __init__(self, path: str, dtype: str = "float32",
                 max_new_tokens: int = 128, temperature: float = 0.0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"local:{os.path.basename(path)}"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=getattr(torch, dtype))
        self.model.eval()
        # 加载后把采样参数关掉，避免 transformers 每次生成都警告
        self.model.generation_config.do_sample = temperature > 0
        self.model.generation_config.temperature = temperature or None
        self.model.generation_config.top_p = 1.0
        self.model.generation_config.top_k = 0

    def count_tokens(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> GenResult:
        torch = self._torch
        mx = max_new_tokens or self.max_new_tokens
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_special_tokens=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=mx,
                do_sample=self.temperature > 0,
                attention_mask=torch.ones_like(ids),
                pad_token_id=self.tok.eos_token_id,
            )
        new = out[0][ids.shape[1]:]
        return GenResult(
            text=self.tok.decode(new, skip_special_tokens=True).strip(),
            n_prompt_tokens=int(ids.shape[1]),
            n_new_tokens=int(new.shape[0]),
            seconds=time.time() - t0,
        )


class DummyLLM:
    """不加载模型，返回固定串。用于测管线、跑单元测试。"""

    name = "dummy"

    def __init__(self, reply: str = "unknown"):
        self.reply = reply

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> GenResult:
        return GenResult(text=self.reply, n_prompt_tokens=len(prompt.split()))


class ApiClient:
    """任意 OpenAI 兼容端点（DeepSeek / Qwen / vLLM / Ollama 都可）。

    只依赖 `requests`（已装），不引入 openai SDK——多一个依赖就多一个
    在 4 GB 容器里装不上的风险。

    **API key 只从环境变量读，绝不写进 config 文件或代码**。
    `api.api_key_env` 指定变量名（默认 `DEEPSEEK_API_KEY`）。

    token 计数用端点返回的 `usage`（比本地估准），
    `count_tokens` 在拿不到 tokenizer 时退化为按空白切分——
    这只影响 `context_tokens` 这个辅助指标，不影响 `prompt_tokens`
    （后者来自服务端 usage）。
    """

    def __init__(self, base_url: str, model: str, api_key_env: str = "DEEPSEEK_API_KEY",
                 temperature: float = 0.0, max_tokens: int = 256, timeout: float = 120.0,
                 reasoning_effort: str | None = "none"):
        import os

        if not base_url:
            raise ValueError("ApiClient 需要 base_url")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            raise ValueError(
                f"环境变量 {api_key_env} 未设置。API key 只从环境变量读，"
                f"不要写进配置文件。设置方法：export {api_key_env}=...")
        self.name = f"api:{model}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # 同济 agi 端点的 DeepSeek-V4.1-Flash 是**推理模型**：默认先想再答，
        # 40 个 reasoning token 才吐一个 `08:45`。填空任务只需要几个 token，
        # 若不关推理，max_tokens=24 会全部烧在思考上、content 返回 null
        # （实测 16/24 都是空，64 才够）。`reasoning_effort="none"` 实测把
        # reasoning_tokens 压到 0、延迟从 0.68s 降到 0.36s，答案不变。
        self.reasoning_effort = reasoning_effort

    def count_tokens(self, text: str) -> int:
        # 无本地 tokenizer。按空白估，偏小约 20%（英文口语文本）。
        # 只用于 context_tokens 辅助列，不用于报告主表的 token 消耗。
        return len(text.split())

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> GenResult:
        import time

        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_new_tokens or self.max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        t0 = time.time()
        # 429 = 套餐并发上限（该端点实测 3 个）。必须退避重试，否则整批
        # 实验会因个别请求撞限而中断——并发跑几千条时这几乎是必然事件。
        last_err = ""
        for attempt in range(6):
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            if r.status_code == 200:
                break
            last_err = f"API {r.status_code}: {r.text[:300]}"
            if r.status_code in (429, 500, 502, 503, 504):
                import random
                time.sleep(min(2 ** attempt + random.random(), 30))
                continue
            raise RuntimeError(last_err)
        else:
            raise RuntimeError(f"重试 6 次仍失败：{last_err}")
        d = r.json()
        msg = d["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        usage = d.get("usage") or {}
        n_reason = int((usage.get("completion_tokens_details") or {})
                       .get("reasoning_tokens", 0) or 0)
        return GenResult(
            text=text,
            n_prompt_tokens=int(usage.get("prompt_tokens", 0)),
            n_new_tokens=int(usage.get("completion_tokens", 0)),
            seconds=time.time() - t0,
            meta={"model": d.get("model", self.model),
                  # 空 content 且 reasoning>0 时，说明 token 预算被思考吃光了。
                  # 这是配置错误而非模型答错，必须能看见。
                  "reasoning_tokens": n_reason,
                  "truncated_by_reasoning": bool(not text and n_reason > 0)},
        )


def build_llm(cfg: dict, max_new_tokens: int | None = None) -> LLMClient:
    """按 `configs/agent.yaml` 造 backbone。

    三种：`local`（默认，0.5B）、`api`（正式实验）、`dummy`（测管线）。
    """
    from ..common import path_of

    backbone = (cfg.get("backbone") or "local").lower()
    if backbone == "dummy":
        return DummyLLM(reply=cfg.get("dummy", {}).get("reply", "unknown"))
    if backbone == "api":
        api = cfg.get("api", {})
        return ApiClient(
            base_url=api.get("base_url", ""),
            model=api.get("model", ""),
            api_key_env=api.get("api_key_env", "DEEPSEEK_API_KEY"),
            temperature=api.get("temperature", 0.0),
            max_tokens=api.get("max_tokens", 256),
            reasoning_effort=api.get("reasoning_effort", "none"),
        )
    local = cfg.get("local", {})
    return LocalQwen(
        str(path_of(local["path"])),
        dtype=local.get("dtype", "float32"),
        max_new_tokens=max_new_tokens or local.get("max_new_tokens", 128),
        temperature=local.get("temperature", 0.0),
    )
