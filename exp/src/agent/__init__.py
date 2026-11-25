"""agent — prompt、LLM 客户端、缓存、多轮主循环。

依赖方向：agent 只依赖 memory（不依赖 scorer 的模型类，只接受一个可调用的
打分模型对象），也不依赖 eval。这样 eval 能单独导入前三模块跑 L1。

模块划分：
    prompt.py  prompt 模板与答案匹配（JGA 填空口径 + 多轮上下文渲染）
    llm.py     LLMClient 协议 + LocalQwen / ApiClient / DummyLLM
    cache.py   ResponseCache，key = sha256(model + prompt + params)
    loop.py    MemoryAgent：检索 → 打分 → 选择 → 生成 → 记录 token
"""

from .cache import ResponseCache
from .llm import (ApiClient, DummyLLM, GenResult, LLMClient, LocalQwen,
                  build_llm)
from .loop import MemoryAgent, TurnResult, run_scenario, run_turn, select_all
from .prompt import (
    SLOT_BLANK,
    SYMBOLIC,
    answer_matches,
    askable_slots,
    build_prompt,
    normalize,
    render_context,
)

__all__ = [
    "ResponseCache",
    "GenResult",
    "LLMClient",
    "LocalQwen",
    "ApiClient",
    "DummyLLM",
    "build_llm",
    "MemoryAgent",
    "TurnResult",
    "run_turn",
    "run_scenario",
    "select_all",
    "SLOT_BLANK",
    "SYMBOLIC",
    "askable_slots",
    "render_context",
    "build_prompt",
    "normalize",
    "answer_matches",
]
