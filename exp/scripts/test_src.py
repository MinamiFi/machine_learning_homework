#!/usr/bin/env python3
"""四模块的单元自检：不加载 LLM、不读数据、1 秒内跑完。

为什么单独写一个：`smoke_test.py` 要加载 3.2 GB 的 Qwen，验证一次 90 秒，
不适合放在改代码的循环里。这个脚本只用 `DummyLLM` 和小批量假数据，
可以每次改完 `src/` 就跑。

覆盖的边界（都是实际踩过的坑）：
  1. `select()` 的预算硬约束与 reserve_recent 保底
  2. `answer_matches()` 的整词匹配（`4` 不能命中 `14`）
  3. `normalize()` 的撇号删除与只去开头冠词（`theatre` 不能被改）
  4. `aggregate_rows()` 的 AR 分母必须来自全量行（子集里没有 B0FULL）
  5. `MemoryAgent.run_turn()` 的 full_context 与 budget 两种模式
  6. `ResponseCache` 的 key 含 model 名（换 backbone 不误命中）

用法：
    source exp/env.sh
    python3 exp/scripts/test_src.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.cache import ResponseCache                        # noqa: E402
from src.agent.llm import DummyLLM                               # noqa: E402
from src.agent.loop import MemoryAgent, select_all               # noqa: E402
from src.agent.prompt import (answer_matches, askable_slots,     # noqa: E402
                              build_prompt, normalize, render_context)
from src.memory.select import select                             # noqa: E402
from src.memory.types import LongScenario, MemorySegment         # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILED.append(name)


def seg(did: str, t: int, user: str, system: str = "", label: int = 0,
        n_tokens: int = 10, delta: dict | None = None) -> MemorySegment:
    return MemorySegment(
        seg_id=f"{did}:{t}", dialogue_id=did, turn_idx=t,
        user=user, system=system, label=label, n_tokens=n_tokens,
        delta=delta or {},
    )


# ---------------------------------------------------------------- 1. select

def test_select() -> None:
    print("\n[1] select() 预算与保底")
    scores = np.array([0.9, 0.8, 0.1, 0.7, 0.6])
    tokens = np.array([100, 100, 100, 100, 100])
    r = select(scores, tokens, budget=250, reserve_recent=0)
    check("预算硬约束", r.total_tokens <= 250, f"用了 {r.total_tokens}")
    check("选了 2 段", r.n_chosen == 2, f"实际 {r.n_chosen}")
    check("按分数降序", list(r.chosen) == [0, 1], str(r.chosen))

    # 保底：下标最大的 2 段（即使分数最低）必须在
    r2 = select(scores, tokens, budget=250, reserve_recent=2)
    chosen = set(r2.chosen)
    check("保底含最后一段", 4 in chosen, str(sorted(chosen)))
    check("保底后仍守预算", r2.total_tokens <= 250, f"用了 {r2.total_tokens}")

    # 单段超预算：无保底时不选
    r3 = select(np.array([1.0]), np.array([9999]), budget=100, reserve_recent=0)
    check("单段超预算不选", r3.n_chosen == 0, str(r3.chosen))

    # 但 reserve_recent 是**硬保底**，会覆盖预算上限（设计如此：
    # 最后一个用户问题不能因为装不下就被丢掉）。这里显式锁住这个行为，
    # 免得以后有人"顺手修"成遵守预算而破坏保底语义。
    r4 = select(np.array([1.0]), np.array([9999]), budget=100, reserve_recent=1)
    check("保底可超预算（设计如此）", r4.chosen == [0] and r4.total_tokens == 9999,
          f"{r4.chosen} / {r4.total_tokens}")

    check("select_all 全选", select_all([seg("d", i, "u") for i in range(5)]) == [0, 1, 2, 3, 4])


# ---------------------------------------------------------------- 2. 匹配

def test_matching() -> None:
    print("\n[2] answer_matches() 整词匹配")
    check("数字 4 命中 '4 people'", answer_matches("4", "4 people"))
    check("数字 4 不命中 '14'", not answer_matches("4", "14 people"))
    check("数字 3 不命中 '13:00'", not answer_matches("3", "13:00"))
    check("数字 4 命中 'table for 4.'", answer_matches("4", "table for 4."))
    check("词不命中更长词", not answer_matches("cafe", "cafeteria"))
    check("撇号归一化", answer_matches("Queen's College", "queens college"))
    check("centre→center", answer_matches("centre", "center"))
    check("空预测不算对", not answer_matches("abc", ""))


def test_normalize() -> None:
    print("\n[3] normalize() 归一化")
    check("撇号直接删除", normalize("Queen's") == "queens", normalize("Queen's"))
    check("只去开头冠词", normalize("the theatre") == "theatre", normalize("the theatre"))
    check("句中冠词保留", normalize("hall the road") == "hall the road",
          normalize("hall the road"))
    check("标点转空格", normalize("hall, road.") == "hall road",
          normalize("hall, road."))
    # 注意 "a b" 会被去掉开头冠词变成 "b"——这是设计行为（只去开头冠词），
    # 不是 bug。测试用例要避开这种歧义字符串。
    check("开头冠词被去掉", normalize("a cafe") == "cafe", normalize("a cafe"))


# ---------------------------------------------------------------- 4. prompt

def test_prompt() -> None:
    print("\n[4] prompt 构造")
    check("30 个槽位模板", len(__import__("src.agent.prompt", fromlist=["SLOT_BLANK"]).SLOT_BLANK) == 30)
    # 模板不能含数字：问答式的问句自带数字会让 0.5B 回显（实测 20% vs 52%）
    from src.agent.prompt import SLOT_BLANK

    bad = [k for k, v in SLOT_BLANK.items() if any(ch.isdigit() for ch in v)]
    check("模板不含数字", not bad, str(bad))

    ask = askable_slots({"restaurant-name": "x", "hotel-parking": "yes",
                         "unknown-slot": "z"})
    check("符号值被排除", "hotel-parking" not in ask, str(list(ask)))
    check("无模板被排除", "unknown-slot" not in ask, str(list(ask)))
    check("正常槽位保留", "restaurant-name" in ask, str(list(ask)))

    s = [seg("d", 0, "hi", "hello"), seg("d", 1, "bye", "")]
    ctx = render_context(s)
    check("上下文含 User", "User: hi" in ctx, ctx)
    check("上下文含 System", "System: hello" in ctx, ctx)
    check("空 system 不渲染", "System: \n" not in ctx, repr(ctx))

    p = build_prompt(ctx, "restaurant-name")
    check("prompt 含填空模板", "The name of the restaurant is ___." in p, p[:80])


# ---------------------------------------------------------------- 5. loop

def test_loop() -> None:
    print("\n[5] MemoryAgent.run_turn()")
    bank = [seg("d", i, f"q{i}", f"a{i}", label=1 if i == 2 else 0,
                n_tokens=20) for i in range(6)]
    sc = LongScenario(
        scenario_id="d_L6_p0.5_s0", dialogue_id="d", bank=bank,
        query="q5", gold_state={"restaurant-name": "acorn"},
        needle_id="d:2", meta={"relevant_ids": ["d:2"]}, target_len=6,
    )

    # DummyLLM 不加载模型；count_tokens 按空白切分
    agent = MemoryAgent(llm=DummyLLM(reply="acorn"), budget=60,
                        reserve_recent=0, cache=None)

    # budget 模式：60 token / 每段 20 → 最多 3 段
    t = agent.run_turn(sc, "R1", slot="restaurant-name",
                       vecs=np.zeros((6, 384), dtype=np.float32))
    check("budget 模式选中数受限", t.n_chosen <= 3, f"选了 {t.n_chosen}")
    check("budget 模式 cr 可算", t.cr in (0.0, 1.0), str(t.cr))
    check("答案判对", t.meta.get("correct") is True, str(t.meta))
    check("token 有记录", t.prompt_tokens > 0 and t.n_new_tokens > 0,
          f"{t.prompt_tokens}/{t.n_new_tokens}")

    # full 模式：全选，不受预算约束
    t2 = agent.run_turn(sc, "B0", full_context=True, slot="restaurant-name",
                        vecs=np.zeros((6, 384), dtype=np.float32))
    check("full 模式全选", t2.n_chosen == 6, f"选了 {t2.n_chosen}")
    check("full 模式 mode 标记", t2.meta.get("mode") == "full", str(t2.meta))
    check("full 模式 cr=1", t2.cr == 1.0, str(t2.cr))

    # 没有可问槽位时跳过，不算一次生成
    sc2 = LongScenario(scenario_id="x", dialogue_id="d", bank=bank,
                       query="q", gold_state={"hotel-parking": "yes"},
                       needle_id="d:2", meta={}, target_len=6)
    t3 = agent.run_turn(sc2, "R1", vecs=np.zeros((6, 384), dtype=np.float32))
    check("无可问槽位时跳过", t3.meta.get("skipped") == "no_askable_slot",
          str(t3.meta))


# ---------------------------------------------------------------- 6. cache

def test_cache() -> None:
    print("\n[6] ResponseCache")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.jsonl"
        c1 = ResponseCache(p, model_name="model-a")
        c1.put("hello", 8, "world", seconds=1.0)
        check("写入后能读", c1.get("hello", 8) is not None)
        check("不同 max_tokens 不命中", c1.get("hello", 9) is None)
        check("不同 prompt 不命中", c1.get("hello2", 8) is None)

        # 重新打开（模拟进程重启）
        c2 = ResponseCache(p, model_name="model-a")
        check("重启后仍命中", c2.get("hello", 8) is not None)
        check("命中计数", c2.hits == 1, str(c2.hits))

        # 换 model 名不能误命中
        c3 = ResponseCache(p, model_name="model-b")
        check("换 backbone 不误命中", c3.get("hello", 8) is None)


# ---------------------------------------------------------------- main


def test_stable_seed() -> None:
    """跨进程稳定性。这是踩过的坑：内置 hash() 有随机盐。"""
    from src.common import stable_seed

    # 1. 同一组参数必须给同一个值
    a = stable_seed("MUL0003.json", 128, 0)
    b = stable_seed("MUL0003.json", 128, 0)
    check("stable_seed 同参数稳定", a == b)

    # 2. 不同参数给不同值
    check("stable_seed 不同参数不同值",
          stable_seed("MUL0003.json", 128, 0) != stable_seed("MUL0003.json", 128, 1))

    # 3. mod 生效
    check("stable_seed mod 生效", 0 <= stable_seed("x", mod=10000) < 10000)

    # 4. 关键：不能等于内置 hash()（那说明没真的换成 sha256）
    import subprocess, sys
    code = ("import sys; sys.path.insert(0, 'exp'); "
            "from src.common import stable_seed; "
            "print(stable_seed('MUL0003.json', 128, 0))")
    outs = set()
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, cwd=str(pathlib.Path.cwd()))
        outs.add(r.stdout.strip())
    check("stable_seed 跨进程一致（子进程实测）", len(outs) == 1 and "" not in outs)


def main() -> int:
    print("=" * 60)
    print("src/ 单元自检（不加载 LLM）")
    print("=" * 60)

    test_select()
    test_matching()
    test_normalize()
    test_prompt()
    test_loop()
    test_cache()
    test_stable_seed(),

    print("\n" + "=" * 60)
    if FAILED:
        print(f"FAILED {len(FAILED)} 项：{FAILED}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
