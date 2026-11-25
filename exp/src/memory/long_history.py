"""长历史构造：把 7.7 轮的短对话膨胀到 L=128/256/512 轮，并埋入 needle 与诱饵。

为什么必须做：MultiWOZ 平均只有 7.7 轮，约 300 token，
远不到"上下文窗口溢出"。题目要考察长上下文，就必须自己造长历史。

**关键设计：query 不能是 needle 本身。**

第一版我把 query 设成 needle 的 user 文本，结果 B1（纯向量检索）会
天然把 needle 排到第一——因为 query 和 needle 文本一模一样。
CR 直接满分，整个 L1 失去意义。

正确设定（与真实智能体一致）：
    - query  = 对话的**最后一轮**用户请求（当前要回答的问题）
    - bank   = 该轮**之前**的所有历史轮 + 干扰轮（query 不在库里）
    - needle = 历史中某一轮，携带了回答 query 所必需的信息
               （label=1，即该轮引入的 slot 值到对话结束仍有效）

这样 needle 与 query 是**不同**的文本，检索难度是真实的。

合成方式（根 README §4.2）：
  1. 取一段真实对话作为主对话，最后一轮作为 query，其余作为历史；
  2. 从同领域其他对话抽轮次作为干扰，插在历史里；
  3. needle 在历史中挑（label=1 优先），按 needle_pos 指定的相对位置放入；
  4. 额外埋 decoy：与 needle 同槽位、但值不同，用于检验打分器
     是不是只做了"领域匹配"而不是"值匹配"。

needle_pos 必须随机（0.0–0.9），不能固定在前 10–20%：
标签有强首因偏置（正例率 turn0 85% → turn10 16%），
位置固定的话，任何位置基线都会自动赢。
"""

from __future__ import annotations

import random

import numpy as np

from ..common import get_logger, stable_seed
from .slice import group_by_dialogue, segment_text
from .types import LongScenario, MemorySegment

log = get_logger(__name__)


def _primary_domain(segs: list[MemorySegment]) -> str:
    from collections import Counter

    c = Counter()
    for s in segs:
        for d in s.services:
            c[d] += 1
    return c.most_common(1)[0][0] if c else "restaurant"


CLOSING_PATTERNS = (
    "all set", "thank you", "thanks", "goodbye", "have a nice day",
    "that is all", "that's all", "that is it", "that's it", "no that is",
)


def is_substantive(seg: MemorySegment) -> bool:
    """这一轮是否承载实际内容。

    MultiWOZ 100% 的对话以 "none" 或客套话结尾（实测 1000/1000），
    直接拿最后一轮当 query 会得到空问题，整个 L1 失去意义。

    判据用**数据驱动**的 delta，而不是文本正则：
    客套话轮不改变对话状态（delta 为空），而"我要订 6 人桌"这类
    实质请求一定带来新槽位。实测比关键词匹配可靠得多——
    关键词会漏掉 "I am sure. Thank you so much." 这类混合句。
    """
    u = (seg.user or "").strip().lower()
    if not u or u == "none":
        return False
    if seg.delta:
        return True
    # delta 为空时，用文本兜底排除纯客套
    return not any(p in u for p in CLOSING_PATTERNS)


def last_substantive(turns: list[MemorySegment]) -> MemorySegment:
    """从后往前找第一个实质轮。"""
    for s in reversed(turns):
        if is_substantive(s):
            return s
    return turns[-1]


def _pick_needle(
    history: list[MemorySegment],
    query_seg: MemorySegment,
    rng: random.Random,
) -> MemorySegment:
    """挑 needle：优先选"其引入的 slot 值到对话结束仍有效、且与 query 相关"的轮次。

    判据：label=1（delta 非空且至少一个 slot 在结束时仍有效），
    且该轮的 delta 槽位出现在 query 轮的状态里——说明这些信息
    确实是回答当前请求时要用到的。
    """
    final = query_seg.state
    cands = [s for s in history if s.label == 1 and s.delta]
    if not cands:
        return rng.choice(history)

    # 打分：delta 里有多少槽位在对话结束时仍是原值
    def n_still_valid(s: MemorySegment) -> int:
        return sum(1 for k, v in s.delta.items() if final.get(k) == v)

    scored = [(n_still_valid(s), s) for s in cands]
    best = max(n for n, _ in scored)
    top = [s for n, s in scored if n == best]
    return rng.choice(top)


def build_long_history(
    dialogue_id: str,
    target_len: int,
    by_dialogue: dict[str, list[MemorySegment]],
    needle_pos: float = 0.5,
    seed: int = 0,
    n_decoy: int = 3,
    pool_cap: int = 4000,
    hard_distractor_ratio: float = 0.2,
) -> LongScenario:
    """构造一条长度为 target_len 的长历史场景。

    needle_pos ∈ [0, 1)：needle 在记忆库中的相对位置。
    返回的 bank 已按时间序排列，needle 落在 int(needle_pos * (len(bank)-1)) 处。
    """
    rng = random.Random(seed)

    if dialogue_id not in by_dialogue:
        raise KeyError(f"对话 {dialogue_id} 不存在")

    primary = by_dialogue[dialogue_id]
    if len(primary) < 3:
        raise ValueError(f"{dialogue_id} 只有 {len(primary)} 轮，无法构造历史")

    # ---- 1. 切出 query（最后一个实质轮）与 history（其之前的所有轮）
    # 不能直接取 primary[-1]：实测 100% 的对话以 "none"/客套话结尾
    query_seg = last_substantive(primary)
    history = [s for s in primary if s.turn_idx < query_seg.turn_idx]
    if not history:
        history = [s for s in primary if s.seg_id != query_seg.seg_id]
    domain = _primary_domain(primary)

    # ---- 2. 挑 needle
    needle = _pick_needle(history, query_seg, rng)

    # ---- 3. 抽干扰轮：同领域、来自其他对话
    # 分两类：
    #   软干扰 label=0 —— 纯冗余历史，对应题目说的「记忆冗余」
    #   硬干扰 label=1 —— 在它自己的对话里重要，但对当前 query 无关
    # 硬干扰是真实的多会话场景（上一段会话的重点在下一段可能是噪声），
    # 也是纯文本打分器的固有盲区：它无法知道某轮属于哪段对话。
    other_ids = [d for d, v in by_dialogue.items()
                 if d != dialogue_id and v and domain in v[0].services]
    pool: list[MemorySegment] = []
    for d in other_ids:
        pool.extend(by_dialogue[d])
        if len(pool) >= pool_cap:
            break
    pool = [s for s in pool if s.seg_id != needle.seg_id]

    # 关键：干扰轮的 turn_idx 必须与主对话历史轮的 turn_idx 分布匹配。
    #
    # 否则会出现一个假的"位置优势"：主对话的相关轮都是对话前几轮
    # （首因偏置，正例率 turn0 85% → turn10 16%），而干扰轮均匀取自
    # 各轮次。于是 -turn_idx 和 LEN 这两个与内容无关的基线会凭空拿到
    # 0.5+ 的 CR，看起来像"位置很重要"，其实是采样偏差。
    #
    # 修法：从干扰池里只取 turn_idx 落在主对话历史范围内的轮次。
    hist_idx = {s.turn_idx for s in history}
    pool = [s for s in pool if s.turn_idx in hist_idx] or pool

    soft = [s for s in pool if s.label == 0]
    hard = [s for s in pool if s.label == 1]
    rng.shuffle(soft)
    rng.shuffle(hard)

    others_primary = [s for s in history if s.seg_id != needle.seg_id]
    n_fill = max(0, target_len - 1 - len(others_primary))   # -1 给 needle
    n_hard = min(int(n_fill * hard_distractor_ratio), len(hard))
    n_soft = min(n_fill - n_hard, len(soft))
    distractors = soft[:n_soft] + hard[:n_hard]
    rng.shuffle(distractors)

    # ---- 4. 混排：主对话轮次**均匀铺开**，needle 落在指定相对位置
    #
    # 这里踩过一个严重的坑：最初写成"每个位置以 50% 概率优先放主对话轮"，
    # 结果主对话的 7 轮在前 14 个位置就放完了，后面 240 轮全是干扰。
    # 后果是所有相关轮都挤在开头，R0（取最早的）Recall 高达 0.81，
    # 位置基线凭空获胜，评测完全失效。
    #
    # 正确做法：把主对话轮次的目标位置均匀分布在 [0, final_len) 上，
    # 只让 needle 单独占据 needle_pos 指定的位置。
    final_len = 1 + len(others_primary) + len(distractors)
    needle_idx = int(round(needle_pos * (final_len - 1)))
    needle_idx = max(0, min(needle_idx, final_len - 1))

    # 主对话其余轮的目标位置：在去掉 needle 位置后的空位里等距取点
    free_slots = [p for p in range(final_len) if p != needle_idx]
    n_op = len(others_primary)
    if n_op:
        # 等距 + 抖动，避免所有场景的相关轮都落在同一批下标上
        stride = len(free_slots) / n_op
        target = []
        for k in range(n_op):
            base = int(k * stride)
            jitter = rng.randint(0, max(0, int(stride) - 1))
            target.append(min(base + jitter, len(free_slots) - 1))
        # 去重并补齐（等距取点一般不会重复，抖动可能造成）
        target = sorted(set(target))
        while len(target) < n_op:
            cand = rng.randrange(len(free_slots))
            if cand not in target:
                target.append(cand)
        target = sorted(target)[:n_op]
    else:
        target = []

    slot_to_seg: dict[int, MemorySegment] = {}
    for slot, seg in zip(target, others_primary):
        slot_to_seg[slot] = seg

    bank: list[MemorySegment] = []
    di = 0
    for pos in range(final_len):
        if pos == needle_idx:
            bank.append(needle)
        elif pos in slot_to_seg:
            bank.append(slot_to_seg[pos])
        else:
            if di < len(distractors):
                bank.append(distractors[di])
                di += 1
    # 兜底：没用完的干扰轮（正常情况下不会有）
    bank.extend(distractors[di:])
    bank = bank[:final_len]

    # ---- 5. 诱饵：同槽位、不同值
    decoys: list[MemorySegment] = []
    needle_slots = set(needle.delta.keys())
    bank_ids = {s.seg_id for s in bank}
    if needle_slots:
        for s in pool:
            if s.seg_id in bank_ids:
                continue
            shared = needle_slots & set(s.delta.keys())
            if shared and any(s.delta[k] != needle.delta[k] for k in shared):
                decoys.append(s)
                if len(decoys) >= n_decoy:
                    break

    # 场景内的"相关"只算主对话里仍有效的轮次；干扰轮对本 query 一律不相关
    relevant_ids = [s.seg_id for s in bank
                    if s.dialogue_id == dialogue_id and s.label == 1]

    return LongScenario(
        scenario_id=f"{dialogue_id}_L{target_len}_p{needle_pos:.2f}_s{seed}",
        dialogue_id=dialogue_id,
        bank=bank,
        query=query_seg.user,
        gold_state=dict(query_seg.state),
        needle_id=needle.seg_id,
        decoy_ids=[d.seg_id for d in decoys],
        meta={
            "domain": domain,
            "query_turn_idx": query_seg.turn_idx,
            "n_primary": len(primary),
            "n_history": len(history),
            "needle_idx": needle_idx,
            "needle_turn_idx": needle.turn_idx,
            "needle_delta": needle.delta,
            "needle_pos_actual": round(needle_idx / max(final_len - 1, 1), 3),
            "relevant_ids": relevant_ids,
            "n_relevant": len(relevant_ids),
            "n_hard_distractor": n_hard,
            "n_soft_distractor": n_soft,
        },
        target_len=final_len,
        needle_pos=needle_pos,
        seed=seed,
        n_distractors=len(distractors),
    )


def build_scenarios(
    segs: list[MemorySegment],
    lengths: tuple[int, ...] = (128, 256, 512),
    n_per_length: int = 30,
    seeds: tuple[int, ...] = (0, 1, 2),
    needle_pos_range: tuple[float, float] = (0.0, 0.9),
    min_turns: int = 4,
    hard_distractor_ratio: float = 0.2,
) -> list[LongScenario]:
    """批量构造评测场景。

    每条对话在每个长度上生成 len(seeds) 个场景，needle_pos 在范围内随机
    （不是固定值——这是对首因偏置的唯一有效对冲）。
    """
    by_dialogue = group_by_dialogue(segs)
    ids = [d for d in sorted(by_dialogue) if len(by_dialogue[d]) >= min_turns]
    out: list[LongScenario] = []

    for L in lengths:
        chosen = ids[:n_per_length]
        for did in chosen:
            for seed in seeds:
                rng = random.Random(stable_seed(did, L, seed))
                pos = rng.uniform(*needle_pos_range)
                out.append(
                    build_long_history(
                        did, L, by_dialogue,
                        needle_pos=round(pos, 3), seed=seed,
                        hard_distractor_ratio=hard_distractor_ratio,
                    )
                )
    log.info("构造 %d 个长历史场景（L=%s, 每种 %d 对话 × %d seed）",
             len(out), lengths, n_per_length, len(seeds))
    return out


def scenario_token_stats(scenarios: list[LongScenario]) -> dict:
    """场景的实际 token 量。用来验证 L 是否真的达到了目标长度。"""
    import statistics

    by_L: dict[int, list[int]] = {}
    for sc in scenarios:
        toks = sum(s.n_tokens for s in sc.bank)
        by_L.setdefault(sc.target_len, []).append(toks)
    return {
        L: {
            "n": len(v),
            "tokens_mean": int(statistics.mean(v)),
            "tokens_median": int(statistics.median(v)),
            "tokens_max": max(v),
        }
        for L, v in sorted(by_L.items())
    }
