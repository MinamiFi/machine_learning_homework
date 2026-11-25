"""JGA 口径校验用的 prompt 与答案匹配。

为什么单独一个模块：0.5B 的抽取能力对 prompt 极度敏感，实测三种问法差
2.6 倍（20% / 52% / 36%）。把这些模板集中放在一处，换 backbone 时只改这里，
而且报告里能逐字引用问了什么。

核心结论（`exp/docs/L2_JGA校验.md` §2）：**填空式**远好于问答式。
原因不是"填空更简单"，而是问答式的问句里自带数字（"for how many people"），
0.5B 会把问句本身回显出来；填空式把待填位置和问句分离，模型只需补词。
"""

from __future__ import annotations

import re

__all__ = ["SLOT_BLANK", "SYMBOLIC", "render_context", "build_prompt",
           "normalize", "answer_matches", "askable_slots"]

# 每个槽位的填空模板。占位符 ___ 处应填答案。
# 模板本身**不能含数字**，否则模型回显（实测 restaurant-people 全错就是这个原因）。
SLOT_BLANK: dict[str, str] = {
    "restaurant-name": "The name of the restaurant is ___.",
    "restaurant-area": "The restaurant is located in the ___ area.",
    "restaurant-food": "The restaurant serves ___ food.",
    "restaurant-pricerange": "The restaurant is in the ___ price range.",
    "restaurant-people": "The restaurant table is booked for ___ people.",
    "restaurant-day": "The restaurant booking is for ___.",
    "restaurant-time": "The restaurant booking is at ___.",
    "hotel-name": "The name of the hotel is ___.",
    "hotel-area": "The hotel is located in the ___ area.",
    "hotel-pricerange": "The hotel is in the ___ price range.",
    "hotel-type": "The accommodation type is ___.",
    "hotel-stars": "The hotel has ___ stars.",
    "hotel-parking": "Parking at the hotel: ___.",
    "hotel-internet": "Internet at the hotel: ___.",
    "hotel-people": "The hotel room is booked for ___ people.",
    "hotel-day": "The hotel stay starts on ___.",
    "hotel-stay": "The hotel stay lasts ___ nights.",
    "train-destination": "The train goes to ___.",
    "train-departure": "The train departs from ___.",
    "train-day": "The train travels on ___.",
    "train-arriveby": "The train should arrive by ___.",
    "train-leaveat": "The train should leave at ___.",
    "train-people": "The train tickets are for ___ people.",
    "attraction-name": "The name of the attraction is ___.",
    "attraction-area": "The attraction is located in the ___ area.",
    "attraction-type": "The attraction type is ___.",
    "taxi-departure": "The taxi picks up from ___.",
    "taxi-destination": "The taxi goes to ___.",
    "taxi-leaveat": "The taxi should leave at ___.",
    "taxi-arriveby": "The taxi should arrive by ___.",
}

# 符号值：原文里不会字面出现（"yes" 对应 "free wifi"），无法用文本匹配校验，
# 也无法用填空抽取验证。必须排除，否则会把"测不了"算成"答错"。
SYMBOLIC = {"yes", "no", "dontcare", "none", ""}

_TEMPLATE = (
    "Context:\n{ctx}\n\n"
    "Read the context and fill in the blank. "
    "Reply with only the missing words.\n{blank}"
)


def askable_slots(gold: dict[str, str]) -> dict[str, str]:
    """能从 gold 里出题的槽位：有模板、值非符号、非空。"""
    return {
        k: v for k, v in gold.items()
        if k in SLOT_BLANK and str(v).strip().lower() not in SYMBOLIC
    }


def render_context(segs, max_turns: int | None = None) -> str:
    """把若干轮渲染成 prompt 里的上下文。

    只放 user/system 两句，不放 state/delta —— 那是标注，不是对话内容。
    """
    rows = []
    for s in (segs[:max_turns] if max_turns else segs):
        u = (s.user or "").strip()
        y = (s.system or "").strip()
        if u:
            rows.append(f"User: {u}")
        if y:
            rows.append(f"System: {y}")
    return "\n".join(rows) if rows else "(empty)"


def build_prompt(ctx: str, slot: str) -> str:
    return _TEMPLATE.format(ctx=ctx, blank=SLOT_BLANK[slot])


_PUNCT = re.compile(r"[^\w\s:]")


def normalize(s: str) -> str:
    """答案归一化：小写、去撇号、去标点、折叠空白、英式拼写统一、去开头冠词。

    三个细节都是实测踩出来的：
      1. 撇号必须**直接删掉**而不是换成空格。`Queen's College` 若变成
         `queen s college`，就和 gold 的 `queens college` 对不上，
         造成假阴性（实测出现过多例）。
      2. 只去掉**开头**的冠词。原来的 `.replace("the ", " ")` 会把
         `theatre` 之类词中间的部分也改掉，且会误伤句中冠词。
      3. `centre→center` 统一，MultiWOZ 两种拼写都有。
    """
    s = str(s).lower().strip().replace("centre", "center")
    s = s.replace("'", "").replace("\u2019", "")
    s = _PUNCT.sub(" ", s)
    s = " ".join(s.split())
    return re.sub(r"^(the|a|an) ", "", s)


def answer_matches(gold: str, pred: str) -> bool:
    """判对：gold 作为完整词出现在 pred 里。

    整词匹配是必须的——`4` 不能命中 `14`，`3` 不能命中 `13:00`。
    这也是覆盖率脚本里踩过的同一个坑。
    """
    g, p = normalize(gold), normalize(pred)
    if not g or not p:
        return False
    if g.isdigit():
        # 数字：要求独立出现，不能是更长数字的一部分
        return re.search(rf"(?<!\d){re.escape(g)}(?!\d)", p) is not None
    return re.search(rf"(?<!\w){re.escape(g)}(?!\w)", p) is not None
