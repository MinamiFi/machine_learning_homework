#!/usr/bin/env python3
"""环境自检：句向量 → 打分网训练链路 → LLM 生成，三段全跑一遍。

    source exp/env.sh
    python exp/scripts/smoke_test.py            # 全跑
    python exp/scripts/smoke_test.py --skip-llm # 只测前两段（省内存）

三段各自验证一件事：
  1. embedder  模型能加载、输出 384 维、语义相近的句子余弦相似度更高
  2. scorer    2 层 MLP + BCE + AdamW 这条训练链路能收敛（合成数据，验管线不验任务）
  3. llm       本地 backbone 能按 chat template 生成，且内存不超 4GB

注意：第 2 段用的是合成数据，只证明"训练代码跑得通"，不代表真实标签可用。
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
MODELS = EXP_ROOT / "models"

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} {detail}")


def peak_mb() -> float:
    """当前进程峰值 RSS（MB）。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cgroup_limit_gb() -> float | None:
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(p).read_text().strip()
            if raw and raw != "max":
                return int(raw) / 1e9
        except OSError:
            continue
    return None


def banner(cfg: dict) -> None:
    print("=" * 72)
    print("环境自检")
    print("=" * 72)
    print(f"  python            {sys.version.split()[0]}")
    print(f"  torch             {torch.__version__}   cuda={torch.cuda.is_available()}")
    print(f"  threads           {torch.get_num_threads()}")
    print(f"  cgroup 内存上限    {cgroup_limit_gb() or '未设置'}")
    print(f"  models dir        {MODELS}")
    print(f"  embedder          {cfg['models']['embedder']}")
    print(f"  llm               {cfg['models']['llm']}")
    print()


# ---------------------------------------------------------------- 1. embedder
SENTENCES = [
    "I need a cheap hotel in the north of town.",
    "Looking for a budget place to stay in the north.",
    "Can you book a table for four at an Italian restaurant?",
    "I would like a table for four, Italian food please.",
    "What is the phone number of the hotel?",
    "Could you tell me the postcode of that guest house?",
]


def test_embedder(cfg: dict) -> None:
    from sentence_transformers import SentenceTransformer

    path = EXP_ROOT / cfg["models"]["embedder"]
    if not path.exists():
        check("embedder 模型存在", False, f"缺 {path}")
        return
    check("embedder 模型存在", True, f"{path.name}")

    t0 = time.time()
    model = SentenceTransformer(str(path), device="cpu")
    load_s = time.time() - t0

    t0 = time.time()
    emb = model.encode(SENTENCES, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    enc_s = time.time() - t0

    dim = int(emb.shape[1])
    check("输出维度 == 384", dim == cfg["models"]["embed_dim"], f"dim={dim}")
    check("向量已归一化", bool(np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)))

    # 同义句对内相似度应高于跨主题对
    sim = emb @ emb.T
    intra = (sim[0, 1] + sim[2, 3] + sim[4, 5]) / 3
    inter = (sim[0, 2] + sim[0, 4] + sim[2, 5]) / 3
    check("同义句相似度 > 跨主题", bool(intra > inter),
          f"intra={intra:.3f} inter={inter:.3f}")
    check("编码速度", True, f"load {load_s:.1f}s, {len(SENTENCES)} 句 {enc_s:.2f}s")
    print()


# ------------------------------------------------------------------ 2. scorer
def build_mlp(cfg: dict) -> torch.nn.Module:
    m = cfg["model"]
    layers: list[torch.nn.Module] = [torch.nn.LayerNorm(m["input_dim"])]
    dims = [m["input_dim"]] + [m["hidden_dim"]] * m["num_layers"] + [1]
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [torch.nn.GELU(), torch.nn.Dropout(m["dropout"])]
    return torch.nn.Sequential(*layers)


def test_scorer(cfg: dict) -> None:
    from sklearn.metrics import roc_auc_score

    m, t = cfg["model"], cfg["train"]
    torch.manual_seed(t["seed"])

    # 合成数据：固定线性 teacher，标签可完全学会；只验训练链路
    n, d = 4096, m["input_dim"]
    X = torch.randn(n, d)
    w = torch.randn(d)
    y = (X @ w > 0).float().unsqueeze(1)
    Xtr, ytr = X[:3072], y[:3072]
    Xva, yva = X[3072:], y[3072:]

    net = build_mlp(cfg)
    n_params = sum(p.numel() for p in net.parameters())
    check("MLP 参数量", True, f"{n_params:,}")

    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)])
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=t["lr"],
                            weight_decay=t["weight_decay"])

    bs, losses = t["batch_size"], []
    t0 = time.time()
    for epoch in range(8):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            losses.append(loss.item())
    train_s = time.time() - t0

    net.eval()
    with torch.no_grad():
        auc = roc_auc_score(yva.numpy(), torch.sigmoid(net(Xva)).numpy())

    check("loss 下降", bool(losses[-1] < losses[0]),
          f"{losses[0]:.4f} → {losses[-1]:.4f}")
    check("合成任务 AUC > 0.95", bool(auc > 0.95), f"AUC={auc:.4f}")
    check("训练速度", True, f"{len(losses)} step / {train_s:.1f}s")
    print()


# --------------------------------------------------------------------- 3. llm
MEMORIES = [
    "[M1] user: I am looking for a hotel in the north. system: there are 5 hotels, "
    "including acorn guest house, a cheap B&B.",
    "[M2] user: I want a cheap option. system: acorn guest house is in the north "
    "and is cheap.",
    "[M3] user: What is the phone number? system: 01223 353535.",
]
QUERY = "我刚才说的那家北边的便宜酒店，叫什么名字？"


def build_prompt() -> list[dict]:
    sys_prompt = (
        "你是任务型对话助手。下面是从长期记忆中检索到的片段，"
        "可能包含无关、过时或互相矛盾的内容。只使用与当前问题相关且仍然有效的信息回答，"
        "信息不足时直接说不知道，不要编造。"
    )
    mem_block = "\n".join(MEMORIES)
    user = f"【记忆片段】\n{mem_block}\n\n【当前问题】\n{QUERY}"
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]


def test_llm(cfg: dict) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = EXP_ROOT / cfg["models"]["llm"]
    if not path.exists():
        check("llm 模型存在", False, f"缺 {path}")
        return
    check("llm 模型存在", True, f"{path.name}")

    tok = AutoTokenizer.from_pretrained(str(path))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 0
    load_s = time.time() - t0
    check("llm 加载", True, f"{load_s:.1f}s, 峰值 {peak_mb():.0f} MB")

    text = tok.apply_chat_template(build_prompt(), tokenize=False,
                                   add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    n_in = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen_s = time.time() - t0

    n_new = out.shape[1] - n_in
    reply = tok.decode(out[0, n_in:], skip_special_tokens=True).strip()
    tps = n_new / gen_s if gen_s > 0 else 0.0

    check("能生成非空回复", bool(reply), f"{n_new} tok / {gen_s:.1f}s = {tps:.1f} tok/s")
    check("内存 < 4GB", peak_mb() < 4000, f"峰值 {peak_mb():.0f} MB")

    print("\n  ---- 模型输出 ----")
    for line in reply.splitlines():
        print(f"  | {line}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true", help="不加载 LLM（省内存）")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP_ROOT / "configs" / "env.yaml").read_text("utf-8"))
    scorer_cfg = yaml.safe_load((EXP_ROOT / "configs" / "scorer.yaml").read_text("utf-8"))
    torch.set_num_threads(cfg["runtime"]["num_threads"])

    banner(cfg)
    print("[1/3] 句向量模型")
    test_embedder(cfg)
    print("[2/3] 打分网络训练链路（合成数据）")
    test_scorer(scorer_cfg)
    if args.skip_llm:
        print("[3/3] LLM  —— 已跳过（--skip-llm）\n")
    else:
        print("[3/3] 对话 backbone")
        test_llm(cfg)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 72)
    print(f"结果：{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过"
          f"，峰值内存 {peak_mb():.0f} MB")
    if failed:
        print("失败项：" + "；".join(failed))
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
