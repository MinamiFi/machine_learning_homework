#!/usr/bin/env python3
"""用 GitHub Git Data API 推送本地提交（绕过不稳定的 git-upload-pack 传输层）。

github.com:443 的 git 传输在此环境时通时断（HTTP2 framing / GnuTLS recv / 连接超时），
但 api.github.com 稳定。因此逐提交：blob → tree(base_tree) → commit → PATCH ref。

关键：提交 sha 必须与本地逐位一致，所以
  1) message 用 `git cat-file commit` 取原始字节再 split(b"\\n\\n", 1)，
     不能用 `git log --format=%B`（末尾会补一个 \\n，sha 就变了）
  2) author/committer 的 name/email/date 原样传给 API

用法：GH_PAT=... python3 push_api.py [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWNER, REPO = "MinamiFi", "machine_learning_homework"
BRANCH = "main"
API = "https://api.github.com"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def git_text(*args: str) -> str:
    return git(*args).decode("utf-8").strip()


def call(method: str, path: str, payload: dict | None = None) -> dict:
    token = os.environ.get("GH_PAT")
    if not token:
        sys.exit("!! 需要 GH_PAT 环境变量")
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "push-api",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"!! {method} {path} → {e.code}\n{e.read().decode()[:800]}")


def parse_commit(sha: str) -> dict:
    raw = git("cat-file", "commit", sha)
    head, _, message = raw.partition(b"\n\n")
    fields: dict[str, str] = {}
    for line in head.decode("utf-8").splitlines():
        if line.startswith(" "):
            continue
        k, _, v = line.partition(" ")
        fields.setdefault(k, v)
    parents = [ln.split(" ", 1)[1] for ln in head.decode("utf-8").splitlines() if ln.startswith("parent ")]
    ident = fields["author"]
    # "Name <email> 1732345678 +0800" → API 要 ISO 8601，不认 unix 时间戳
    name_email, _, ts = ident.rpartition("> ")
    name, _, email = name_email.partition(" <")
    stamp = _iso(ts.strip())
    return {
        "message": message.decode("utf-8"),
        "parents": parents,
        "author": {"name": name, "email": email, "date": stamp},
        "committer": {"name": name, "email": email, "date": stamp},
    }


def _iso(ts: str) -> str:
    """'1732345678 +0800' → '2026-11-25T09:39:49+08:00'。"""
    secs, _, off = ts.partition(" ")
    tz = f"{off[:3]}:{off[3:]}" if off else "+00:00"
    return datetime.fromtimestamp(int(secs), tz=timezone(timedelta(hours=int(off[:3])))).strftime(
        "%Y-%m-%dT%H:%M:%S"
    ) + tz


def blob_for(path: str) -> str:
    data = (ROOT / path).read_bytes()
    try:
        text = data.decode("utf-8")
        payload = {"content": text, "encoding": "utf-8"}
    except UnicodeDecodeError:
        payload = {"content": base64.b64encode(data).decode(), "encoding": "base64"}
    return call("POST", f"/repos/{OWNER}/{REPO}/git/blobs", payload)["sha"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    remote = call("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")["object"]["sha"]
    local = git_text("rev-parse", "HEAD")
    print(f"远端 {BRANCH}: {remote[:7]}   本地 HEAD: {local[:7]}")

    if remote == local:
        print("已一致，无需推送")
        return 0

    todo = git_text("rev-list", "--reverse", f"{remote}..{local}").split()
    if not todo:
        sys.exit(f"!! 本地 {local[:7]} 不领先远端 {remote[:7]}，可能已分叉")

    base_tree = call("GET", f"/repos/{OWNER}/{REPO}/git/commits/{remote}")["tree"]["sha"]

    for sha in todo:
        meta = parse_commit(sha)
        changed = [p for p in git("diff", "--name-only", "-z", f"{sha}^", sha).decode().split("\0") if p]
        print(f"\n→ {sha[:7]}  {meta['message'].splitlines()[0]}  ({len(changed)} 文件)")

        entries = []
        for p in changed:
            full = ROOT / p
            if full.exists():
                entries.append({"path": p, "mode": "100755" if os.access(full, os.X_OK) else "100644",
                                "type": "blob", "sha": blob_for(p)})
            else:
                entries.append({"path": p, "mode": "100644", "type": "blob", "sha": None})

        if a.dry_run:
            print("   dry-run，跳过写入")
            continue

        tree = call("POST", f"/repos/{OWNER}/{REPO}/git/trees",
                    {"base_tree": base_tree, "tree": entries})["sha"]
        commit = call("POST", f"/repos/{OWNER}/{REPO}/git/commits", {
            "message": meta["message"], "tree": tree, "parents": [remote],
            "author": meta["author"], "committer": meta["committer"],
        })
        print(f"   远端提交 {commit['sha'][:7]}  (本地 {sha[:7]})")
        if commit["sha"] != sha:
            print("   !! sha 不一致，停止（检查 message 末尾换行与 author 时间戳）")
            return 2
        remote = commit["sha"]
        base_tree = tree

    if not a.dry_run:
        call("PATCH", f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
             {"sha": remote, "force": False})
        git("update-ref", f"refs/remotes/origin/{BRANCH}", remote)
        print(f"\n推送完成：远端 {BRANCH} = {remote[:7]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
