"""src 包根。

模块依赖是单向的：memory → scorer → agent → eval。
eval 可以只导入前三个模块跑 L1，不必加载 LLM。
"""

__version__ = "0.1.0"
