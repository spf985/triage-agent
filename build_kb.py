# -*- coding: utf-8 -*-
"""构建知识库（等价于 python triage_agent.py build）"""
from triage_agent import build_kb
print(f"知识库已构建，共 {build_kb()} 个文本块")
