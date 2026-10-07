# -*- coding: utf-8 -*-
"""智能预问诊与分诊 AI Agent —— Streamlit 前端"""
import json
import os
import tempfile
import streamlit as st
from langchain_core.messages import HumanMessage

from triage_agent import build_graph, retrieve
from llm_utils import parse_lab_image, DISCLAIMER, MODEL_CHAT

st.set_page_config(page_title="智能预问诊与分诊AI", page_icon="🏥", layout="wide")
st.title("🏥 智能预问诊与分诊 AI Agent")
st.caption(f"模型：智谱 {MODEL_CHAT} | 知识库：BGE-small-zh + Chroma")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "thread" not in st.session_state:
    st.session_state.thread = {"configurable": {"thread_id": "demo"}}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lab_json" not in st.session_state:
    st.session_state.lab_json = None


def _run_agent(user_msg: str, api_key: str | None):
    """统一调用 graph，返回AI回复文本，并写入消息历史"""
    st.session_state.messages.append({"role": "user", "content": user_msg})
    resp = st.session_state.graph.invoke(
        {"messages": [HumanMessage(content=user_msg)],
         "api_key": api_key or None},
        config=st.session_state.thread,
    )
    reply = ""
    for m in resp["messages"]:
        if getattr(m, "type", "") == "ai" and m.content:
            reply = m.content
    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
    return reply


# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("智谱 API Key（留空用环境变量 ZHIPU_API_KEY）",
                            type="password", value=os.getenv("ZHIPU_API_KEY", ""))
    st.divider()
    st.header("🧪 化验单解析（多模态）")
    up = st.file_uploader("上传化验单图片", type=["jpg", "jpeg", "png", "webp", "gif"])
    if up and st.button("🔍 解析化验单"):
        with st.spinner("GLM 视觉模型解析中…"):
            suffix = os.path.splitext(up.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tf.write(up.getbuffer())
                tmp = tf.name
            try:
                data = parse_lab_image(tmp, api_key or None)
                st.session_state.lab_json = data
            except Exception as e:
                st.error(f"解析失败：{e}")
            finally:
                os.unlink(tmp)
    if st.session_state.lab_json:
        st.success("✅ 已解析")
        st.json(st.session_state.lab_json)
        if st.button("➕ 将解析结果带入问诊"):
            abnormal = st.session_state.lab_json.get("abnormal_summary", "")
            items = st.session_state.lab_json.get("items", [])
            brief = "；".join(
                f"{i.get('name')}={i.get('value')}({i.get('flag')})"
                for i in items[:6]
            )
            msg = f"我的化验单显示：{brief}。异常总结：{abnormal}。请结合化验单帮我分诊。"
            # 去重：最后一条不是这句话才注入
            last_user = ""
            for m in reversed(st.session_state.messages):
                if m["role"] == "user":
                    last_user = m["content"]; break
            if last_user != msg:
                with st.spinner("AI 正在分析化验单并分诊…"):
                    _run_agent(msg, api_key)
            st.session_state.lab_json = None
            st.rerun()
    st.divider()
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.session_state.lab_json = None
        st.rerun()
    st.markdown(f"> {DISCLAIMER}")

# ---------------- 聊天界面 ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("请描述您的症状（如：28岁，发烧咳嗽两天，38.5度）"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("预问诊/分诊中…"):
            reply = _run_agent(prompt, api_key)
        if reply:
            st.markdown(reply)
