# -*- coding: utf-8 -*-
"""LangGraph 智能预问诊与分诊状态机 + RAG（Chroma + BGE-small-zh）"""
import os
import re
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
try:
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as _Saver
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from llm_utils import chat, extract_json, DISCLAIMER

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medical_docs")
COLLECTION = "medical_kb"


_COLLECTION_CACHE = None

def get_collection():
    global _COLLECTION_CACHE
    if _COLLECTION_CACHE is None:
        client = chromadb.PersistentClient(path=KB_DIR)
        ef = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
        _COLLECTION_CACHE = client.get_or_create_collection(
            COLLECTION, embedding_function=ef,
            metadata={"hnsw:space": "cosine"})
    return _COLLECTION_CACHE


def build_kb(docs_dir: str = DOCS_DIR):
    import glob
    client = chromadb.PersistentClient(path=KB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    ef = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
    col = client.get_or_create_collection(COLLECTION, embedding_function=ef,
                                          metadata={"hnsw:space": "cosine"})
    files = glob.glob(os.path.join(docs_dir, "*.txt"))
    if not files:
        raise FileNotFoundError(f"{docs_dir} 下没有 .txt 知识文档")
    ids, docs, metas = [], [], []
    n = 0
    for fp in files:
        text = open(fp, encoding="utf-8").read()
        title = os.path.splitext(os.path.basename(fp))[0]
        step, size = 200, 300
        for i in range(0, max(1, len(text) - size + 1), step):
            ids.append(f"{title}-{n}")
            docs.append(text[i:i + size])
            metas.append({"source": title})
            n += 1
        if len(text) <= size:
            ids.append(f"{title}-{n}"); docs.append(text); metas.append({"source": title}); n += 1
    col.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def retrieve(query: str, k: int = 4) -> list:
    col = get_collection()
    if col.count() == 0:
        build_kb()
    res = col.query(query_texts=[query], n_results=k)
    return res["documents"][0] if res["documents"] else []


REQUIRED_FIELDS = ["age", "symptoms", "duration"]

SYS_INTAKE = (
    "从用户话语中提取：age(年龄,数字或null)、gender(男/女/未知)、"
    "symptoms(症状描述)、duration(持续时间)、history(既往史)、"
    "lab_report(化验单信息或null)。"
    '只输出JSON：{"age":null,"gender":"未知","symptoms":"","duration":"","history":null,"lab_report":null}'
)

SYS_TRIAGE = """根据患者信息与检索知识，输出JSON：
{"department":"科室","urgency":"门诊/急诊/观察","reason":"理由(1-2句)","advice":["建议1","建议2"]}
高热不退/剧烈胸痛/呼吸困难/意识模糊/大出血 → "急诊"，否则"门诊"。
只输出JSON，不加任何解释。"""


class TriageState(TypedDict):
    messages: Annotated[list, add_messages]
    patient: dict
    intake_complete: bool
    retrieved_docs: list
    triage_result: dict | None
    finished: bool
    api_key: str | None


def _last_user_text(state) -> str:
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human":
            return m.content
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content", "")
    return ""


def _regex_extract(text: str) -> dict:
    """本地正则抽取年龄、性别、持续时间（不调用 LLM，几乎瞬间完成）"""
    out = {}
    m = re.search(r"(\d{1,3})\s*岁", text)
    if m:
        try:
            out["age"] = int(m.group(1))
        except Exception:
            pass
    if "男" in text and "女" not in text:
        out["gender"] = "男"
    elif "女" in text:
        out["gender"] = "女"
    m = re.search(r"(\d+\s*(?:小时|天|日|周|星期|个月|月|年))", text)
    if m:
        out["duration"] = m.group(1).replace(" ", "")
    return out


def _looks_like_symptom(text: str) -> bool:
    """判断是否像症状描述，过滤掉"男"、"45岁"、"无"、"没有"这类纯回答"""
    t = (text or "").strip()
    if not t:
        return False
    stripped = re.sub(r"\d+\s*(岁|公斤|kg|cm|厘米|米|斤)", "", t)
    stripped = re.sub(r"[，,。.、\s]+", "", stripped)
    if len(stripped) < 2:
        return False
    if stripped in {"男", "女", "未知", "无", "没有", "否", "没了"}:
        return False
    if any(stripped.startswith(x) for x in ("无", "没有", "不", "否")):
        return False
    return True


def intake_node(state: TriageState) -> dict:
    user_text = _last_user_text(state)
    info = _regex_extract(user_text)
    patient = {**(state.get("patient") or {})}
    for k, v in info.items():
        if v:
            patient[k] = v
    if _looks_like_symptom(user_text):
        if not patient.get("symptoms"):
            patient["symptoms"] = user_text
        elif user_text not in patient["symptoms"]:
            patient["symptoms"] = patient["symptoms"] + "；" + user_text

    missing = [f for f in REQUIRED_FIELDS if not patient.get(f)]
    if missing:
        ask_map = {
            "age": "请问您的年龄是多少岁？",
            "symptoms": "请详细描述您的主要症状？",
            "duration": "这些症状持续多长时间了？",
        }
        reply = "为了分诊，我还需要了解：" + "；".join(ask_map[f] for f in missing) + "。"
        return {"patient": patient, "intake_complete": False,
                "messages": [("assistant", reply)]}

    if not patient.get("_history_asked"):
        patient["_history_asked"] = True
        return {"patient": patient, "intake_complete": False,
                "messages": [("assistant", "请问您有既往病史、药物过敏史吗？（没有可回复：无）")]}

    return {"patient": patient, "intake_complete": True, "messages": []}


def route_after_intake(state: TriageState) -> str:
    return "retrieve" if state.get("intake_complete") else END


def retrieve_node(state: TriageState) -> dict:
    p = state["patient"]
    query = f"{p.get('symptoms','')} {p.get('history','')} 相关疾病与就诊建议"
    docs = retrieve(query, k=4)
    return {"retrieved_docs": docs}


def triage_node(state: TriageState) -> dict:
    p = state["patient"]
    kb = "\n---\n".join(state.get("retrieved_docs") or []) or "（知识库无相关内容）"
    prompt = (SYS_TRIAGE + f"\n\n患者信息：{p}\n\n检索知识：\n{kb}")
    try:
        out = chat([{"role": "system", "content": prompt}],
                   api_key=state.get("api_key"), force_json=True)
        result = extract_json(out)
    except Exception as e:
        import traceback
        err_detail = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        try:
            import streamlit as st
            st.error(f"[triage 失败] {err_detail}")
        except Exception:
            print(f"[triage] 解析失败: {err_detail}")
        sym = p.get("symptoms", "")
        if any(k in sym for k in ["鼻塞", "流涕", "咽痛", "喷嚏"]):
            dept, urg = "耳鼻喉科/儿科", "门诊"
        elif any(k in sym for k in ["胸痛", "胸闷", "心悸"]):
            dept, urg = "心内科", "急诊" if "出汗" in sym else "门诊"
        elif any(k in sym for k in ["腹痛", "腹泻", "呕吐"]):
            dept, urg = "消化内科", "门诊"
        elif any(k in sym for k in ["发热", "发烧", "咳嗽"]):
            dept, urg = "呼吸内科/发热门诊", "门诊"
        else:
            dept, urg = "全科门诊", "门诊"
        result = {"department": dept, "urgency": urg,
                  "reason": "基于症状关键词的初步分诊建议。",
                  "advice": ["携带既往病历", "如有加重及时就医"]}

    reply = (
        f"🏥 **分诊结果**\n\n"
        f"- 推荐科室：**{result.get('department','未知')}**\n"
        f"- 紧急程度：**{result.get('urgency','门诊')}**\n"
        f"- 分诊理由：{result.get('reason','')}\n"
        f"- 就医前建议：\n" +
        "\n".join(f"  {i+1}. {a}" for i, a in enumerate(result.get("advice", [])))
        + f"\n\n{DISCLAIMER}"
    )
    return {"triage_result": result, "finished": True,
            "messages": [("assistant", reply)]}


def build_graph():
    g = StateGraph(TriageState)
    g.add_node("intake", intake_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("triage", triage_node)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake, {"retrieve": "retrieve", END: END})
    g.add_edge("retrieve", "triage")
    g.add_edge("triage", END)
    return g.compile(checkpointer=_Saver())


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        print(f"知识库已构建，共 {build_kb()} 个文本块")
