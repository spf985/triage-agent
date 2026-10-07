# -*- coding: utf-8 -*-
"""智谱 GLM 多模态 API 封装（OpenAI 兼容接口）"""
import base64
import json
import os
import re
from openai import OpenAI

API_KEY = os.getenv("ZHIPU_API_KEY", "")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MODEL_CHAT = "glm-4-flash"       # 文本任务：快、稳、JSON友好
MODEL_VISION = "glm-4.6v-flash"  # 视觉任务

DISCLAIMER = "⚠️ 免责声明：本AI不能替代专业医生，仅供健康参考；如有急症请立即拨打120或前往医院急诊。"


def get_client(api_key: str | None = None) -> OpenAI:
    key = api_key or API_KEY
    if not key:
        raise ValueError("请设置环境变量 ZHIPU_API_KEY 或在侧边栏填入 API Key")
    return OpenAI(api_key=key, base_url=BASE_URL)


def chat(messages: list, api_key: str | None = None, temperature: float = 0.3,
         force_json: bool = False) -> str:
    """普通对话/结构化输出。force_json=True 时强制返回JSON"""
    client = get_client(api_key)
    kwargs = {"model": MODEL_CHAT, "messages": messages, "temperature": temperature,
              "max_tokens": 400}
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def extract_json(text: str) -> dict:
    """更健壮的JSON提取，容忍各种markdown和多余文字"""
    if not text:
        raise ValueError("模型返回空内容")
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    raise ValueError(f"未找到有效JSON: {text[:300]}")


def parse_lab_image(image_path: str, api_key: str | None = None) -> dict:
    """上传化验单图片 -> 解析为结构化 JSON"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(image_path)[1].lower().lstrip(".") or "png"
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/png")
    client = get_client(api_key)
    prompt = (
        "你是医学检验报告解析助手。请读取这张化验单图片，输出严格JSON（不要输出任何其他文字），格式：\n"
        '{\n  "report_type": "检验报告类型",\n  "items": ['
        '{\"name\": \"项目名称\", \"value\": \"检测结果\", '
        '\"unit\": \"单位\", \"reference_range\": \"参考范围\", '
        '\"flag\": \"异常标记:高/低/正常/未知\"}],\n'
        '  "abnormal_summary": "异常项目一句话总结",\n'
        '  "patient_hint": "患者可见的基本信息"\n}'
    )
    resp = client.chat.completions.create(
        model=MODEL_VISION,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
        temperature=0.0,
    )
    return extract_json(resp.choices[0].message.content)
