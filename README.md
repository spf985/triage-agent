# 智能预问诊与分诊 AI Agent

> ⚠️ 免责声明：本AI不能替代专业医生，仅供健康参考。

## 1. 安装依赖
```bat
cd /d D:\triage_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 配置 API Key
```bat
setx ZHIPU_API_KEY "你的智谱APIKey"
```
（或启动后在 Streamlit 侧边栏直接填写）

## 3. 构建知识库（首次运行）
```bat
python triage_agent.py build
```

## 4. 运行
```bat
streamlit run app.py
```

## 目录结构
```
D:\triage_project
├─ app.py              # Streamlit 前端
├─ triage_agent.py     # LangGraph 状态机 + RAG
├─ llm_utils.py        # 智谱 GLM-4.6V-Flash 封装
├─ requirements.txt
├─ medical_docs\       # 本地医疗知识库文本（放 .txt）
│  ├─ 感冒与流感.txt
│  ├─ 消化系统疾病.txt
│  └─ 心血管疾病.txt
└─ chroma_db\          # 自动生成的向量库
```
