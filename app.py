from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RAGAgent = None
agent_import_error: str | None = None

try:
    from rag_agent.agent import RAGAgent as ImportedRAGAgent

    RAGAgent = ImportedRAGAgent
except Exception as e:
    # 兜底：直接从文件加载，避免 import 路径问题
    agent_path = SRC_DIR / "rag_agent" / "agent.py"
    if agent_path.exists():
        try:
            from importlib import util

            spec = util.spec_from_file_location("rag_agent.agent", str(agent_path))
            module = util.module_from_spec(spec)
            sys.modules["rag_agent.agent"] = module
            spec.loader.exec_module(module)
            RAGAgent = getattr(module, "RAGAgent", None)
            if RAGAgent is None:
                agent_import_error = "`RAGAgent` class not found in agent.py"
            else:
                agent_import_error = None
        except Exception as load_error:
            RAGAgent = None
            agent_import_error = str(load_error)
    else:
        agent_import_error = str(e)

st.set_page_config(page_title="Hybrid RAG Agent", page_icon="📚")
st.title("Hybrid RAG Agent")
st.caption("LangChain + LangGraph + FAISS：本地知识库混合检索问答")

st.sidebar.markdown("### 设置")

if "agent" not in st.session_state:
    with st.spinner("正在初始化 RAGAgent..."):
        if RAGAgent is None:
            st.session_state.agent = None
            st.error(f"初始化失败: {agent_import_error}")
        else:
            try:
                st.session_state.agent = RAGAgent()
                st.success("RAGAgent 已初始化")
            except Exception as e:
                st.session_state.agent = None
                st.error(f"初始化失败: {e}")

agent = st.session_state.get("agent")
if agent is not None:
    info = agent.retriever.describe()
    st.sidebar.success(f"知识库已加载：{info['chunks']} 个文本块")
    st.sidebar.caption(
        f"Embedding：{info['embedding_model']}\n\n"
        f"混合权重：dense {info['dense_weight']} / lexical {info['lexical_weight']}"
    )

query = st.text_input("请输入问题：", value="示例问题：如何使用 RAG Agent?")
if st.button("问答"):
    if not query.strip():
        st.warning("请先输入问题。")
    else:
        if agent is None:
            st.error("RAGAgent 未能初始化，无法回答。")
        else:
            with st.spinner("正在检索并生成回答..."):
                try:
                    result = agent.answer_with_sources(query)
                    st.markdown("**回答：**")
                    st.write(result["answer"])
                    sources = result.get("sources", [])
                    if sources:
                        with st.expander(f"查看引用来源（{len(sources)}）"):
                            for source in sources:
                                section = source.get("section") or "未命名章节"
                                st.markdown(
                                    f"- `{source.get('source')}` · {section} "
                                    f"（混合分数：{source.get('hybrid_score')}）"
                                )
                except Exception as e:
                    st.error(f"生成回答时出错: {e}")
