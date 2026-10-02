# ========== 1. 导入工具 ==========
import streamlit as st
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_deepseek import ChatDeepSeek

# 注意：这次不需要 load_dotenv()，因为 Key 由用户在网页上输入
# load_dotenv()

# 网页配置
st.set_page_config(page_title="AI求职助理", layout="centered")
st.title("🤖 AI求职助理")
st.caption("输入你的需求，主管AI自动派活给合适的员工")


# ========== 2. 黑板（State） ==========
class State(TypedDict):
    messages: Annotated[list, operator.add]


# ========== 3. 三个员工节点 ==========
def resume_expert(state: State, llm) -> dict:
    prompt = f"你是一个简历专家。根据用户的对话历史，帮他优化简历。\n对话历史：\n{state['messages']}"
    response = llm.invoke(prompt)
    return {"messages": [{"role": "assistant", "content": f"[简历专家] {response.content}"}]}


def cover_letter_expert(state: State, llm) -> dict:
    prompt = f"你是一个求职信专家。根据用户的对话历史，帮他写一封求职信。\n对话历史：\n{state['messages']}"
    response = llm.invoke(prompt)
    return {"messages": [{"role": "assistant", "content": f"[求职信专家] {response.content}"}]}


def interviewer(state: State, llm) -> dict:
    prompt = f"你是一个面试官。根据用户的对话历史，给他出3道面试题。\n对话历史：\n{state['messages']}"
    response = llm.invoke(prompt)
    return {"messages": [{"role": "assistant", "content": f"[面试官] {response.content}"}]}


# ========== 4. 主管节点 ==========
def supervisor(state: State, llm) -> dict:
    prompt = f"""你是一个主管。根据下面的对话历史，决定下一步派谁去干活。
你只能从以下四个选项中选一个，不要选别的：
- resume_expert（简历专家）
- cover_letter_expert（求职信专家）
- interviewer（面试官）
- FINISH（任务已完成，结束）

对话历史：
{state['messages']}

请只输出一个单词（选项名）："""
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    return {"messages": [{"role": "assistant", "content": f"[主管决定] {decision}"}]}


# ========== 5. 构建图（用用户输入的 Key） ==========

@st.cache_resource
def build_graph():
    """构建图的结构，LLM 在运行时由 get_llm 提供"""
    graph = StateGraph(State)

    # 用 lambda 把 llm 延迟传入节点
    graph.add_node("supervisor", lambda state: supervisor(state, st.session_state.llm))
    graph.add_node("resume_expert", lambda state: resume_expert(state, st.session_state.llm))
    graph.add_node("cover_letter_expert", lambda state: cover_letter_expert(state, st.session_state.llm))
    graph.add_node("interviewer", lambda state: interviewer(state, st.session_state.llm))

    graph.add_edge(START, "supervisor")
    graph.add_edge("resume_expert", "supervisor")
    graph.add_edge("cover_letter_expert", "supervisor")
    graph.add_edge("interviewer", "supervisor")

    def route_supervisor(state: State) -> str:
        last_msg = state["messages"][-1]["content"]
        if "resume_expert" in last_msg:
            return "resume_expert"
        elif "cover_letter_expert" in last_msg:
            return "cover_letter_expert"
        elif "interviewer" in last_msg:
            return "interviewer"
        else:
            return END

    graph.add_conditional_edges("supervisor", route_supervisor, {
        "resume_expert": "resume_expert",
        "cover_letter_expert": "cover_letter_expert",
        "interviewer": "interviewer",
        END: END
    })

    return graph.compile()


# ========== 6. 网页界面 ==========

# 初始化 session_state
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""
if "llm" not in st.session_state:
    st.session_state.llm = None
if "history" not in st.session_state:
    st.session_state.history = []  # 存储所有对话消息

# --- 顶部：输入 API Key ---
user_key = st.text_input(
    "请输入你的 DeepSeek API Key（仅本页面使用）：",
    type="password",
    value=st.session_state.user_api_key
)

if user_key and user_key != st.session_state.user_api_key:
    st.session_state.user_api_key = user_key
    # 用新 Key 初始化 LLM
    st.session_state.llm = ChatDeepSeek(model="deepseek-chat", api_key=user_key)

if not st.session_state.llm:
    st.info("👆 请先填入你的 DeepSeek API Key，然后就能开始使用。")
    st.stop()

# --- 中间：显示对话历史 ---
st.markdown("### 💬 对话记录")
if not st.session_state.history:
    st.caption("还没有对话，请在下方输入你的需求。")
else:
    for msg in st.session_state.history:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f"**🧑 你：** {content}")
        elif "[主管决定]" in content:
            st.markdown(f"**👔 主管：** {content.replace('[主管决定]', '').strip()}")
        elif "[简历专家]" in content:
            st.markdown(f"**📄 简历专家：** {content.replace('[简历专家]', '').strip()}")
        elif "[求职信专家]" in content:
            st.markdown(f"**✉️ 求职信专家：** {content.replace('[求职信专家]', '').strip()}")
        elif "[面试官]" in content:
            st.markdown(f"**🎤 面试官：** {content.replace('[面试官]', '').strip()}")
        st.markdown("---")

# --- 底部：输入框和发送按钮 ---
user_input = st.text_input("请输入你的需求：", placeholder="比如：帮我写一封求职信，突出我的Python能力")

if st.button("发送 🚀"):
    if user_input.strip():
        # 1. 把用户输入加到历史里
        st.session_state.history.append({"role": "user", "content": user_input})

        # 2. 调图，跑流程
        app = build_graph()
        result = app.invoke({"messages": st.session_state.history})

        # 3. 用图返回的完整消息列表，替换历史（保留最全的）
        st.session_state.history = result["messages"]

        # 4. 刷新页面
        st.rerun()
    else:
        st.warning("请输入内容！")

# --- 底部：清空对话按钮 ---
if st.session_state.history:
    if st.button("🗑️ 清空对话"):
        st.session_state.history = []
        st.rerun()