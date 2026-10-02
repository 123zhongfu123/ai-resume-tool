import streamlit as st
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_deepseek import ChatDeepSeek

# 注意：这里我们不再需要 load_dotenv() 了，因为 Key 不再从 .env 读
# load_dotenv()   ← 删掉这行

st.set_page_config(page_title="AI简历修改器", layout="centered")
st.title("📝 AI简历修改器")
st.caption("输入你的简历草稿，AI帮你改到满意为止")

# ========== 新增：在页面最上方，让用户输入自己的 Key ==========
# 用 password 类型，输入的内容会显示为黑点，保护隐私
# 如果用户之前填过，就从 session_state 里把值读出来（防止刷新页面丢失）
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

user_key = st.text_input(
    "请输入你的 DeepSeek API Key（不会泄露，仅本页面使用）：",
    type="password",
    value=st.session_state.user_api_key
)

# 如果用户输入了 Key，就更新到 session_state
if user_key:
    st.session_state.user_api_key = user_key

# 如果没有填 Key，就显示提示，并停止运行下面的逻辑
if not st.session_state.user_api_key:
    st.info("👆 请先在上面填入你的 DeepSeek API Key，然后就能开始改简历了。")
    st.stop()  # st.stop() 会阻止下面的代码执行


# ========== 新增结束 ==========

# ========== 黑板定义（没变） ==========
class State(TypedDict):
    resume: str
    feedback: str
    improved: str
    decision: str


# ========== 关键改动：LLM 不再一开始就建好，而是用用户输入的 Key 动态创建 ==========
# 注意：这里我们不用 @st.cache_resource 了，因为 Key 是用户输入的，每次都不一样

def get_llm(user_key: str):
    """根据用户输入的 Key，返回一个 LLM 对象"""
    return ChatDeepSeek(model="deepseek-chat", api_key=user_key)


# ========== 节点定义（稍作修改，传入 llm） ==========
def hr_review(state: State, llm) -> dict:
    current = state.get("improved") or state["resume"]
    prompt = f"你是一个严格的HR，请指出以下简历的3个缺点，简短一点。\n简历：{current}"
    response = llm.invoke(prompt)
    return {"feedback": response.content}


def rewrite(state: State, llm) -> dict:
    prompt = f"""你是一个求职者。请根据HR的意见，修改你的简历。
原简历：{state['resume']}
HR意见：{state['feedback']}
请直接输出修改后的简历。"""
    response = llm.invoke(prompt)
    return {"improved": response.content, "decision": ""}


# ========== 画图部分，改成动态构建 ==========
# 注意：因为 llm 是用用户的 Key 建的，所以每次都需要重新 build 图
# 我们用 st.cache_resource 缓存图的结构，但 llm 在运行时注入

@st.cache_resource
def build_graph():
    """只构建图的结构，不绑定具体的 LLM"""
    graph = StateGraph(State)

    # 用 lambda 包装一下，让节点能拿到 llm（llm 在运行时通过闭包传递）
    graph.add_node("hr_review", lambda state: hr_review(state, get_llm(st.session_state.user_api_key)))
    graph.add_node("rewrite", lambda state: rewrite(state, get_llm(st.session_state.user_api_key)))

    graph.add_edge(START, "hr_review")
    graph.add_edge("hr_review", "rewrite")

    def route_after_rewrite(state: State) -> str:
        if state.get("decision") == "rewrite":
            return "hr_review"
        return END

    graph.add_conditional_edges("rewrite", route_after_rewrite, {
        "hr_review": "hr_review",
        END: END
    })

    memory = InMemorySaver()
    return graph.compile(checkpointer=memory, interrupt_after=["rewrite"])


# 获取图
app = build_graph()

# ========== 下面的 Streamlit 界面逻辑（基本没变） ==========
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "streamlit-user-001"}}
if "started" not in st.session_state:
    st.session_state.started = False
if "paused" not in st.session_state:
    st.session_state.paused = False

if not st.session_state.started:
    user_resume = st.text_area("在这里粘贴你的简历草稿：", height=200,
                               placeholder="例如：我叫小王，会写Python，想找个工作……")
    if st.button("开始修改 🚀"):
        if user_resume.strip():
            st.session_state.started = True
            app.invoke(
                {"resume": user_resume, "feedback": "", "improved": "", "decision": ""},
                st.session_state.config
            )
            st.session_state.paused = True
            st.rerun()
        else:
            st.warning("请先输入简历内容！")

elif st.session_state.paused:
    state = app.get_state(st.session_state.config)
    st.subheader("修改后的简历：")
    st.markdown(state.values["improved"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 满意，结束"):
            app.update_state(st.session_state.config, {"decision": "ok"})
            app.invoke(None, st.session_state.config)
            st.session_state.paused = False
            st.rerun()
    with col2:
        if st.button("🔄 不满意，重新改"):
            app.update_state(st.session_state.config, {"decision": "rewrite"})
            app.invoke(None, st.session_state.config)
            st.rerun()

else:
    st.balloons()
    st.success("搞定！最终简历如下：")
    state = app.get_state(st.session_state.config)
    st.markdown(state.values["improved"])
    if st.button("再改一份 🔁"):
        st.session_state.started = False
        st.session_state.paused = False
        st.rerun()