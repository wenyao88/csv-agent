import streamlit as st
import pandas as pd
from agent import run_agent
from tools import execute_python

st.set_page_config(page_title="CSV智能分析助手", layout="wide")
st.title("CSV 智能分析助手")

if "history" not in st.session_state:
    st.session_state.history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "messages_display" not in st.session_state:
    st.session_state.messages_display = []


def auto_profile(df: pd.DataFrame):
    st.subheader("数据概览")
    col1, col2, col3 = st.columns(3)
    col1.metric("总行数", df.shape[0])
    col2.metric("总列数", df.shape[1])
    col3.metric("缺失值", int(df.isnull().sum().sum()))

    with st.expander("字段详情"):
        info = pd.DataFrame({
            "字段": df.columns,
            "类型": df.dtypes.values,
            "缺失数": df.isnull().sum().values,
            "缺失率": (df.isnull().sum().values / len(df) * 100).round(1)
        })
        st.dataframe(info, use_container_width=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        with st.expander("数值列统计"):
            st.dataframe(df[num_cols].describe().round(2), use_container_width=True)


with st.sidebar:
    st.header("上传数据")
    uploaded_file = st.file_uploader("选择CSV文件", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if st.session_state.df is None or uploaded_file.name != st.session_state.get("last_file"):
                st.session_state.df = df
                st.session_state["last_file"] = uploaded_file.name
                st.session_state.history = []
                st.session_state.messages_display = []
            st.success(f"已加载：{df.shape[0]}行 × {df.shape[1]}列")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"读取失败：{e}")

    if st.button("清空对话"):
        st.session_state.history = []
        st.session_state.messages_display = []
        st.rerun()

if st.session_state.df is None:
    st.info("请先在左侧上传CSV文件")
else:
    auto_profile(st.session_state.df)
    st.divider()

    for item in st.session_state.messages_display:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("steps"):
                with st.expander(f"🔍 查看分析过程（共{len(item['steps'])}步）"):
                    for step in item["steps"]:
                        st.markdown(f"**第{step['iteration']}步 - 执行代码：**")
                        st.code(step["code"], language="python")
                        st.markdown(f"**执行结果：**")
                        st.text(step["output"])
            if item.get("fig") is not None:
                st.plotly_chart(item["fig"], use_container_width=True)

    if prompt := st.chat_input("问关于数据的任何问题..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("分析中..."):
                answer, fig, steps = run_agent(prompt, st.session_state.df, st.session_state.history)
            st.markdown(answer)
            if steps:
                with st.expander(f"🔍 查看分析过程（共{len(steps)}步）"):
                    for step in steps:
                        st.markdown(f"**第{step['iteration']}步 - 执行代码：**")
                        st.code(step["code"], language="python")
                        st.markdown(f"**执行结果：**")
                        st.text(step["output"])
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": answer})
        st.session_state.messages_display.append({
            "role": "user", "content": prompt
        })
        st.session_state.messages_display.append({
            "role": "assistant", "content": answer, "steps": steps, "fig": fig
        })