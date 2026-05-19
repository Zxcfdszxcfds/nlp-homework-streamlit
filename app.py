
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import random
import os
from transformers import pipeline
from typing import List, Tuple

# ----------------------------- 页面配置 -----------------------------
st.set_page_config(
    page_title="情感分析与舆情监测平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义CSS
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.1rem; font-weight: 500; }
    .stButton button { border-radius: 20px; transition: all 0.2s; }
    .stButton button:hover { transform: scale(1.02); }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ----------------------------- 模型加载 -----------------------------
@st.cache_resource(show_spinner="正在加载情感分析模型，首次运行需下载…")
def load_sentiment_pipeline():
    if not os.getenv("HF_ENDPOINT"):
        st.info("💡 提示：若下载缓慢或失败，可在终端运行: $env:HF_ENDPOINT = 'https://hf-mirror.com' (Windows)")
    try:
        model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        return pipeline("sentiment-analysis", model=model_name, truncation=True)
    except Exception as e:
        st.error(f"模型加载失败: {e}\n\n请检查网络连接，或设置镜像源后重启。")
        return None


sentiment_pipe = load_sentiment_pipeline()
MODEL_READY = sentiment_pipe is not None


def analyze_text(text: str) -> Tuple[str, float]:
    if not text or not MODEL_READY:
        return "无输入", 0.0
    result = sentiment_pipe(text)[0]
    label = result['label'].upper()
    label_map = {"POSITIVE": "积极", "NEGATIVE": "消极", "NEUTRAL": "中性"}
    return label_map.get(label, "中性"), result['score']


def batch_analyze(texts: List[str]) -> pd.DataFrame:
    if not MODEL_READY:
        return pd.DataFrame()
    results = sentiment_pipe(texts, truncation=True)
    rows = []
    for text, res in zip(texts, results):
        raw = res['label'].upper()
        label_cn = {"POSITIVE": "积极", "NEGATIVE": "消极", "NEUTRAL": "中性"}.get(raw, "中性")
        rows.append({"评论内容": text, "情感极性": label_cn, "置信度": round(res['score'], 4)})
    return pd.DataFrame(rows)


def generate_mock_comments(num_samples: int = 15) -> List[str]:
    pool = [
        "这个手机屏幕太清晰了，色彩鲜艳，非常满意！",
        "物流超级快，隔天就到了，包装完好，好评！",
        "音质出色，低音浑厚，听歌简直是一种享受。",
        "操作流畅，系统优化很好，没有任何卡顿。",
        "客服态度热情，问题解决及时，大赞！",
        "质量太差了，用了三天就坏掉，垃圾产品。",
        "续航严重缩水，充满电用不到半天，差评！",
        "拍照效果模糊不清，还不如千元机。",
        "售后态度恶劣，不给解决问题，非常失望。",
        "屏幕有坏点，申请换货被拒绝，再也不买这个品牌。",
        "充电两小时，游戏十分钟。",
        "在太阳底下根本看不清屏幕上的字。",
        "打开十个应用就开始卡顿，频繁闪退。",
        "充满电放一晚上，第二天早上只剩30%。",
        "手机握在手里边框割手，而且很重。",
        "价格适中，性能一般，日常使用勉强可以。",
        "外观设计普通，电池续航中规中矩。",
        "送来的手机没有贴膜，需要自己购买。",
        "耳机接口和充电口共用一个，不太方便。"
    ]
    return random.sample(pool, min(num_samples, len(pool)))


def plot_gauge(confidence: float, title: str = "置信度"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': title, 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 40}, 'valueformat': '.1f'},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "royalblue", 'thickness': 0.3},
            'steps': [
                {'range': [0, 0.5], 'color': '#f8d7da'},
                {'range': [0.5, 0.75], 'color': '#fff3cd'},
                {'range': [0.75, 1], 'color': '#d4edda'}
            ],
            'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': confidence}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ----------------------------- 初始化 session state（无key冲突） -----------------------------
if "explicit_input" not in st.session_state:
    st.session_state.explicit_input = "这屏幕画质太垃圾了，颗粒感明显！"
if "implicit_input" not in st.session_state:
    st.session_state.implicit_input = "在太阳底下根本看不清屏幕上的字"
if "single_text" not in st.session_state:
    st.session_state.single_text = "这款手机电池续航超强，重度使用一整天没问题！"

# ----------------------------- 标签页 -----------------------------
tab1, tab2, tab3 = st.tabs(["🔍 基础情感分类", "💡 显式情感 vs 隐式情感", "📈 舆情挖掘仪表盘"])

# ========================= 模块1 =========================
with tab1:
    st.markdown("### 🧠 单条评论细粒度情感分析")
    col_input, col_gauge = st.columns([3, 2])
    with col_input:
        # 无key，只用value绑定session_state
        single_text = st.text_area("✍️ 评论内容", value=st.session_state.single_text, height=120,
                                   key="single_text_area")
        st.session_state.single_text = single_text  # 手动同步
        analyze_btn = st.button("🔎 分析情感", use_container_width=True)
    if single_text and MODEL_READY:
        label, conf = analyze_text(single_text)
        with col_gauge:
            st.subheader(f"📌 情感极性：**{label}**")
            st.plotly_chart(plot_gauge(conf), use_container_width=True)
        st.info("💡 **工程启示**：置信度反映模型判断的确定性，低于0.6可标记待人工复核。")
    elif not MODEL_READY:
        st.error("模型未加载，请检查网络后重启。")

# ========================= 模块2（完全无冲突版）=========================
with tab2:
    st.markdown("### 📖 显式与隐式情感识别实验")
    st.markdown(
        "> **显式情感**：直接使用褒贬词（如“太棒了”、“垃圾”）。\n> **隐式情感**：客观事实暗含态度（如“玩游戏半小时就没电”）。")

    col_exp, col_imp = st.columns(2)

    # 显式情感输入区
    with col_exp:
        st.subheader("🗣️ 显式情感评价")
        # 不使用key，只绑定value，并通过后续按钮修改session_state
        explicit_text = st.text_area("包含明显褒贬词的评价", value=st.session_state.explicit_input, height=100,
                                     key="explicit_text_area")
        st.session_state.explicit_input = explicit_text
        if st.button("📎 示例显式评价", use_container_width=True):
            st.session_state.explicit_input = "这屏幕画质太垃圾了，颗粒感明显！"
            st.rerun()
        if explicit_text and MODEL_READY:
            label_exp, conf_exp = analyze_text(explicit_text)
            st.metric("模型判断", label_exp, delta=f"置信度 {conf_exp:.2%}")

    # 隐式情感输入区
    with col_imp:
        st.subheader("🌫️ 隐式客观描述")
        implicit_text = st.text_area("无明显情感词但暗含态度", value=st.session_state.implicit_input, height=100,
                                     key="implicit_text_area")
        st.session_state.implicit_input = implicit_text
        if st.button("📎 示例隐式评价", use_container_width=True):
            st.session_state.implicit_input = "在太阳底下根本看不清屏幕上的字"
            st.rerun()
        if implicit_text and MODEL_READY:
            label_imp, conf_imp = analyze_text(implicit_text)
            st.metric("模型判断", label_imp, delta=f"置信度 {conf_imp:.2%}")
            if "看不清" in implicit_text and label_imp != "消极":
                st.warning("⚠️ 隐式负面表达可能被误判，这是当前情感分析的主要挑战之一。")

    st.divider()
    st.info("📌 **实验观察**：显式情感通常被高置信度正确识别；隐式情感因缺乏情感词易被误判，需结合上下文或方面级分析。")

# ========================= 模块3 =========================
with tab3:
    st.markdown("### 📊 批量舆情监测与可视化看板")
    if "batch_df" not in st.session_state:
        st.session_state.batch_df = None

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        generate_click = st.button("🔄 生成测试舆情数据 (15条)", type="primary", use_container_width=True)

    if generate_click:
        with st.spinner("分析中..."):
            comments = generate_mock_comments(15)
            df_result = batch_analyze(comments)
            st.session_state.batch_df = df_result

    if st.session_state.batch_df is not None and not st.session_state.batch_df.empty:
        df = st.session_state.batch_df
        counts = df["情感极性"].value_counts()
        total = len(df)
        pos, neu, neg = counts.get("积极", 0), counts.get("中性", 0), counts.get("消极", 0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📝 总评论数", total)
        c2.metric("😊 积极", f"{pos} 条", delta=f"{pos / total:.0%}" if total else "0%")
        c3.metric("😐 中性", f"{neu} 条", delta=f"{neu / total:.0%}" if total else "0%")
        c4.metric("😞 消极", f"{neg} 条", delta=f"{neg / total:.0%}" if total else "0%")

        fig_pie = px.pie(names=counts.index, values=counts.values, title="整体情感分布",
                         color=counts.index,
                         color_discrete_map={"积极": "#2ecc71", "中性": "#f39c12", "消极": "#e74c3c"},
                         hole=0.3)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("📋 查看详细分析结果"):
            st.dataframe(df, use_container_width=True, height=400)

        if neg > total * 0.3:
            st.error("⚠️ **舆情预警**：消极评价占比超30%，建议立即排查。")
        elif pos > total * 0.6:
            st.success("✅ **积极信号**：正面口碑主导，可加大宣传。")
        else:
            st.info("🔍 **中性态势**：建议人工复核低置信度评论。")

        fig_hist = px.histogram(df, x="置信度", color="情感极性", nbins=20,
                                color_discrete_map={"积极": "#2ecc71", "中性": "#f39c12", "消极": "#e74c3c"})
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    elif MODEL_READY:
        st.info("👆 点击上方按钮，生成模拟评论并执行批量情感分析。")
    else:
        st.error("模型未就绪，无法进行批量分析。")

st.divider()
st.caption("🎓 情感分析与舆情监测平台 | 模型：distilbert-base-multilingual-cased-sentiments-student")
