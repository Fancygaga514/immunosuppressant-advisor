"""
免疫抑制剂智能用药指导系统 - 主应用文件

使用 Streamlit 构建，包含以下核心功能：
1. 个体化剂量推荐
2. 血药浓度解读
3. 药物相互作用检查
4. 知识库问答（RAG）

Author: Immunosuppressant Advisor Team
"""

import os
import time
from typing import List, Dict, Optional

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

import utils
import prompts


PERSIST_DIR = "./chroma_db"

st.set_page_config(
    page_title="免疫抑制剂智能用药指导系统",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* 全局样式 */
    body {
        background-color: #f0f4f8;
    }
    
    /* 免责声明横幅 - 更醒目 */
    .disclaimer-banner {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: linear-gradient(90deg, #dc2626 0%, #ef4444 50%, #dc2626 100%);
        color: white;
        text-align: center;
        padding: 12px;
        font-size: 15px;
        font-weight: bold;
        z-index: 9999;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4);
        border-bottom: 3px solid #991b1b;
    }
    
    .disclaimer-banner::before {
        content: "⚠️ ";
        font-size: 18px;
    }

    /* 主内容区域 */
    .main-content {
        padding-top: 60px;
    }

    /* 医院标题样式 */
    .hospital-title {
        font-size: 32px;
        font-weight: bold;
        color: #1e40af;
        text-align: center;
        margin-bottom: 8px;
        letter-spacing: 2px;
    }

    .system-title {
        font-size: 20px;
        color: #4b5563;
        text-align: center;
        margin-bottom: 24px;
    }

    /* 卡片样式 */
    .card {
        background-color: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
    }

    .card-header {
        font-size: 18px;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 2px solid #dbeafe;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dbeafe 0%, #bfdbfe 50%, #dbeafe 100%);
    }

    section[data-testid="stSidebar"] .stTitle {
        color: #1e40af;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #374151;
    }

    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background-color: transparent;
        padding: 8px 0;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 500;
        background-color: #e0e7ff;
        border-radius: 12px 12px 0 0;
        color: #3b82f6;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #c7d2fe;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        color: #1e40af;
        box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.05);
    }

    /* 按钮样式 */
    .stButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    .stButton button:disabled {
        background: #9ca3af;
        cursor: not-allowed;
    }

    /* 输入框样式 */
    .stNumberInput input, .stSelectbox select {
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 8px 12px;
        transition: all 0.3s ease;
    }

    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    /* 多选框样式 */
    .stMultiSelect [data-baseweb="select"] {
        border: 2px solid #e5e7eb;
        border-radius: 8px;
    }

    .stMultiSelect [data-baseweb="select"]:focus {
        border-color: #3b82f6;
    }

    /* 警告框样式 */
    .danger-alert {
        padding: 20px;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 5px solid #dc2626;
        border-radius: 8px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.2);
    }

    .danger-alert strong {
        color: #dc2626;
        font-size: 16px;
    }

    /* 成功/信息提示 */
    .stSuccess, .stInfo {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-radius: 8px;
        border-left: 4px solid #10b981;
    }

    .stWarning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
    }

    /* 指标卡片 */
    .stMetric {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #bfdbfe;
    }

    .stMetric label {
        color: #64748b;
        font-size: 14px;
    }

    .stMetric div[data-testid="stMetricValue"] {
        color: #1e40af;
        font-size: 24px;
        font-weight: bold;
    }

    /* 聊天消息 */
    .stChatMessage {
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .stChatMessage[data-role="user"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 16px 16px 4px 16px;
    }

    .stChatMessage[data-role="assistant"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px 16px 16px 4px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    本系统仅为 AI 辅助决策参考，不可替代专业药师判断！
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_vector_db():
    """
    获取或初始化向量数据库
    使用缓存避免重复加载
    """
    if not os.path.exists(PERSIST_DIR):
        return None

    try:
        api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return None

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )

        vector_db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )
        return vector_db
    except Exception as e:
        st.error(f"加载向量数据库失败: {e}")
        return None


def get_llm_model():
    """
    获取 LLM 模型实例
    使用 DeepSeek API
    """
    api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None

    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.3,
            max_tokens=2000,
        )
        return llm
    except Exception as e:
        st.error(f"初始化 LLM 模型失败: {e}")
        return None


def query_knowledge_base(question: str, vector_db: Chroma, k: int = 5) -> str:
    """
    从知识库检索相关内容

    参数:
        question: 用户问题
        vector_db: 向量数据库
        k: 返回的相关文档数量

    返回:
        检索到的上下文内容
    """
    if not vector_db:
        return "知识库未初始化，请先运行 build_index.py 构建索引"

    try:
        docs = vector_db.similarity_search(question, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context if context else "未找到相关内容"
    except Exception as e:
        return f"检索失败: {e}"


def render_disclaimer():
    """渲染免责声明"""
    st.markdown("---")
    st.markdown(prompts.DISCLAIMER_TEXT)
    st.markdown("---")


def render_dose_recommendation_tab():
    """渲染剂量推荐标签页"""
    st.markdown("""
    <div class="card">
        <div class="card-header">
            💊 个体化剂量推荐
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("患者信息")
        
        age = st.number_input("年龄（岁）", min_value=18, max_value=100, value=45)
        weight = st.number_input("体重（kg）", min_value=30.0, max_value=200.0, value=65.0, step=0.1)
        creatinine_umol = st.number_input("血清肌酐（μmol/L）", min_value=1.0, max_value=1000.0, value=88.4, step=1.0)
        creatinine = creatinine_umol / 88.4  # 转换为 mg/dL
        is_female = st.checkbox("女性")
        is_black = st.checkbox("黑人（用于 eGFR 计算）")

        st.subheader("治疗信息")
        drug_options = ["他克莫司", "环孢素", "霉酚酸酯", "西罗莫司", "依维莫司"]
        selected_drug = st.selectbox("选择免疫抑制剂", drug_options)

        transplant_options = ["肾移植", "肝移植", "心脏移植"]
        selected_transplant = st.selectbox("移植类型", transplant_options)

        st.subheader("联用药物")
        concomitant_options = [
            "五酯胶囊/软胶囊",
            "氟康唑", "利福平", "酮康唑",
            "红霉素", "克拉霉素", "奥美拉唑", "地尔硫卓"
        ]
        selected_concomitant = st.multiselect(
            "选择联用药物（如有）",
            options=concomitant_options,
            default=[],
            placeholder="可选多个药物..."
        )

        calculate_btn = st.button("计算推荐剂量", type="primary", use_container_width=True)

    with col2:
        st.subheader("计算结果")

        if calculate_btn:
            with st.spinner("计算中..."):
                time.sleep(0.5)

                egfr = utils.calculate_egfr(age, creatinine, is_female, is_black)
                egfr_stage, stage_desc = utils.get_egfr_stage(egfr)

                dose_result = utils.calculate_initial_dose(
                    selected_drug, weight, egfr, selected_transplant, selected_concomitant
                )

                if "error" in dose_result:
                    st.error(dose_result["error"])
                else:
                    st.success("✅ 计算完成")

                    st.metric("估算 eGFR", f"{egfr} mL/min/1.73m²")
                    st.caption(f"CKD 分期: {egfr_stage} - {stage_desc}")

                    st.divider()

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("推荐起始剂量", f"{dose_result['calculated_dose']} {dose_result['dose_unit']}")
                    with col_b:
                        st.metric("按体重", f"{dose_result['dose_per_kg']} {dose_result['dose_unit_per_kg']}")

                    target = dose_result["target_concentration"]
                    if target:
                        st.info(
                            f"目标血药浓度: {target['min']}-{target['max']} {target['unit']} "
                            f"（{selected_transplant}）"
                        )

                    st.caption(f"📝 {dose_result['adjustment_note']}")
                    st.caption(f"💡 {dose_result['therapeutic_range_note']}")

                    validation = utils.validate_llm_dose_response(
                        selected_drug,
                        dose_result["calculated_dose"],
                        weight
                    )

                    if validation.get("within_range"):
                        st.success("✅ 剂量在推荐范围内")
                    elif validation.get("warning"):
                        st.warning(validation["warning"])

                    st.divider()

                    llm = get_llm_model()
                    if llm:
                        with st.spinner("正在生成详细建议..."):
                            system_prompt = prompts.get_system_prompt("dose_recommendation")
                            user_prompt = prompts.build_user_prompt(
                                "dose_recommendation",
                                age=age,
                                gender="女性" if is_female else "男性",
                                weight=weight,
                                creatinine=creatinine,
                                egfr=egfr,
                                egfr_stage=f"{egfr_stage} ({stage_desc})",
                                transplant_type=selected_transplant,
                                drug_name=selected_drug,
                                calculated_dose=dose_result["calculated_dose"],
                                adjustment_note=dose_result["adjustment_note"],
                                target_min=target.get("min", "N/A"),
                                target_max=target.get("max", "N/A"),
                                unit=target.get("unit", "ng/mL"),
                            )

                            try:
                                response = llm.invoke(
                                    f"System: {system_prompt}\n\nUser: {user_prompt}"
                                )
                                st.markdown("### AI 详细建议")
                                st.markdown(response.content)
                            except Exception as e:
                                st.error(f"生成建议失败: {e}")
                    else:
                        st.warning("请配置 DeepSeek API Key 以获取 AI 详细建议")

    st.markdown("</div>", unsafe_allow_html=True)
    render_disclaimer()


def render_concentration_interpretation_tab():
    """渲染血药浓度解读标签页"""
    st.markdown("""
    <div class="card">
        <div class="card-header">
            🩸 血药浓度解读
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("检测信息")

        drug_options = ["他克莫司", "环孢素", "霉酚酸酯", "西罗莫司", "依维莫司"]
        selected_drug = st.selectbox("药物名称", drug_options, key="conc_drug")

        transplant_options = ["肾移植", "肝移植", "心脏移植"]
        selected_transplant = st.selectbox("移植类型", transplant_options, key="conc_transplant")

        concentration = st.number_input(
            "实测血药浓度",
            min_value=0.0,
            max_value=1000.0,
            value=10.0,
            step=0.1,
            key="conc_value"
        )

        sampling_time = st.selectbox(
            "采样时间",
            ["谷浓度", "峰浓度", "随机浓度"],
            key="sampling_time"
        )

        drug_info = utils.get_drug_info(selected_drug)
        target_range = drug_info.get("target_concentration", {}).get(selected_transplant, {})
        target_max = target_range.get("max", 100)
        target_unit = target_range.get("unit", "ng/mL")

        danger_threshold = target_max * 2
        if concentration > danger_threshold:
            st.markdown(f"""
            <div class="danger-alert">
                <strong>🚨 危险警告：浓度严重超标！</strong><br/>
                实测浓度 <strong>{concentration} {target_unit}</strong> 超过安全阈值<br/>
                目标上限的 {round(concentration/target_max, 1)} 倍，可能导致严重毒性反应！<br/>
                <strong>建议立即联系主治医师！</strong>
            </div>
            """, unsafe_allow_html=True)
        elif concentration > target_max:
            st.warning(f"⚠️ 浓度偏高：{concentration} {target_unit}，超过目标上限 {target_max} {target_unit}")

        interpret_btn = st.button("解读浓度", type="primary", use_container_width=True)

    with col2:
        st.subheader("解读结果")

        if interpret_btn:
            with st.spinner("解读中..."):
                time.sleep(0.5)

                result = utils.interpret_concentration(
                    selected_drug,
                    concentration,
                    selected_transplant,
                    sampling_time
                )

                if "error" in result:
                    st.error(result["error"])
                else:
                    status_colors = {
                        "偏低": "🔵",
                        "偏高": "🔴",
                        "达标": "🟢"
                    }
                    status_icon = status_colors.get(result["status"], "⚪")

                    st.markdown(f"### {status_icon} 浓度状态: **{result['status']}**")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("实测浓度", f"{result['measured_concentration']} {result['unit']}")
                    with col_b:
                        st.metric("目标范围", f"{result['target_min']}-{result['target_max']} {result['unit']}")

                    st.info(f"**建议采取的行动**: {result['action']}")
                    st.markdown(f"**详细建议**: {result['recommendation']}")

                    st.divider()

                    llm = get_llm_model()
                    if llm:
                        with st.spinner("正在生成详细分析..."):
                            system_prompt = prompts.get_system_prompt("concentration_interpretation")
                            user_prompt = prompts.build_user_prompt(
                                "concentration_interpretation",
                                drug_name=selected_drug,
                                measured_concentration=concentration,
                                unit=result["unit"],
                                sampling_time=sampling_time,
                                transplant_type=selected_transplant,
                                target_min=result["target_min"],
                                target_max=result["target_max"],
                                status=result["status"],
                                action=result["action"],
                            )

                            try:
                                response = llm.invoke(
                                    f"System: {system_prompt}\n\nUser: {user_prompt}"
                                )
                                st.markdown("### AI 详细分析")
                                st.markdown(response.content)
                            except Exception as e:
                                st.error(f"生成分析失败: {e}")
                    else:
                        st.warning("请配置 DeepSeek API Key 以获取 AI 详细分析")

    st.markdown("</div>", unsafe_allow_html=True)
    render_disclaimer()


def render_drug_interaction_tab():
    """渲染药物相互作用标签页"""
    st.markdown("""
    <div class="card">
        <div class="card-header">
            💉 药物相互作用检查
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("药物选择")

        drug_options = ["他克莫司", "环孢素", "霉酚酸酯", "西罗莫司", "依维莫司"]
        selected_drug = st.selectbox("主要免疫抑制剂", drug_options)

        st.markdown("**选择联用药物（可多选）**")

        interaction_drugs = [
            "氟康唑", "伏立康唑", "伊曲康唑", "艾沙康唑",
            "奈玛特韦/利托那韦", "利福平", "五酯胶囊/软胶囊", "西柚汁"
        ]

        selected_drugs = st.multiselect(
            "搜索并选择联用药物",
            options=interaction_drugs,
            default=[],
            placeholder="输入药物名称搜索...",
            label_visibility="collapsed"
        )

        check_btn = st.button(
            f"检查与 {selected_drug} 的相互作用",
            type="primary",
            use_container_width=True,
            disabled=len(selected_drugs) == 0
        )

    with col2:
        st.subheader("相互作用分析结果")

        if check_btn:
            with st.spinner("分析中..."):
                time.sleep(0.5)

                results = utils.check_drug_interactions(selected_drug, selected_drugs)

                severity_order = {"严重": 0, "中等": 1, "轻微": 2, "未发现": 3}

                sorted_results = sorted(
                    results,
                    key=lambda x: severity_order.get(x["severity"], 4)
                )

                for i, result in enumerate(sorted_results):
                    severity = result["severity"]
                    severity_emoji = {
                        "严重": "🚨",
                        "中等": "⚠️",
                        "轻微": "ℹ️",
                        "未发现": "✅"
                    }.get(severity, "❓")

                    with st.container():
                        st.markdown(f"#### {severity_emoji} {result['concomitant_drug']}")
                        st.markdown(f"**严重程度**: {severity}")
                        st.markdown(f"**作用机制**: {result['effect']}")
                        st.markdown(f"**临床建议**: {result['recommendation']}")
                        st.markdown("---")

                st.divider()

                llm = get_llm_model()
                if llm:
                    with st.spinner("正在生成综合建议..."):
                        system_prompt = prompts.get_system_prompt("drug_interaction")

                        interaction_results_str = "\n".join([
                            f"- {r['concomitant_drug']}: {r['severity']} - {r['recommendation']}"
                            for r in sorted_results
                        ])

                        user_prompt = prompts.build_user_prompt(
                            "drug_interaction",
                            immunosuppressant=selected_drug,
                            concomitant_drugs=", ".join(selected_drugs),
                            interaction_results=interaction_results_str,
                        )

                        try:
                            response = llm.invoke(
                                f"System: {system_prompt}\n\nUser: {user_prompt}"
                            )
                            st.markdown("### AI 综合用药建议")
                            st.markdown(response.content)
                        except Exception as e:
                            st.error(f"生成建议失败: {e}")
                else:
                    st.warning("请配置 DeepSeek API Key 以获取 AI 综合建议")

    st.markdown("</div>", unsafe_allow_html=True)
    render_disclaimer()


def render_qa_tab():
    """渲染知识库问答标签页"""
    st.markdown("""
    <div class="card">
        <div class="card-header">
            📚 免疫抑制剂知识库问答
        </div>
    """, unsafe_allow_html=True)

    vector_db = get_vector_db()

    if not vector_db:
        st.warning(
            "⚠️ 知识库未初始化。"
            "请先将 PDF 药品说明书放入 ./data/ 文件夹，然后运行 `python build_index.py` 构建索引。"
        )
        st.code("python build_index.py", language="bash")
        st.markdown("</div>", unsafe_allow_html=True)
        render_disclaimer()
        return

    st.success("✅ 知识库已就绪，可以开始提问")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                context = query_knowledge_base(prompt, vector_db, k=5)

                llm = get_llm_model()
                if llm:
                    system_prompt = prompts.get_system_prompt("knowledge_base")
                    user_prompt = prompts.build_user_prompt(
                        "knowledge_base",
                        context=context,
                        question=prompt
                    )

                    try:
                        response = llm.invoke(
                            f"System: {system_prompt}\n\nUser: {user_prompt}"
                        )
                        full_response = response.content
                    except Exception as e:
                        full_response = f"抱歉，发生了错误: {e}"
                else:
                    full_response = (
                        "抱歉，API 未配置。请在 .streamlit/secrets.toml 中配置 DEEPSEEK_API_KEY。"
                    )

                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    if st.button("清除对话历史"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    render_disclaimer()


def main():
    """主函数"""

    with st.sidebar:
        st.title("💊 用药指导系统")
        st.markdown("---")
        st.markdown("### 关于本系统")
        st.markdown("""
        免疫抑制剂智能用药指导系统

        **功能模块:**
        - 个体化剂量推荐
        - 血药浓度解读
        - 药物相互作用检查
        - 知识库问答
        """)
        st.markdown("---")
        st.markdown("### 使用说明")
        st.markdown("""
        1. 在左侧输入患者信息
        2. 选择药物和移植类型
        3. 点击相应按钮获取建议
        4. 所有结果仅供参考
        """)
        st.markdown("---")

        api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if api_key and api_key != "your-deepseek-api-key-here":
            st.success("✅ API 已配置")
        else:
            st.error("⚠️ 请配置 API Key")

        st.markdown("---")
        st.markdown("**版本**: 1.0.0")

    # 医院标题
    st.markdown('<div class="hospital-title">福州大学附属省立医院</div>', unsafe_allow_html=True)
    st.markdown('<div class="system-title">免疫抑制剂智能用药指导系统</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "💊 剂量推荐",
        "🩸 浓度解读",
        "💉 相互作用",
        "📚 知识问答"
    ])

    with tab1:
        render_dose_recommendation_tab()

    with tab2:
        render_concentration_interpretation_tab()

    with tab3:
        render_drug_interaction_tab()

    with tab4:
        render_qa_tab()


if __name__ == "__main__":
    main()
