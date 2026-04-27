"""
Performance Analytics — прототип ИС для оценки эффективности участников ИТ-проекта
Реализует гибридную многоуровневую модель HMLPE (P / R / V / S layers)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yaml
import io
import os
from pathlib import Path

# ── Путь к директории приложения ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ── Импорт вычислительного модуля ─────────────────────────────────────────────
import sys
sys.path.insert(0, str(BASE_DIR))
from hmlpe import load_config, load_data, calculate_scores, export_excel

# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация страницы
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Performance Analytics",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #161b27; border-right: 1px solid #2d3448; }
    .main-title {
        font-size: 2rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(135deg, #6c8eff 0%, #a78bfa 50%, #34d399 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #8892b0; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2538 0%, #252d42 100%);
        border: 1px solid #2d3448; border-radius: 12px;
        padding: 1.2rem 1.5rem; text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #6c8eff; }
    .metric-label { font-size: 0.75rem; color: #8892b0; text-transform: uppercase; letter-spacing: 0.05em; }
    .layer-badge {
        display: inline-block; padding: 0.15rem 0.6rem; border-radius: 99px;
        font-size: 0.75rem; font-weight: 700; margin-right: 0.3rem;
    }
    .badge-p { background: #1e40af; color: #93c5fd; }
    .badge-r { background: #065f46; color: #6ee7b7; }
    .badge-v { background: #78350f; color: #fcd34d; }
    .badge-s { background: #6b21a8; color: #d8b4fe; }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
        border-left: 3px solid #6c8eff; padding-left: 0.75rem;
        margin: 1.5rem 0 0.75rem 0;
    }
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stTabs [data-baseweb="tab"] { color: #8892b0; }
    .stTabs [aria-selected="true"] { color: #6c8eff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Боковая панель
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Performance Analytics")
    st.markdown("---")

    # Источник данных
    st.markdown("**Источник данных**")
    data_source = st.radio("", ["📂 Демо-данные", "⬆️ Загрузить файлы"], label_visibility="collapsed")

    metrics_df = None
    users_df = None
    config = None

    if data_source == "📂 Демо-данные":
        try:
            config = load_config(str(BASE_DIR / "config.yaml"))
            metrics_df, users_df = load_data(
                str(BASE_DIR / "data" / "metrics.csv"),
                str(BASE_DIR / "data" / "users.csv"),
            )
            st.success("✅ Демо-данные загружены")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
    else:
        col1, col2 = st.columns(2)
        m_file = st.file_uploader("metrics.csv", type="csv")
        u_file = st.file_uploader("users.csv",   type="csv")
        c_file = st.file_uploader("config.yaml", type=["yaml", "yml"])

        if m_file and u_file:
            metrics_df = pd.read_csv(m_file, parse_dates=["created_at", "closed_at"])
            users_df   = pd.read_csv(u_file)
            if c_file:
                config = yaml.safe_load(c_file)
            else:
                config = load_config(str(BASE_DIR / "config.yaml"))
            st.success("✅ Данные загружены")

    st.markdown("---")
    st.markdown("**Веса модели HMLPE**")

    if config:
        w = config.get("weights", {})
        alpha = st.slider("α — P-layer (процессы)",  0.0, 1.0, float(w.get("alpha", 0.30)), 0.05)
        beta  = st.slider("β — R-layer (роль)",      0.0, 1.0, float(w.get("beta",  0.25)), 0.05)
        gamma = st.slider("γ — V-layer (ценность)",  0.0, 1.0, float(w.get("gamma", 0.25)), 0.05)
        delta = st.slider("δ — S-layer (социум)",    0.0, 1.0, float(w.get("delta", 0.20)), 0.05)

        total = alpha + beta + gamma + delta
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Сумма весов = {total:.2f} (должна быть 1.0)")
        else:
            st.success(f"✓ Сумма весов = {total:.2f}")

        config["weights"] = {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}
    else:
        st.info("Загрузите данные для настройки весов")

    st.markdown("---")
    st.markdown("<small style='color:#4a5568'>Гибридная модель HMLPE<br>© 2026 Performance Analytics</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Заголовок
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">📊 Performance Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Гибридная многоуровневая оценка эффективности участников ИТ-проекта (HMLPE)</div>', unsafe_allow_html=True)

# Формула
st.markdown("""
<div style="background:#1e2538;border:1px solid #2d3448;border-radius:10px;padding:0.8rem 1.2rem;display:inline-block;margin-bottom:1rem;">
<span style="color:#8892b0;font-size:0.85rem;">Формула: </span>
<strong style="color:#e2e8f0;">E<sub>i</sub> = α·P<sub>i</sub> + β·R<sub>i</sub> + γ·V<sub>i</sub> + δ·S<sub>i</sub></strong>
<span style="color:#4a5568;font-size:0.8rem;margin-left:1rem;">где α+β+γ+δ = 1</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Основной контент
# ─────────────────────────────────────────────────────────────────────────────

if metrics_df is None or users_df is None or config is None:
    st.info("👈 Выберите источник данных в боковой панели.")
    st.stop()

# Расчёт
with st.spinner("Расчёт оценок HMLPE..."):
    df = calculate_scores(metrics_df, users_df, config)

if df.empty:
    st.error("Нет данных для расчёта. Проверьте соответствие логинов в metrics.csv и users.csv.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# KPI-карточки
# ─────────────────────────────────────────────────────────────────────────────

top_user  = df.iloc[0]
avg_score = df["E"].mean()
total_closed = int(df["closed_count"].sum())
total_prs    = int(df["pr_count"].sum())

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, label in [
    (c1, len(df), "Участников"),
    (c2, f"{avg_score:.1f}", "Средняя оценка E"),
    (c3, f"{top_user['E']:.1f}", f"Лидер: {top_user['login']}"),
    (c4, total_closed, "Закрыто задач"),
    (c5, total_prs, "Слито PR"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Вкладки
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Рейтинг", "📈 Аналитика слоёв", "👤 Профиль участника", "📋 Данные"])

# ── TAB 1: Рейтинг ───────────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-header">Итоговый рейтинг участников</div>', unsafe_allow_html=True)

        display_df = df[["login", "role", "level", "team", "P", "R", "V", "S", "E"]].copy()
        display_df.index.name = "Место"

        def color_e(val):
            if val >= 65: return "background-color:#14532d;color:#86efac"
            if val >= 45: return "background-color:#713f12;color:#fde68a"
            return "background-color:#7f1d1d;color:#fca5a5"
        
        def color_layer(val):
            intensity = int((val / 100) * 40)
            return f"background-color: rgba(99,149,255,0.{intensity:02d}); color: #e2e8f0"

        styled = display_df.style.map(color_e, subset=["E"]) \
            .format({"P": "{:.1f}", "R": "{:.1f}", "V": "{:.1f}", "S": "{:.1f}", "E": "{:.1f}"}) \
            .map(color_layer, subset=["P", "R", "V", "S"])
        st.dataframe(styled, use_container_width=True, height=350)

    with col_right:
        st.markdown('<div class="section-header">Оценки E по участникам</div>', unsafe_allow_html=True)

        colors = ["#6c8eff" if e >= 65 else "#fbbf24" if e >= 45 else "#f87171" for e in df["E"]]
        fig_bar = go.Figure(go.Bar(
            x=df["login"],
            y=df["E"],
            marker_color=colors,
            text=df["E"].apply(lambda x: f"{x:.1f}"),
            textposition="outside",
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892b0"),
            yaxis=dict(range=[0, 110], gridcolor="#2d3448"),
            xaxis=dict(gridcolor="#2d3448"),
            margin=dict(t=20, b=10, l=10, r=10),
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Команды
    st.markdown('<div class="section-header">Сравнение команд</div>', unsafe_allow_html=True)
    team_df = df.groupby("team")[["P", "R", "V", "S", "E"]].mean().round(1)

    fig_teams = go.Figure()
    colors_map = {"Alpha": "#6c8eff", "Beta": "#34d399"}
    for team in team_df.index:
        fig_teams.add_trace(go.Bar(
            name=team,
            x=["P-layer", "R-layer", "V-layer", "S-layer", "Итог E"],
            y=[team_df.loc[team, "P"], team_df.loc[team, "R"],
               team_df.loc[team, "V"], team_df.loc[team, "S"], team_df.loc[team, "E"]],
            marker_color=colors_map.get(team, "#a78bfa"),
        ))
    fig_teams.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892b0"), yaxis=dict(range=[0, 100], gridcolor="#2d3448"),
        margin=dict(t=10, b=10, l=10, r=10), height=280,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_teams, use_container_width=True)


# ── TAB 2: Аналитика слоёв ───────────────────────────────────────────────────
with tab2:
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="section-header">Радарная диаграмма слоёв</div>', unsafe_allow_html=True)

        categories = ["P-layer", "R-layer", "V-layer", "S-layer"]
        fig_radar = go.Figure()

        palette = ["#6c8eff", "#34d399", "#f59e0b", "#f87171", "#a78bfa", "#38bdf8", "#fb7185", "#4ade80"]

        def hex_to_rgba(hex_color, alpha=0.1):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        for i, (_, row) in enumerate(df.iterrows()):
            vals = [row["P"], row["R"], row["V"], row["S"]]
            color = palette[i % len(palette)]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                name=row["login"],
                line=dict(color=color, width=2),
                fill="toself",
                fillcolor=hex_to_rgba(color, 0.07),
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2d3448", color="#4a5568"),
                angularaxis=dict(gridcolor="#2d3448", color="#8892b0"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892b0"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            margin=dict(t=30, b=20, l=30, r=30),
            height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Тепловая карта слоёв</div>', unsafe_allow_html=True)

        heat_data = df[["login", "P", "R", "V", "S"]].set_index("login")
        fig_heat = go.Figure(go.Heatmap(
            z=heat_data.values,
            x=["P-layer", "R-layer", "V-layer", "S-layer"],
            y=heat_data.index.tolist(),
            colorscale="Blues",
            text=heat_data.values.round(1),
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(tickfont=dict(color="#8892b0")),
        ))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892b0"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=400,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Scatter: P vs E
    st.markdown('<div class="section-header">Зависимость слоёв от итоговой оценки</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for col, layer, label in [(c1, "P", "P-layer (процессы)"), (c2, "V", "V-layer (ценность)")]:
        fig_sc = px.scatter(
            df.reset_index(), x=layer, y="E", text="login",
            color="role", size_max=40,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={layer: label, "E": "Интегральная оценка E"},
        )
        fig_sc.update_traces(textposition="top center", marker=dict(size=12))
        fig_sc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892b0"),
            xaxis=dict(gridcolor="#2d3448"), yaxis=dict(gridcolor="#2d3448"),
            margin=dict(t=20, b=10, l=10, r=10), height=280,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        col.plotly_chart(fig_sc, use_container_width=True)


# ── TAB 3: Профиль участника ─────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Профиль участника</div>', unsafe_allow_html=True)
    selected = st.selectbox("Выберите участника", df["login"].tolist())

    row = df[df["login"] == selected].iloc[0]
    user_metrics = metrics_df[metrics_df["author"].str.lower() == selected.lower()]

    col1, col2, col3, col4 = st.columns(4)
    for c, val, lbl in [
        (col1, f"{row['E']:.1f}", "Итоговая оценка E"),
        (col2, f"{row['closed_count']:.0f}", "Закрыто задач"),
        (col3, f"{row['avg_close_time']:.0f} ч", "Ср. время закрытия"),
        (col4, f"{row['avg_comments']:.1f}", "Ср. комментариев"),
    ]:
        c.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([1, 2], gap="large")

    with cl:
        st.markdown(f"""
        <div style="background:#1e2538;border:1px solid #2d3448;border-radius:12px;padding:1rem 1.5rem;">
            <p style="color:#8892b0;margin:0 0 0.5rem 0;font-size:0.75rem;text-transform:uppercase;">Информация</p>
            <p><strong style="color:#e2e8f0;">{selected}</strong></p>
            <p style="color:#8892b0;">Роль: <span style="color:#6c8eff;">{row['role'].capitalize()}</span></p>
            <p style="color:#8892b0;">Уровень: <span style="color:#34d399;">{row['level'].capitalize()}</span></p>
            <p style="color:#8892b0;">Команда: <span style="color:#a78bfa;">{row['team']}</span></p>
            <hr style="border-color:#2d3448;">
            <p style="color:#8892b0;font-size:0.8rem;">Слои:</p>
            <table style="width:100%;color:#e2e8f0;font-size:0.9rem;">
                <tr><td>P-layer</td><td style="text-align:right;color:#93c5fd;font-weight:700;">{row['P']:.1f}</td></tr>
                <tr><td>R-layer</td><td style="text-align:right;color:#6ee7b7;font-weight:700;">{row['R']:.1f}</td></tr>
                <tr><td>V-layer</td><td style="text-align:right;color:#fcd34d;font-weight:700;">{row['V']:.1f}</td></tr>
                <tr><td>S-layer</td><td style="text-align:right;color:#d8b4fe;font-weight:700;">{row['S']:.1f}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with cr:
        # Радар одного участника
        cats = ["P-layer", "R-layer", "V-layer", "S-layer"]
        vals = [row["P"], row["R"], row["V"], row["S"]]
        avg_vals = [df["P"].mean(), df["R"].mean(), df["V"].mean(), df["S"].mean()]

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            name=selected, fill="toself",
            line=dict(color="#6c8eff", width=2),
            fillcolor="rgba(108,142,255,0.15)",
        ))
        fig_p.add_trace(go.Scatterpolar(
            r=avg_vals + [avg_vals[0]], theta=cats + [cats[0]],
            name="Среднее", fill="toself",
            line=dict(color="#8892b0", width=1.5, dash="dot"),
            fillcolor="rgba(136,146,176,0.07)",
        ))
        fig_p.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2d3448", color="#4a5568"),
                angularaxis=dict(gridcolor="#2d3448", color="#8892b0"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8892b0"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=20, l=20, r=20), height=300,
        )
        st.plotly_chart(fig_p, use_container_width=True)

    # Задачи участника
    st.markdown('<div class="section-header">Задачи и Pull Requests</div>', unsafe_allow_html=True)
    st.dataframe(
        user_metrics[["id", "type", "labels", "state", "time_to_close_hours", "comments", "created_at"]].reset_index(drop=True),
        use_container_width=True, height=250
    )


# ── TAB 4: Данные ─────────────────────────────────────────────────────────────
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">metrics.csv</div>', unsafe_allow_html=True)
        st.dataframe(metrics_df, use_container_width=True, height=300)
    with c2:
        st.markdown('<div class="section-header">users.csv</div>', unsafe_allow_html=True)
        st.dataframe(users_df, use_container_width=True, height=300)

    st.markdown('<div class="section-header">Конфигурация (config.yaml)</div>', unsafe_allow_html=True)
    st.json(config)

# ─────────────────────────────────────────────────────────────────────────────
# Экспорт
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
col_exp1, col_exp2, _ = st.columns([1, 1, 4])

with col_exp1:
    excel_bytes = export_excel(df, config.get("weights", {}))
    st.download_button(
        label="📥 Скачать Excel-отчёт",
        data=excel_bytes,
        file_name="performance_analytics_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with col_exp2:
    csv_bytes = df.to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        label="📥 Скачать CSV",
        data=csv_bytes,
        file_name="performance_analytics_scores.csv",
        mime="text/csv",
        use_container_width=True,
    )
