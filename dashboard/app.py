"""
dashboard/app.py
================
Streamlit dashboard for football match analysis.
Reads precomputed data from data/processed/ (no live ML inference).

Run:
    streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")

BLOCK_LABELS = ["0–15'", "15–30'", "30–45'", "45–60'", "60–75'", "75–90'"]
RISK_COLORS  = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444", "INSUFFICIENT": "#94a3b8"}
RISK_BG      = {"LOW": "#f0fdf4", "MEDIUM": "#fffbeb", "HIGH": "#fef2f2", "INSUFFICIENT": "#f1f5f9"}
RISK_TEXT    = {"LOW": "#15803d", "MEDIUM": "#b45309", "HIGH": "#b91c1c", "INSUFFICIENT": "#475569"}

PITCH_L = 105   # metres (for heatmap overlay)
PITCH_W = 68

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Match Analysis · Mitus.AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* card component */
.stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.stat-card .value { font-size: 1.7rem; font-weight: 700; color: #1e293b; }
.stat-card .label { font-size: 0.78rem; color: #64748b; margin-top: 2px; }

/* risk badge */
.badge {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

/* hide streamlit hamburger */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading match data…")
def load_data():
    required = [
        DATA_DIR / "player_summary.json",
        DATA_DIR / "risk_scores.json",
        DATA_DIR / "player_stats.parquet",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        return None, None, None, missing

    with open(DATA_DIR / "player_summary.json") as f:
        summary = json.load(f)
    with open(DATA_DIR / "risk_scores.json") as f:
        risk = json.load(f)
    stats_df = pd.read_parquet(DATA_DIR / "player_stats.parquet")
    return summary, risk, stats_df, []


# ── Pitch heatmap ──────────────────────────────────────────────────────────────

def draw_pitch_heatmap(heatmap: list[list[float]], title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#1a1a2e")
    ax.set_facecolor("#1e5631")

    arr = np.array(heatmap)
    im = ax.imshow(
        arr.T,
        extent=[0, PITCH_L, 0, PITCH_W],
        origin="lower",
        cmap="hot",
        alpha=0.75,
        aspect="auto",
        vmin=0,
    )

    # Pitch markings
    c, lw = "white", 1.2
    ax.plot([0, PITCH_L, PITCH_L, 0, 0], [0, 0, PITCH_W, PITCH_W, 0], c, lw=lw)
    ax.plot([PITCH_L/2, PITCH_L/2], [0, PITCH_W], c, lw=lw)
    ax.add_patch(plt.Circle((PITCH_L/2, PITCH_W/2), 9.15, color=c, fill=False, lw=lw))
    # Penalty boxes
    ax.plot([0, 16.5, 16.5, 0], [13.85, 13.85, 54.15, 54.15], c, lw=lw)
    ax.plot([PITCH_L, PITCH_L-16.5, PITCH_L-16.5, PITCH_L],
            [13.85, 13.85, 54.15, 54.15], c, lw=lw)

    cbar = plt.colorbar(im, ax=ax, shrink=0.55)
    cbar.set_label("Time fraction", color="white")
    cbar.ax.tick_params(colors="white")
    ax.set_xlim(-2, PITCH_L + 2)
    ax.set_ylim(-2, PITCH_W + 2)
    ax.axis("off")
    if title:
        ax.set_title(title, color="white", fontsize=10, pad=4)
    plt.tight_layout(pad=0.5)
    return fig


# ── Reusable components ────────────────────────────────────────────────────────

def risk_badge(flag: str) -> str:
    bg   = RISK_BG.get(flag, "#f1f5f9")
    text = RISK_TEXT.get(flag, "#475569")
    return (f'<span class="badge" style="background:{bg};color:{text};'
            f'border:1px solid {text}33;">{flag}</span>')


def block_label(block: int) -> str:
    return BLOCK_LABELS[block] if block < len(BLOCK_LABELS) else f"B{block}"


# ── Team Overview ──────────────────────────────────────────────────────────────

def page_team_overview(summary: dict, risk: dict, stats_df: pd.DataFrame):
    st.header("Team Overview")

    players   = summary["players"]
    dur_min   = summary["match_duration_s"] / 60
    n_players = summary["total_players"]
    n_full    = sum(1 for r in risk.values() if r["risk_flag"] != "INSUFFICIENT")
    n_high    = sum(1 for r in risk.values() if r["risk_flag"] == "HIGH")

    # Average distance over full-match tracks only (partial fragments would drag it down)
    full_dists = [
        players[tid]["total_distance_m"] for tid, r in risk.items()
        if r["risk_flag"] != "INSUFFICIENT" and tid in players
    ]
    avg_dist = (np.mean(full_dists) / 1000) if full_dists else \
               (np.mean([p["total_distance_m"] for p in players.values()]) / 1000)

    # ── Top KPI row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{dur_min:.0f} min",                "Match Duration"),
        (c2, f"{n_full} / {n_players}",            "Full-match / total tracks"),
        (c3, f"{avg_dist:.2f} km",                "Avg Distance (full-match)"),
        (c4, str(n_high),                          "High-Risk Players"),
    ]:
        col.markdown(
            f'<div class="stat-card"><div class="value">{val}</div>'
            f'<div class="label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    only_full = st.checkbox(
        "Show only full-match tracks (hide partial / ID-switch fragments)",
        value=True,
    )

    # ── Player table ───────────────────────────────────────────────────────────
    rows = []
    for tid, p in players.items():
        r = risk.get(tid, {})
        flag = r.get("risk_flag", "—")
        if only_full and flag == "INSUFFICIENT":
            continue
        score = r.get("risk_score")
        rows.append({
            "ID": tid,
            "Coverage": round(p.get("coverage_frac", 0) * 100),
            "Distance (km)":  round(p["total_distance_m"] / 1000, 2),
            "Avg Speed":       round(p["mean_speed_ms"], 2),
            "Peak Speed":      round(p["peak_speed_ms"], 2),
            "Sprints":         p["total_sprints"],
            "Active (min)":    round(p["active_time_s"] / 60, 1),
            "Risk":            score if score is not None else np.nan,
            "Flag":            flag,
        })

    table_df = (
        pd.DataFrame(rows)
        .sort_values("Risk", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    col_table, col_charts = st.columns([3, 2], gap="large")

    with col_table:
        st.subheader("Player Rankings")

        def color_flag(val):
            style = {
                "HIGH":         "background-color:#fef2f2;color:#b91c1c;font-weight:600",
                "MEDIUM":       "background-color:#fffbeb;color:#b45309;font-weight:600",
                "LOW":          "background-color:#f0fdf4;color:#15803d;font-weight:600",
                "INSUFFICIENT": "background-color:#f1f5f9;color:#475569;font-weight:600",
            }
            return style.get(val, "")

        styled = table_df.style.map(color_flag, subset=["Flag"])
        st.dataframe(styled, width="stretch", hide_index=True)

    with col_charts:
        # Risk distribution donut
        counts = table_df["Flag"].value_counts()
        fig_pie = px.pie(
            values=counts.values,
            names=counts.index,
            color=counts.index,
            color_discrete_map=RISK_COLORS,
            hole=0.5,
            title="Injury-Risk Distribution",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, width="stretch")

    # ── Team distance per block ────────────────────────────────────────────────
    st.subheader("Team Total Distance per 15-min Block")
    team_dist = stats_df.groupby("block")["distance_m"].sum().reset_index()
    team_dist["period"] = team_dist["block"].apply(block_label)
    team_dist["half"] = team_dist["block"].apply(lambda x: "1st Half" if x < 3 else "2nd Half")

    fig_bar = px.bar(
        team_dist, x="period", y="distance_m", color="half",
        color_discrete_map={"1st Half": "#3b82f6", "2nd Half": "#8b5cf6"},
        labels={"period": "Period", "distance_m": "Distance (m)", "half": ""},
    )
    fig_bar.update_layout(margin=dict(t=10, b=10), legend_title_text="")
    st.plotly_chart(fig_bar, width="stretch")


# ── Player Detail ──────────────────────────────────────────────────────────────

def page_player_detail(summary: dict, risk: dict, stats_df: pd.DataFrame):
    players = summary["players"]
    ids = sorted(players.keys(), key=lambda x: int(x))

    selected = st.sidebar.selectbox(
        "Select player", ids, format_func=lambda x: f"Player {x}"
    )
    if not selected:
        st.info("Select a player from the sidebar.")
        return

    player = players[selected]
    r      = risk.get(selected, {})
    pstats = stats_df[stats_df["track_id"] == int(selected)].copy()
    pstats["period"] = pstats["block"].apply(block_label)

    flag  = r.get("risk_flag", "N/A")
    score = r.get("risk_score")
    coverage = player.get("coverage_frac", 0)

    # ── Header ─────────────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([4, 1])
    col_h1.header(f"Player {selected}")
    score_txt = f"<b>{score}/100</b>" if score is not None else ""
    col_h2.markdown(
        f"<br>{risk_badge(flag)}&nbsp;{score_txt}",
        unsafe_allow_html=True,
    )
    st.caption(f"Active for {player['active_time_s']/60:.1f} min "
               f"· {coverage*100:.0f}% match coverage")
    if flag == "INSUFFICIENT":
        st.warning(
            "Partial track — this ID is not present across both halves "
            "(likely a substitution or a ByteTrack ID switch), so a fatigue "
            "score is intentionally not computed. Movement stats below are "
            "still valid for the period this ID was tracked."
        )
    st.divider()

    # ── KPI metrics ────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl in [
        (k1, f"{player['total_distance_m']/1000:.2f} km", "Total Distance"),
        (k2, f"{player['mean_speed_ms']:.2f} m/s",        "Avg Speed"),
        (k3, f"{player['peak_speed_ms']:.2f} m/s",        "Peak Speed"),
        (k4, str(player["total_sprints"]),                 "Total Sprints"),
    ]:
        col.markdown(
            f'<div class="stat-card"><div class="value">{val}</div>'
            f'<div class="label">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Speed over match ───────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Speed Over Match")
        tl = {int(k): v for k, v in player.get("speed_timeline", {}).items()}
        if tl:
            blocks_sorted = sorted(tl.keys())
            fig_speed = go.Figure()
            fig_speed.add_trace(go.Scatter(
                x=[block_label(b) for b in blocks_sorted],
                y=[tl[b] for b in blocks_sorted],
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2.5),
                marker=dict(size=8, color="#1d4ed8"),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.08)",
                name="Avg speed",
                hovertemplate="%{x}: <b>%{y:.2f} m/s</b><extra></extra>",
            ))
            # Halfway mark
            fig_speed.add_vline(
                x=2.5, line_dash="dot", line_color="#94a3b8",
                annotation_text="Half-time", annotation_position="top right",
            )
            fig_speed.update_layout(
                yaxis_title="Speed (m/s)", xaxis_title="Period",
                margin=dict(t=10, b=10), hovermode="x unified",
            )
            st.plotly_chart(fig_speed, width="stretch")

    with col_r:
        st.subheader("Sprints per Period")
        if len(pstats) > 0:
            fig_sprint = px.bar(
                pstats.groupby(["period", "block"])["sprint_count"]
                      .sum().reset_index(),
                x="period", y="sprint_count",
                color_discrete_sequence=["#f59e0b"],
                labels={"period": "Period", "sprint_count": "Sprints"},
            )
            fig_sprint.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig_sprint, width="stretch")

    # ── Distance per block ─────────────────────────────────────────────────────
    st.subheader("Distance per 15-min Block")
    if len(pstats) > 0:
        dist_data = pstats.groupby(["period", "block", "half"])["distance_m"] \
                          .sum().reset_index()
        dist_data["Half"] = dist_data["half"].map({0: "1st Half", 1: "2nd Half"})
        fig_dist = px.bar(
            dist_data, x="period", y="distance_m", color="Half",
            color_discrete_map={"1st Half": "#3b82f6", "2nd Half": "#8b5cf6"},
            labels={"period": "Period", "distance_m": "Distance (m)", "Half": ""},
        )
        fig_dist.update_layout(margin=dict(t=10, b=10), legend_title_text="")
        st.plotly_chart(fig_dist, width="stretch")

    # ── Heatmap + Fatigue breakdown ────────────────────────────────────────────
    col_hm, col_fat = st.columns(2, gap="large")

    with col_hm:
        st.subheader("Position Heatmap")
        hm = player.get("heatmap")
        if hm:
            fig_hm = draw_pitch_heatmap(hm, f"Player {selected}")
            st.pyplot(fig_hm)
            plt.close(fig_hm)

    with col_fat:
        st.subheader("Fatigue & Risk Breakdown")
        breakdown = r.get("score_breakdown", {})
        indicators = r.get("fatigue_indicators", {})

        if breakdown:
            labels = ["Speed Decay", "Sprint Drop", "Distance Drop", "HSR Load"]
            values = list(breakdown.values())
            bar_colors = [
                "#ef4444" if v > 18 else "#f59e0b" if v > 10 else "#22c55e"
                for v in values
            ]
            fig_fat = go.Figure(go.Bar(
                y=labels,
                x=values,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:.1f}" for v in values],
                textposition="inside",
                hovertemplate="%{y}: <b>%{x:.1f}/25</b><extra></extra>",
            ))
            fig_fat.update_layout(
                xaxis=dict(title="Score contribution (0–25 each)", range=[0, 27]),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_fat, width="stretch")

        if indicators:
            c1f, c2f = st.columns(2)
            c1f.metric("Speed trend",
                       f"{indicators.get('speed_decay_slope_ms_per_block', 0):+.4f} m/s/block",
                       help="Negative = declining pace")
            c2f.metric("Sprint drop 2nd half",
                       f"{indicators.get('sprint_drop_pct', 0):.1f}%")
            c1f.metric("Distance drop 2nd half",
                       f"{indicators.get('distance_drop_pct', 0):.1f}%")
            c2f.metric("H1 / H2 sprints",
                       f"{indicators.get('h1_sprints', 0)} / {indicators.get('h2_sprints', 0)}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    summary, risk, stats_df, missing = load_data()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("## ⚽ Match Analysis")
    if summary:
        dur  = summary["match_duration_s"] / 60
        n    = summary["total_players"]
        fps  = summary.get("effective_fps", "?")
        st.sidebar.caption(f"{dur:.0f} min · {n} players tracked · {fps:.0f} FPS processed")
        st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["Team Overview", "Player Detail"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption("Powered by YOLOv8 + ByteTrack")

    # ── Main area ──────────────────────────────────────────────────────────────
    if missing:
        st.error("Processed data not found. Run the pipeline first:")
        st.code(
            "python pipeline/01_track.py --video data/raw/match.mp4\n"
            "python pipeline/02_stats.py\n"
            "python pipeline/03_fatigue.py",
            language="bash",
        )
        st.info("No video yet? Generate synthetic demo data:")
        st.code("python scripts/gen_dummy_data.py", language="bash")
        return

    if page == "Team Overview":
        page_team_overview(summary, risk, stats_df)
    else:
        page_player_detail(summary, risk, stats_df)


if __name__ == "__main__":
    main()
