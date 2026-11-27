"""Reusable Enterprise HR Analytics SaaS styling and UI components for Streamlit.

This module provides presentation-only helpers (CSS tokens, cards, status banners,
chart theming, and aggregate download links). It contains no ML, SHAP, or monitoring logic.
"""
from __future__ import annotations

import base64
import html
from typing import Any

import matplotlib.axes
import matplotlib.figure
import plotly.graph_objects as go
import streamlit as st

STATUS_STYLES: dict[str, tuple[str, str, str]] = {
    "Healthy": ("#10b981", "rgba(16, 185, 129, 0.12)", "rgba(16, 185, 129, 0.35)"),
    "Available": ("#10b981", "rgba(16, 185, 129, 0.12)", "rgba(16, 185, 129, 0.35)"),
    "Warning": ("#f59e0b", "rgba(245, 158, 11, 0.12)", "rgba(245, 158, 11, 0.35)"),
    "Critical": ("#ef4444", "rgba(239, 68, 68, 0.12)", "rgba(239, 68, 68, 0.35)"),
    "Insufficient Data": ("#94a3b8", "rgba(148, 163, 184, 0.12)", "rgba(148, 163, 184, 0.30)"),
}


def inject_saas_theme() -> None:
    """Inject clean Enterprise HR Analytics SaaS CSS for dark-mode Streamlit."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1380px;
        }
        .saas-hero {
            background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 1.35rem 1.6rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.35);
        }
        .saas-hero-top {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.45rem;
        }
        .saas-kicker {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #38bdf8;
        }
        .saas-env-pill {
            font-size: 0.74rem;
            font-weight: 600;
            color: #cbd5e1;
            background: rgba(56, 189, 248, 0.10);
            border: 1px solid rgba(56, 189, 248, 0.28);
            border-radius: 999px;
            padding: 0.18rem 0.65rem;
        }
        .saas-hero-title {
            font-size: 1.85rem;
            font-weight: 700;
            color: #f8fafc;
            margin: 0 0 0.25rem 0;
            line-height: 1.2;
        }
        .saas-hero-subtitle {
            font-size: 0.96rem;
            color: #94a3b8;
            margin: 0;
            line-height: 1.45;
        }
        .saas-section-header {
            border-left: 3px solid #38bdf8;
            padding-left: 0.85rem;
            margin: 0.5rem 0 1rem 0;
        }
        .saas-section-title {
            font-size: 1.32rem;
            font-weight: 700;
            color: #f8fafc;
            margin: 0.1rem 0 0.2rem 0;
        }
        .saas-section-desc {
            font-size: 0.90rem;
            color: #94a3b8;
            margin: 0;
        }
        .saas-status-banner {
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin: 0.75rem 0 1rem 0;
        }
        .saas-status-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.45rem;
        }
        .saas-badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
        }
        .saas-profile-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
            gap: 0.65rem;
            margin: 0.75rem 0 1rem 0;
        }
        .saas-profile-item {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 10px;
            padding: 0.65rem 0.8rem;
        }
        .saas-profile-label {
            font-size: 0.74rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.2rem;
        }
        .saas-profile-value {
            font-size: 1.0rem;
            font-weight: 600;
            color: #f8fafc;
        }
        div[data-testid="stMetric"] {
            background: #111827;
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            box-shadow: 0 2px 10px rgba(2, 6, 23, 0.22);
        }
        div[data-testid="stMetric"] label {
            color: #94a3b8 !important;
            font-size: 0.82rem !important;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-weight: 700 !important;
        }
        div[data-testid="stForm"] {
            background: #0f172a;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 12px;
            padding: 1rem 1.15rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
            padding-bottom: 0.25rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.45rem 0.9rem;
            font-weight: 600;
        }
        .saas-download-btn {
            display: inline-block;
            padding: 0.48rem 0.95rem;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.38);
            border-radius: 8px;
            color: #e0f2fe !important;
            font-size: 0.88rem;
            font-weight: 600;
            text-decoration: none !important;
            margin-top: 0.35rem;
            margin-right: 0.5rem;
        }
        .saas-download-btn:hover {
            background: rgba(56, 189, 248, 0.22);
            border-color: #38bdf8;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }
            .saas-hero-title {
                font-size: 1.45rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_header(title: str, subtitle: str, env_label: str = "Educational Prototype") -> None:
    """Render the top Enterprise SaaS header card."""
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle)
    safe_env = html.escape(env_label)
    st.markdown(
        f"""
        <div class="saas-hero">
            <div class="saas-hero-top">
                <span class="saas-kicker">Responsible Workforce Analytics &amp; Explainable AI</span>
                <span class="saas-env-pill">{safe_env}</span>
            </div>
            <h1 class="saas-hero-title">{safe_title}</h1>
            <p class="saas-hero-subtitle">{safe_subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(kicker: str, title: str, description: str = "") -> None:
    """Render a structured section header with kicker, title, and optional description."""
    safe_kicker = html.escape(kicker)
    safe_title = html.escape(title)
    desc_html = (
        f'<p class="saas-section-desc">{html.escape(description)}</p>'
        if description
        else ""
    )
    st.markdown(
        f"""
        <div class="saas-section-header">
            <div class="saas-kicker">{safe_kicker}</div>
            <div class="saas-section-title">{safe_title}</div>
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(title: str, status: str, reasons: list[str] | None = None) -> None:
    """Render a color-coded monitoring/quality status banner with clear reasons."""
    fg, bg, border = STATUS_STYLES.get(status, STATUS_STYLES["Insufficient Data"])
    safe_title = html.escape(title)
    safe_status = html.escape(status)
    reason_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in (reasons or []))
    reasons_html = f"<ul style='margin:0.35rem 0 0 1.15rem;color:#cbd5e1;'>{reason_items}</ul>" if reason_items else ""
    st.markdown(
        f"""
        <div class="saas-status-banner" style="background:{bg};border:1px solid {border};">
            <div class="saas-status-header">
                <span class="saas-badge" style="color:{fg};background:{bg};border:1px solid {border};">{safe_status}</span>
                <strong style="color:#f8fafc;font-size:1.0rem;">{safe_title}</strong>
            </div>
            {reasons_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_grid(record: dict[str, Any]) -> None:
    """Render an employee feature dictionary as a clean responsive card grid."""
    cards = []
    for key, value in record.items():
        label = html.escape(str(key).replace("_", " "))
        formatted = f"{value:.3f}" if isinstance(value, float) else str(value)
        safe_val = html.escape(formatted)
        cards.append(
            f'<div class="saas-profile-item">'
            f'<div class="saas-profile-label">{label}</div>'
            f'<div class="saas-profile-value">{safe_val}</div>'
            f"</div>"
        )
    st.markdown(
        f'<div class="saas-profile-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_download_link(
    payload: bytes,
    filename: str,
    label: str,
    mime_type: str = "text/csv",
) -> None:
    """Render a styled browser-native download button from a byte payload."""
    encoded = base64.b64encode(payload).decode("ascii")
    safe_filename = html.escape(filename)
    safe_label = html.escape(label)
    safe_mime = html.escape(mime_type)
    st.markdown(
        f'<a href="data:{safe_mime};base64,{encoded}" '
        f'download="{safe_filename}" class="saas-download-btn">{safe_label}</a>',
        unsafe_allow_html=True,
    )


def style_plotly_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    """Apply consistent dark-surface Enterprise SaaS styling to a Plotly figure."""
    layout_kwargs: dict[str, Any] = {
        "paper_bgcolor": "rgba(17, 24, 39, 0.0)",
        "plot_bgcolor": "rgba(17, 24, 39, 0.55)",
        "font": {"color": "#e2e8f0", "size": 12},
        "title_font": {"color": "#f8fafc", "size": 15},
        "margin": {"l": 24, "r": 24, "t": 52, "b": 36},
        "legend": {
            "bgcolor": "rgba(15, 23, 42, 0.65)",
            "bordercolor": "rgba(148, 163, 184, 0.18)",
            "borderwidth": 1,
        },
    }
    if height is not None:
        layout_kwargs["height"] = height
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.14)", zerolinecolor="rgba(148, 163, 184, 0.2)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.14)", zerolinecolor="rgba(148, 163, 184, 0.2)")
    return fig


def style_matplotlib_figure(fig: matplotlib.figure.Figure) -> matplotlib.figure.Figure:
    """Adapt Matplotlib figures to match the dark dashboard surface cleanly."""
    surface = "#111827"
    text_color = "#e2e8f0"
    muted_color = "#94a3b8"
    spine_color = "#334155"

    fig.patch.set_facecolor(surface)
    for ax in fig.axes:
        if isinstance(ax, matplotlib.axes.Axes):
            ax.set_facecolor(surface)
            ax.tick_params(colors=muted_color, labelsize=9)
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.title.set_color("#f8fafc")
            for spine in ax.spines.values():
                spine.set_color(spine_color)
    fig.tight_layout()
    return fig
