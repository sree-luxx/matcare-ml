"""
MATCARE-AI design system — professional dark healthcare theme.

A single source of truth for colours, typography and the CSS that gives the
dashboard a premium clinical-SaaS look. Rendering code should read the tokens
from ``PALETTE`` instead of hard-coding colours.

The Streamlit widget chrome (inputs, sliders, dataframes) is themed through
``.streamlit/config.toml``; this module adds the custom surface styling
(gradient page, glass cards, headers, pills, banners).
"""

import streamlit as st

PALETTE = {
    # ---- page & surfaces -------------------------------------------------
    "page_bg": "#070b16",
    "page_bg_2": "#0a1120",
    "surface": "rgba(19, 28, 47, 0.72)",
    "surface_solid": "#111a2b",
    "surface_muted": "rgba(30, 41, 64, 0.55)",
    "glass_high": "linear-gradient(160deg, rgba(27, 39, 63, 0.92), rgba(13, 20, 36, 0.78))",
    # ---- borders ---------------------------------------------------------
    "border": "rgba(148, 163, 184, 0.16)",
    "border_strong": "rgba(148, 163, 184, 0.34)",
    # ---- text ------------------------------------------------------------
    "navy": "#eef4ff",
    "navy_soft": "#c9d7ef",
    "text": "#dbe6f6",
    "text_muted": "#93a4bf",
    "text_subtle": "#6b7c99",
    # ---- accents ---------------------------------------------------------
    "teal": "#2dd4bf",
    "teal_soft": "rgba(45, 212, 191, 0.15)",
    "cyan": "#22d3ee",
    "blue": "#60a5fa",
    "blue_soft": "rgba(96, 165, 250, 0.16)",
    "green": "#34d399",
    "green_soft": "rgba(52, 211, 153, 0.14)",
    "amber": "#fbbf24",
    "amber_soft": "rgba(251, 191, 36, 0.15)",
    "red": "#f87171",
    "red_soft": "rgba(248, 113, 113, 0.15)",
    "slate_soft": "rgba(148, 163, 184, 0.08)",
    "accent_grad": "linear-gradient(135deg, #22d3ee 0%, #2dd4bf 46%, #60a5fa 100%)",
    # ---- charts ----------------------------------------------------------
    "chart_bg": "rgba(0, 0, 0, 0)",
    "chart_grid": "rgba(148, 163, 184, 0.14)",
    "chart_font": "#aab9d2",
    "chart_zeroline": "rgba(148, 163, 184, 0.35)",
}

STATUS = {
    "normal": {"label": "Normal", "fg": PALETTE["green"], "bg": PALETTE["green_soft"]},
    "warning": {"label": "Review", "fg": PALETTE["amber"], "bg": PALETTE["amber_soft"]},
    "critical": {"label": "Critical", "fg": PALETTE["red"], "bg": PALETTE["red_soft"]},
    "info": {"label": "Info", "fg": PALETTE["blue"], "bg": PALETTE["blue_soft"]},
}

FONT_STACK = "'Inter', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: %(font)s;
        color: %(text)s;
    }

    /* ---------- Page ---------- */
    .stApp {
        background:
            radial-gradient(1150px 620px at 10%% -10%%, rgba(34, 211, 238, 0.10), transparent 60%%),
            radial-gradient(1000px 560px at 100%% -6%%, rgba(96, 165, 250, 0.10), transparent 58%%),
            radial-gradient(900px 700px at 50%% 120%%, rgba(45, 212, 191, 0.07), transparent 60%%),
            linear-gradient(180deg, %(page_bg)s 0%%, %(page_bg_2)s 100%%);
        background-attachment: fixed;
    }

    /* Keep the Streamlit chrome quiet */
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }

    /* ---------- Header ---------- */
    .mc-header {
        position: relative;
        background: %(glass_high)s;
        border: 1px solid %(border)s;
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 18px 46px rgba(2, 6, 23, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(14px);
        overflow: hidden;
    }
    .mc-header::before {
        content: "";
        position: absolute; left: 0; top: 0; bottom: 0;
        width: 5px;
        background: %(accent_grad)s;
    }
    .mc-brand {display: flex; align-items: center; gap: 16px; flex-wrap: wrap;}
    .mc-logo {
        width: 48px; height: 48px;
        border-radius: 14px;
        background: %(accent_grad)s;
        color: #04121a;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 1.2rem; letter-spacing: -0.03em;
        box-shadow: 0 8px 22px rgba(34, 211, 238, 0.28);
    }
    .mc-title {
        font-family: 'Plus Jakarta Sans', %(font)s;
        font-size: 1.55rem; font-weight: 800; color: %(navy)s;
        margin: 0; letter-spacing: -0.02em;
    }
    .mc-subtitle {font-size: 0.9rem; color: %(text_muted)s; margin: 3px 0 0 0;}
    .mc-meta {
        display: flex; gap: 12px; flex-wrap: wrap;
        margin-top: 18px; padding-top: 15px;
        border-top: 1px solid %(border)s;
    }
    .mc-meta-item {
        font-size: 0.8rem; color: %(text_muted)s;
        background: %(surface_muted)s; border: 1px solid %(border)s;
        border-radius: 9px; padding: 6px 12px;
    }
    .mc-meta-item b {color: %(navy)s; font-weight: 600;}

    .mc-disclaimer {
        margin-top: 16px;
        background: %(amber_soft)s;
        color: %(amber)s;
        border: 1px solid rgba(251, 191, 36, 0.28);
        border-radius: 11px;
        padding: 11px 15px;
        font-size: 0.8rem;
        line-height: 1.5;
    }

    /* ---------- Glass cards ---------- */
    .mc-card {
        background: %(glass_high)s;
        border: 1px solid %(border)s;
        border-radius: 16px;
        padding: 18px 21px;
        box-shadow: 0 12px 34px rgba(2, 6, 23, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        height: 100%%;
        transition: border-color 0.18s ease, transform 0.18s ease;
    }
    .mc-card:hover {border-color: %(border_strong)s;}
    .mc-card-title {
        font-family: 'Plus Jakarta Sans', %(font)s;
        font-size: 0.98rem; font-weight: 700; color: %(navy)s; margin: 0 0 4px 0;
    }
    .mc-card-caption {font-size: 0.78rem; color: %(text_muted)s; margin: 0 0 12px 0;}

    /* ---------- KPI cards ---------- */
    .mc-kpi {
        background: %(glass_high)s;
        border: 1px solid %(border)s;
        border-radius: 16px;
        padding: 16px 18px;
        height: 100%%;
        box-shadow: 0 12px 30px rgba(2, 6, 23, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }
    .mc-kpi::after {
        content: "";
        position: absolute; inset: 0 0 auto 0; height: 3px;
        background: %(accent_grad)s; opacity: 0.75;
    }
    .mc-kpi-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em;
        text-transform: uppercase; color: %(text_subtle)s; margin-bottom: 9px;
    }
    .mc-kpi-value {font-size: 1.55rem; font-weight: 800; color: %(navy)s; line-height: 1.1;}
    .mc-kpi-note {font-size: 0.75rem; color: %(text_muted)s; margin-top: 7px;}

    /* ---------- Pills ---------- */
    .mc-pill {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 0.73rem; font-weight: 700; border: 1px solid transparent;
    }
    .mc-pill-normal {background: %(green_soft)s; color: %(green)s; border-color: rgba(52, 211, 153, 0.3);}
    .mc-pill-warning {background: %(amber_soft)s; color: %(amber)s; border-color: rgba(251, 191, 36, 0.3);}
    .mc-pill-critical {background: %(red_soft)s; color: %(red)s; border-color: rgba(248, 113, 113, 0.3);}
    .mc-pill-info {background: %(blue_soft)s; color: %(blue)s; border-color: rgba(96, 165, 250, 0.3);}

    .mc-tag {
        display: inline-block; background: %(surface_muted)s; color: %(text_muted)s;
        border: 1px solid %(border)s; border-radius: 7px;
        padding: 2px 9px; font-size: 0.72rem; font-weight: 600;
    }
    .mc-tag-context {background: %(blue_soft)s; color: %(blue)s; border-color: rgba(96, 165, 250, 0.3);}

    /* ---------- Banners ---------- */
    .mc-banner {
        border-radius: 14px; padding: 17px 21px; margin-bottom: 18px;
        border: 1px solid; display: flex; justify-content: space-between;
        align-items: center; gap: 16px; flex-wrap: wrap;
        backdrop-filter: blur(10px);
    }
    .mc-banner h3 {margin: 0; font-size: 1.12rem; font-weight: 800; color: %(navy)s;}
    .mc-banner p {margin: 7px 0 0 0; font-size: 0.87rem; line-height: 1.5;}
    .mc-banner-critical {background: %(red_soft)s; border-color: rgba(248, 113, 113, 0.4);}
    .mc-banner-critical h3, .mc-banner-critical p {color: %(red)s;}
    .mc-banner-warning {background: %(amber_soft)s; border-color: rgba(251, 191, 36, 0.4);}
    .mc-banner-warning h3, .mc-banner-warning p {color: %(amber)s;}
    .mc-banner-normal {background: %(green_soft)s; border-color: rgba(52, 211, 153, 0.4);}
    .mc-banner-normal h3, .mc-banner-normal p {color: %(green)s;}
    .mc-banner-score {
        font-family: 'Plus Jakarta Sans', %(font)s;
        font-size: 1.7rem; font-weight: 800; color: %(navy)s;
    }

    /* ---------- Section heading ---------- */
    .mc-section {
        font-family: 'Plus Jakarta Sans', %(font)s;
        font-size: 1.06rem; font-weight: 700; color: %(navy)s;
        margin: 28px 0 13px 0; padding-bottom: 8px;
        border-bottom: 1px solid %(border)s;
        display: flex; align-items: baseline; gap: 10px;
    }
    .mc-section::before {
        content: ""; width: 4px; height: 16px; border-radius: 3px;
        background: %(accent_grad)s; display: inline-block;
    }
    .mc-section small {
        font-weight: 500; color: %(text_muted)s; font-size: 0.79rem;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; border-bottom: 1px solid %(border)s; padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px; padding: 0 18px; font-weight: 600;
        font-size: 0.88rem; color: %(text_muted)s;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: %(navy)s;
        background: %(surface_muted)s;
    }

    /* ---------- Tables ---------- */
    [data-testid="stDataFrame"] {
        border-radius: 12px; border: 1px solid %(border)s; overflow: hidden;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%%, #080d18 100%%);
        border-right: 1px solid %(border)s;
    }
    section[data-testid="stSidebar"] .block-container {padding-top: 1.1rem;}
    section[data-testid="stSidebar"] h3 {
        font-family: 'Plus Jakarta Sans', %(font)s;
        color: %(navy)s;
    }
    .mc-side-title {
        font-family: 'Plus Jakarta Sans', %(font)s;
        font-size: 0.95rem; font-weight: 700; color: %(navy)s;
        margin: 4px 0 10px 0; display: flex; align-items: center; gap: 8px;
    }
    .mc-side-title::before {
        content: ""; width: 4px; height: 15px; border-radius: 3px;
        background: %(accent_grad)s; display: inline-block;
    }
    .mc-record {
        background: %(surface_muted)s; border: 1px solid %(border)s;
        border-radius: 11px; padding: 10px 13px; margin-bottom: 9px;
    }
    .mc-record-name {font-size: 0.86rem; font-weight: 700; color: %(navy)s;}
    .mc-record-meta {font-size: 0.74rem; color: %(text_muted)s; margin-top: 3px;}
    .mc-record-risk {font-size: 0.74rem; color: %(text)s; margin-top: 4px;}

    /* ---------- History table ---------- */
    .mc-th {
        display: block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; color: %(text_subtle)s;
        padding-bottom: 7px; border-bottom: 1px solid %(border)s; margin-bottom: 4px;
    }
    .mc-td {display: block; font-size: 0.85rem; color: %(text)s; padding: 3px 0;}
    .mc-spacer {height: 12px;}

    /* ---------- Buttons ---------- */
    .stDownloadButton button, .stButton button {
        border-radius: 10px; font-weight: 600;
        border: 1px solid %(border_strong)s;
        background: %(surface_muted)s; color: %(navy)s;
        transition: all 0.16s ease;
    }
    .stDownloadButton button:hover, .stButton button:hover {
        border-color: %(teal)s; color: %(teal)s;
    }
    .stButton button[kind="primary"] {
        background: %(accent_grad)s; color: #04121a; border: none;
        font-weight: 700;
    }
    .stButton button[kind="primary"]:hover {color: #04121a; filter: brightness(1.06);}

    /* ---------- Page heading ---------- */
    .mc-page-head {margin: 4px 0 14px 0;}
    .mc-page-title {
        font-family: 'Plus Jakarta Sans', %(font)s;
        font-size: 1.5rem; font-weight: 800; color: %(navy)s;
        margin: 0; letter-spacing: -0.02em;
    }
    .mc-page-sub {font-size: 0.88rem; color: %(text_muted)s; margin: 6px 0 0 0; line-height: 1.5;}

    /* ---------- Sidebar navigation ---------- */
    .mc-brand-mini {display: flex; align-items: center; gap: 12px; padding: 2px 2px 6px 2px;}
    .mc-brand-mini .mc-logo {width: 40px; height: 40px; font-size: 1rem; border-radius: 12px;}
    .mc-brand-mini-title {
        font-family: 'Plus Jakarta Sans', %(font)s;
        font-size: 1rem; font-weight: 800; color: %(navy)s; margin: 0; letter-spacing: -0.02em;
    }
    .mc-brand-mini-sub {font-size: 0.7rem; color: %(text_muted)s; margin: 2px 0 0 0;}
    .mc-nav-group {
        font-size: 0.66rem; font-weight: 700; letter-spacing: 0.14em;
        text-transform: uppercase; color: %(text_subtle)s; margin: 18px 0 8px 4px;
    }
    section[data-testid="stSidebar"] .stButton {margin-bottom: 3px;}
    section[data-testid="stSidebar"] .stButton button {
        display: flex; justify-content: flex-start; text-align: left;
        font-size: 0.86rem; padding: 8px 13px;
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {justify-content: flex-start;}
    .mc-nav-foot {
        margin-top: 20px; padding-top: 14px; border-top: 1px solid %(border)s;
        font-size: 0.74rem; color: %(text_subtle)s; line-height: 1.6;
    }

    /* ---------- Patient profile cards ---------- */
    .mc-profile-card {
        background: %(glass_high)s; border: 1px solid %(border)s; border-radius: 16px;
        padding: 17px 19px; height: 100%%; backdrop-filter: blur(12px);
        box-shadow: 0 12px 34px rgba(2, 6, 23, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.03);
        transition: border-color 0.18s ease;
    }
    .mc-profile-card:hover {border-color: %(border_strong)s;}
    .mc-profile-head {display: flex; align-items: center; gap: 12px; margin-bottom: 13px;}
    .mc-avatar {
        width: 46px; height: 46px; border-radius: 13px; flex: 0 0 auto;
        background: %(accent_grad)s; color: #04121a;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 1.02rem; letter-spacing: -0.02em;
    }
    .mc-profile-name {
        font-family: 'Plus Jakarta Sans', %(font)s;
        font-size: 0.99rem; font-weight: 700; color: %(navy)s; margin: 0;
    }
    .mc-profile-id {font-size: 0.75rem; color: %(text_muted)s; margin: 3px 0 0 0;}
    .mc-profile-row {
        display: flex; align-items: center; justify-content: space-between; gap: 10px;
        font-size: 0.82rem; color: %(text)s; padding: 6px 0;
        border-bottom: 1px solid %(border)s;
    }
    .mc-profile-row:last-child {border-bottom: none;}
    .mc-profile-row .mc-profile-key {color: %(text_muted)s;}

    /* ---------- Empty state ---------- */
    .mc-empty {
        background: %(surface_muted)s; border: 1px dashed %(border_strong)s;
        border-radius: 14px; padding: 30px 26px; text-align: center;
        color: %(text_muted)s; font-size: 0.9rem; line-height: 1.6;
    }
    .mc-empty b {color: %(navy)s;}

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar {width: 10px; height: 10px;}
    ::-webkit-scrollbar-track {background: transparent;}
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.25); border-radius: 6px;
    }
    ::-webkit-scrollbar-thumb:hover {background: rgba(148, 163, 184, 0.4);}
</style>
""" % {"font": FONT_STACK, **PALETTE}


def inject_theme():
    """Apply the global stylesheet. Safe to call once per script run."""
    st.markdown(_CSS, unsafe_allow_html=True)


def pill(status_key, text=None):
    meta = STATUS.get(status_key, STATUS["info"])
    return f'<span class="mc-pill mc-pill-{status_key}">{text or meta["label"]}</span>'


def tag(text, context=False):
    """Small neutral chip; ``context=True`` styles it as a context-only marker."""
    css = "mc-tag mc-tag-context" if context else "mc-tag"
    return f'<span class="{css}">{text}</span>'


def kpi_card(label, value, note="", accent=None):
    accent_style = f" style='color:{accent};'" if accent else ""
    note_html = f'<div class="mc-kpi-note">{note}</div>' if note else ""
    return (
        f'<div class="mc-kpi">'
        f'<div class="mc-kpi-label">{label}</div>'
        f'<div class="mc-kpi-value"{accent_style}>{value}</div>'
        f'{note_html}</div>'
    )


def card(title, caption, body_html):
    caption_html = f'<p class="mc-card-caption">{caption}</p>' if caption else ""
    return (
        f'<div class="mc-card">'
        f'<p class="mc-card-title">{title}</p>{caption_html}{body_html}</div>'
    )


def page_head(title, subtitle=""):
    """HTML for a page title block, rendered above the page content."""
    sub_html = f'<p class="mc-page-sub">{subtitle}</p>' if subtitle else ""
    return f'<div class="mc-page-head"><p class="mc-page-title">{title}</p>{sub_html}</div>'


def empty_state(message):
    """HTML for an informative empty state."""
    return f'<div class="mc-empty">{message}</div>'


def initials(name):
    """Two-letter monogram for avatar tiles."""
    parts = [part for part in str(name or "").replace("-", " ").split() if part]
    if not parts:
        return "PT"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def profile_card(name, patient_id, rows, title=None):
    """HTML for one patient profile card.

    ``rows`` is a list of ``(label, value_html)`` pairs rendered as a small
    key/value table under the patient header.
    """
    row_html = "".join(
        f'<div class="mc-profile-row"><span class="mc-profile-key">{label}</span>'
        f"<span>{value}</span></div>"
        for label, value in rows
    )
    return (
        f'<div class="mc-profile-card">'
        f'<div class="mc-profile-head">'
        f'<div class="mc-avatar">{initials(name)}</div>'
        f"<div><p class=\"mc-profile-name\">{name}</p>"
        f'<p class="mc-profile-id">{title or patient_id}</p></div>'
        f"</div>{row_html}</div>"
    )
