from __future__ import annotations

import joblib
import shap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards

from config import DATA_DIR, PLOTS_DIR, MODELS_DIR
from data import FEATURES

pio.templates.default = "plotly_dark"

CYBER_COLORS = ["#06B6D4", "#EF4444", "#F59E0B", "#8B5CF6", "#10B981", "#EC4899"]


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_DIR / 'twitter_clean.parquet')


@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / 'xgboost.joblib')


# Médianes calculées sur les comptes humains uniquement pour que les valeurs
# par défaut ne soient pas bot-skewed (58.6% du dataset sont des bots)
MEDIANS = {
    'statuses_count': 6609.0, 'followers_count': 341.0, 'friends_count': 319.0,
    'favourites_count': 1286.0, 'listed_count': 2.0, 'default_profile': 0.0,
    'default_profile_image': 0.0, 'geo_enabled': 1.0,
    'profile_use_background_image': 1.0, 'verified': 0.0,
    'name_length': 11.0, 'screen_name_length': 11.0, 'description_length': 51.0,
    'name_contains_bot': 0.0, 'screen_name_contains_bot': 0.0,
    'name_entropy': 3.0, 'screen_name_entropy': 3.0,
    'friends_followers_ratio': 0.974, 'lists_followers_ratio': 0.0061,
    'retweet_followers_ratio': 1.783, 'favorites_followers_ratio': 3.815,
    'retweet_status_ratio': 0.096, 'favorites_status_ratio': 0.251,
    'reply_status_ratio': 0.108, 'account_age': 83348.5,
    'followers_account_age_ratio': 0.0041, 'friends_account_age_ratio': 0.0038,
    'statuses_account_age_ratio': 0.078, 'favourites_account_age_ratio': 0.0158,
    'lists_account_age_ratio': 0.0,
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-left: 2rem; padding-right: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0D1117 !important; border-right: 1px solid #1F2937; }

/* Section header avec accent gauche */
.section-header {
    background: linear-gradient(90deg, rgba(6,182,212,0.08) 0%, transparent 100%);
    border-left: 3px solid #06B6D4;
    padding: 0.5rem 1rem;
    border-radius: 0 6px 6px 0;
    margin-bottom: 1rem;
}
.section-header span {
    color: #06B6D4;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: linear-gradient(135deg, #111827 0%, #1a2234 100%);
    border: 1px solid #1F2937;
    border-left: 3px solid #06B6D4;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
}
.kpi-card.danger { border-left-color: #EF4444; }
.kpi-card.warning { border-left-color: #F59E0B; }
.kpi-card.success { border-left-color: #10B981; }
.kpi-label { color: #6B7280; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0; }
.kpi-value { color: #F9FAFB; font-size: 1.9rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin: 4px 0 0; }

/* Glass card */
.glass-card {
    background: rgba(17, 24, 39, 0.8);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Result cards */
.result-bot {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.02));
    border: 1px solid rgba(239,68,68,0.3);
    border-top: 3px solid #EF4444;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.result-human {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(16,185,129,0.02));
    border: 1px solid rgba(16,185,129,0.3);
    border-top: 3px solid #10B981;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
}
.result-icon  { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; letter-spacing: 4px; }
.result-bot   .result-label { color: #EF4444; }
.result-human .result-label { color: #10B981; }
.result-score { font-size: 0.9rem; color: #6B7280; margin-top: 0.5rem; }
.result-score b { color: #E2E8F0; }

/* Signal items */
.signal { display: flex; align-items: flex-start; gap: 0.7rem; padding: 0.8rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; font-size: 0.88rem; }
.signal.danger { background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.2); }
.signal.ok     { background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.2); }
.signal .icon  { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.signal .text  { color: #D1D5DB; }

/* Input labels */
div[data-testid="stNumberInput"] label,
div[data-testid="stToggle"] label { color: #9CA3AF !important; font-size: 0.85rem !important; }

/* Button */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #0891B2, #0E7490) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 0.95rem !important; letter-spacing: 0.5px !important;
    padding: 0.65rem 2rem !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1F2937; border-radius: 8px; overflow: hidden; }
</style>
"""


def section(title: str):
    st.markdown(f'<div class="section-header"><span>{title}</span></div>', unsafe_allow_html=True)


def kpi(value, label, variant=""):
    return f'<div class="kpi-card {variant}"><p class="kpi-label">{label}</p><p class="kpi-value">{value}</p></div>'


def chart_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#9CA3AF", margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1F2937", showgrid=True),
        yaxis=dict(gridcolor="#1F2937", showgrid=True),
    )


def build_app() -> None:
    st.set_page_config(page_title="Twitter Bot Detector", layout="wide", page_icon="🛡️")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            '<div style="padding:1rem 0.5rem 0.5rem">'
            '<span style="font-family:JetBrains Mono,monospace;font-size:1rem;color:#06B6D4;font-weight:700">🛡️ BOT DETECTOR</span><br>'
            '<span style="font-size:0.65rem;color:#374151;letter-spacing:1.5px">TWITTER INTELLIGENCE</span>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown('<hr style="border-color:#1F2937;margin:0.5rem 0 1rem">', unsafe_allow_html=True)

        page = option_menu(
            menu_title=None,
            options=["Vue d'ensemble", "Données", "Analyser un compte", "Résultats"],
            icons=["grid-1x2-fill", "bar-chart-fill", "shield-check", "graph-up-arrow"],
            default_index=0,
            styles={
                "container": {"background-color": "transparent", "padding": "0"},
                "icon": {"color": "#4B5563", "font-size": "14px"},
                "nav-link": {
                    "font-size": "13px", "color": "#6B7280",
                    "border-radius": "6px", "margin": "2px 0",
                    "--hover-color": "#1F2937",
                },
                "nav-link-selected": {
                    "background-color": "rgba(6,182,212,0.1)",
                    "color": "#06B6D4", "font-weight": "600",
                    "border-left": "3px solid #06B6D4",
                },
            },
        )

    # ── VUE D'ENSEMBLE ────────────────────────────────────────────────────
    if page == "Vue d'ensemble":
        st.markdown(
            '<h1 style="font-size:2.2rem;font-weight:700;margin-bottom:0.2rem">Détection de bots Twitter</h1>'
            '<p style="color:#6B7280;font-size:1rem;margin-bottom:1.5rem">Classifier les comptes en <b style="color:#10B981">humains</b> ou <b style="color:#EF4444">bots</b> à partir de leurs métadonnées comportementales.</p>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="kpi-grid">'
            + kpi("8 386", "Comptes analysés")
            + kpi("58.6 %", "Taux de bots", "danger")
            + kpi("0.99", "F1-score XGBoost", "success")
            + kpi("30", "Features sélectionnées", "warning")
            + '</div>',
            unsafe_allow_html=True
        )

        section("Pourquoi détecter les bots ?")
        c1, c2, c3, c4 = st.columns(4)
        items = [
            ("🗳️", "Élections", "Armées de bots qui manipulent l'opinion publique et fabriquent du consensus artificiel."),
            ("📰", "Fake news", "Amplification automatisée de désinformation à des millions d'utilisateurs."),
            ("📈", "Tendances", "Hashtags propulsés artificiellement pour simuler un mouvement social."),
            ("💸", "Fraude", "Spam, phishing et faux influenceurs vendant de la visibilité."),
        ]
        for col, (icon, title, desc) in zip([c1, c2, c3, c4], items):
            col.markdown(
                f'<div class="glass-card">'
                f'<div style="font-size:1.5rem;margin-bottom:0.5rem">{icon}</div>'
                f'<div style="font-weight:600;color:#E2E8F0;margin-bottom:0.4rem">{title}</div>'
                f'<div style="font-size:0.82rem;color:#6B7280">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        section("Approche technique")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Dataset** — Zenodo DOI 10.5281/zenodo.5574403")
            st.markdown("8 386 comptes labellisés · 69 métriques · CC-BY 4.0 · sans API Twitter")
            st.markdown("</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Pipeline ML** — Classification supervisée binaire")
            st.markdown("3 modèles · Cross-validation 5 folds · StandardScaler · sklearn Pipeline")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── DONNÉES ───────────────────────────────────────────────────────────
    elif page == "Données":
        st.markdown('<h1 style="font-size:2rem;font-weight:700;margin-bottom:1rem">Exploration des données</h1>', unsafe_allow_html=True)
        df = load_data()
        df['Type'] = df['class_bot'].map({0: 'Humain', 1: 'Bot'})

        st.markdown(
            '<div class="kpi-grid">'
            + kpi(f"{len(df):,}", "Comptes total")
            + kpi(f"{df['class_bot'].sum():,}", "Bots", "danger")
            + kpi(f"{(df['class_bot']==0).sum():,}", "Humains", "success")
            + kpi(f"{df['class_bot'].mean()*100:.1f}%", "Taux de bots", "warning")
            + '</div>',
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2)
        with col_a:
            section("Distribution des followers (percentile 95)")
            df_f = df[df['followers_count'] < df['followers_count'].quantile(0.95)]
            fig = px.histogram(df_f, x='followers_count', color='Type',
                color_discrete_map={'Humain': '#06B6D4', 'Bot': '#EF4444'},
                nbins=50, barmode='overlay', opacity=0.75)
            fig.update_layout(**chart_layout())
            fig.update_layout(xaxis_title="Followers", yaxis_title="Comptes", legend_title="")
            st.plotly_chart(fig, use_container_width=True, key="followers_hist")

        col_c, col_d = st.columns(2)
        with col_c:
            section("Âge du compte (jours)")
            df_age = df.copy()
            df_age['age_jours'] = df_age['account_age'] / 24
            fig2 = px.box(df_age, x='Type', y='age_jours', color='Type',
                color_discrete_map={'Humain': '#06B6D4', 'Bot': '#EF4444'},
                labels={'age_jours': 'Âge (jours)'})
            fig2.update_layout(**chart_layout(), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, key="age_box")

        with col_d:
            section("Ratio abonnements / followers")
            df_r = df[df['friends_followers_ratio'] < df['friends_followers_ratio'].quantile(0.95)]
            fig3 = px.histogram(df_r, x='friends_followers_ratio', color='Type',
                color_discrete_map={'Humain': '#06B6D4', 'Bot': '#EF4444'},
                nbins=50, barmode='overlay', opacity=0.75)
            fig3.update_layout(**chart_layout(), legend_title="")
            st.plotly_chart(fig3, use_container_width=True, key="friends_ratio_hist")

    # ── ANALYSER UN COMPTE ────────────────────────────────────────────────
    elif page == "Analyser un compte":
        st.markdown(
            '<h1 style="font-size:2rem;font-weight:700;margin-bottom:0.3rem">Analyser un compte Twitter</h1>'
            '<p style="color:#6B7280;margin-bottom:1.5rem">Le modèle XGBoost (F1 = 0.99) analyse les métriques en temps réel.</p>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            section("Activité & Réseau")
            followers   = st.number_input("Followers", min_value=0, value=341, step=10)
            friends     = st.number_input("Abonnements (following)", min_value=0, value=319, step=10)
            statuses    = st.number_input("Tweets publiés", min_value=0, value=6609, step=50)
            favourites  = st.number_input("Likes donnés", min_value=0, value=1286, step=10)
            account_age = st.number_input("Âge du compte (jours)", min_value=1, value=3473, step=30)

        with col2:
            section("Profil")
            default_pic     = st.toggle("Photo de profil par défaut", value=False)
            default_profile = st.toggle("Mise en page profil par défaut", value=False)
            has_bio         = st.toggle("Le compte a une bio", value=True)
            name_has_bot    = st.toggle("Le nom contient 'bot'", value=False)
            verified        = st.toggle("Compte vérifié ✓", value=False)

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("⚡  LANCER L'ANALYSE", use_container_width=True, type="primary")

        if run:
            model = load_model()
            # account_age dans le modèle est en heures — l'utilisateur entre des jours
            account_age_h = account_age * 24
            inp = MEDIANS.copy()
            inp.update({
                'followers_count': followers, 'friends_count': friends,
                'statuses_count': statuses, 'favourites_count': favourites,
                'account_age': account_age_h,
                'default_profile_image': int(default_pic),
                'default_profile': int(default_profile),
                'description_length': 0 if not has_bio else 80,
                'name_contains_bot': int(name_has_bot),
                'screen_name_contains_bot': int(name_has_bot),
                'verified': int(verified),
                'friends_followers_ratio': friends / (followers + 1),
                'statuses_account_age_ratio': statuses / (account_age_h + 1),
                'followers_account_age_ratio': followers / (account_age_h + 1),
                'friends_account_age_ratio': friends / (account_age_h + 1),
                'favourites_account_age_ratio': favourites / (account_age_h + 1),
            })

            X     = pd.DataFrame([inp])[FEATURES]
            proba = model.predict_proba(X)[0]
            pred  = int(model.predict(X)[0])
            score = proba[1]

            st.divider()
            col_res, col_gauge = st.columns([1, 1])

            with col_res:
                if pred == 1:
                    st.markdown(
                        f'<div class="result-bot">'
                        f'<div class="result-icon">🤖</div>'
                        f'<div class="result-label">BOT DÉTECTÉ</div>'
                        f'<div class="result-score">Probabilité bot : <b>{score*100:.1f}%</b></div>'
                        f'</div>', unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-human">'
                        f'<div class="result-icon">👤</div>'
                        f'<div class="result-label">HUMAIN</div>'
                        f'<div class="result-score">Probabilité bot : <b>{score*100:.1f}%</b></div>'
                        f'</div>', unsafe_allow_html=True
                    )

            with col_gauge:
                color = "#EF4444" if pred == 1 else "#10B981"
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(score * 100, 1),
                    number={'suffix': '%', 'font': {'size': 38, 'color': color, 'family': 'JetBrains Mono'}},
                    title={'text': "Score bot", 'font': {'color': '#6B7280', 'size': 13}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickfont': {'color': '#4B5563'}, 'tickcolor': '#1F2937'},
                        'bar': {'color': color, 'thickness': 0.2},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'bordercolor': '#1F2937',
                        'steps': [
                            {'range': [0,  40], 'color': 'rgba(16,185,129,0.08)'},
                            {'range': [40, 60], 'color': 'rgba(245,158,11,0.08)'},
                            {'range': [60,100], 'color': 'rgba(239,68,68,0.08)'},
                        ],
                        'threshold': {'line': {'color': '#6B7280', 'width': 2}, 'value': 50}
                    }
                ))
                fig_g.update_layout(
                    height=230, margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor='rgba(0,0,0,0)', font_color='#9CA3AF'
                )
                st.plotly_chart(fig_g, use_container_width=True, key="gauge")

            section("Signaux détectés")
            signals = []
            if friends / (followers + 1) > 5:
                signals.append(("danger", f"Ratio following/followers = {friends/(followers+1):.1f} — seuil suspect > 5"))
            if default_pic:
                signals.append(("danger", "Photo de profil par défaut — absence d'identité visuelle"))
            if statuses / (account_age + 1) > 1.2:
                signals.append(("danger", f"Fréquence de publication anormale : {statuses/(account_age+1):.1f} tweets/jour"))
            if name_has_bot:
                signals.append(("danger", "Le nom du compte contient explicitement 'bot'"))
            if not has_bio:
                signals.append(("danger", "Aucune biographie — profil incomplet"))
            if not signals:
                signals.append(("ok", "Aucun signal suspect — profil cohérent avec un compte humain"))

            html = ""
            for kind, msg in signals:
                icon = "⚠️" if kind == "danger" else "✅"
                html += f'<div class="signal {kind}"><span class="icon">{icon}</span><span class="text">{msg}</span></div>'
            st.markdown(html, unsafe_allow_html=True)

            verdict = "🤖 BOT" if pred == 1 else "👤 HUMAIN"
            verdict_color = "#EF4444" if pred == 1 else "#10B981"
            section(f"Pourquoi le modèle dit {verdict} ?")

            xgb_pipeline = load_model()
            explainer = shap.TreeExplainer(xgb_pipeline.named_steps['model'])
            X_scaled = xgb_pipeline.named_steps['scaler'].transform(X)
            shap_vals = explainer.shap_values(X_scaled)[0]

            # Top 10 features par valeur absolue
            feat_contrib = pd.DataFrame({
                'Feature': FEATURES,
                'Valeur': X.iloc[0].values,
                'Contribution': shap_vals,
            })
            feat_contrib = feat_contrib.reindex(
                feat_contrib['Contribution'].abs().sort_values(ascending=True).tail(10).index
            )
            feat_contrib['Direction'] = feat_contrib['Contribution'].apply(
                lambda v: '→ bot' if v > 0 else '→ humain'
            )
            feat_contrib['Label'] = feat_contrib.apply(
                lambda r: f"{r['Feature']}  =  {r['Valeur']:.2f}", axis=1
            )

            bar_colors = ['#EF4444' if v > 0 else '#06B6D4' for v in feat_contrib['Contribution']]

            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                x=feat_contrib['Contribution'],
                y=feat_contrib['Label'],
                orientation='h',
                marker_color=bar_colors,
                marker_line_width=0,
                hovertemplate='<b>%{y}</b><br>Contribution SHAP : %{x:.4f}<extra></extra>',
            ))
            fig_shap.add_vline(x=0, line_color='#4B5563', line_width=1.5)
            fig_shap.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#9CA3AF',
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                height=380,
                title=dict(
                    text=f'Top 10 features · <span style="color:{verdict_color}">{verdict}</span> à {score*100:.1f}%',
                    font=dict(size=13, color='#E2E8F0'),
                ),
                xaxis=dict(gridcolor='#1F2937', zeroline=False,
                           title='← pousse vers humain   |   pousse vers bot →',
                           title_font=dict(size=11, color='#6B7280')),
                yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=11)),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_shap, use_container_width=True, key="shap_bar")

            col_leg1, col_leg2, _ = st.columns([1, 1, 2])
            col_leg1.markdown('<span style="color:#EF4444;font-size:0.85rem">■ Pousse vers BOT</span>', unsafe_allow_html=True)
            col_leg2.markdown('<span style="color:#06B6D4;font-size:0.85rem">■ Pousse vers HUMAIN</span>', unsafe_allow_html=True)

    # ── RÉSULTATS ─────────────────────────────────────────────────────────
    elif page == "Résultats":
        st.markdown('<h1 style="font-size:2rem;font-weight:700;margin-bottom:1rem">Performance des modèles</h1>', unsafe_allow_html=True)

        st.markdown('<p style="color:#4B5563;font-size:0.85rem;font-style:italic">Évaluation sur 20% du dataset non vu pendant l\'entraînement (jeu de test holdout)</p>', unsafe_allow_html=True)

        section("Pourquoi XGBoost — le moins de comptes humains bannis à tort")
        col1, col2, col3 = st.columns(3)
        fp_data = [
            ("Régression Logistique", 36, 23, "#F59E0B"),
            ("Random Forest",          4, 17, "#8B5CF6"),
            ("XGBoost",                4, 16, "#10B981"),
        ]
        for col, (name, fp, fn, color) in zip([col1, col2, col3], fp_data):
            col.markdown(
                f'<div class="glass-card" style="border-left:3px solid {color};text-align:center">'
                f'<div style="color:#9CA3AF;font-size:0.75rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.5rem">{name}</div>'
                f'<div style="font-size:2rem;font-weight:700;color:#EF4444;font-family:JetBrains Mono,monospace">{fp}</div>'
                f'<div style="color:#6B7280;font-size:0.78rem;margin-bottom:.8rem">humains bannis à tort</div>'
                f'<div style="font-size:1.4rem;font-weight:600;color:#F59E0B;font-family:JetBrains Mono,monospace">{fn}</div>'
                f'<div style="color:#6B7280;font-size:0.78rem">bots non détectés</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown(
            '<p style="color:#6B7280;font-size:0.82rem;margin-top:.5rem">'
            'Un faux positif = un humain détecté comme bot et banni à tort. '
            'XGBoost et Random Forest commettent 9× moins d\'erreurs que la Régression Logistique sur ce critère. '
            'XGBoost est retenu car il détecte également 1 bot supplémentaire.</p>',
            unsafe_allow_html=True
        )

        section("Indicateurs clés — XGBoost")
        st.image(str(PLOTS_DIR / 'feature_importance.png'), use_container_width=True)

        st.divider()
        st.markdown(
            '<h2 style="font-size:1.5rem;font-weight:700;margin:1.5rem 0 0.5rem">Ce que le modèle a appris</h2>'
            '<p style="color:#6B7280;font-size:0.88rem;margin-bottom:1.5rem">'
            'Quels indicateurs le modèle juge-t-il vraiment utiles, et comment les utilise-t-il ?</p>',
            unsafe_allow_html=True
        )

        col5, col6 = st.columns([1, 1])
        with col5:
            section("Importance des indicateurs (permutation)")
            st.image(str(PLOTS_DIR / 'permutation_importance.png'), use_container_width=True)
            st.markdown(
                '<p style="color:#6B7280;font-size:0.8rem">'
                'On mélange aléatoirement les valeurs d\'un indicateur et on mesure la chute de précision. '
                'Plus la chute est grande, plus cet indicateur est indispensable au modèle.</p>',
                unsafe_allow_html=True
            )
        with col6:
            section("PDP / ICE centré — effet marginal des variables clés")
            st.image(str(PLOTS_DIR / 'pdp_single_feature.png'), use_container_width=True)
            st.markdown(
                '<p style="color:#6B7280;font-size:0.8rem">'
                '<b style="color:#EF4444">■</b> ICE bots &nbsp;·&nbsp; '
                '<b style="color:#06B6D4">■</b> ICE humains &nbsp;·&nbsp; '
                '<b style="color:#F59E0B">—</b> Tendance globale &nbsp;·&nbsp; '
                'Zones colorées = zone dominante bot (rouge) / humain (bleu).<br>'
                'Calculé sur le modèle XGBoost complet. '
                'La courbe ne montre pas la probabilité brute, '
                'mais <b>comment la prédiction varie quand la feature change</b>, '
                'les 29 autres features maintenues à leur valeur réelle.</p>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div style="background:linear-gradient(90deg,rgba(6,182,212,0.06) 0%,transparent 100%);'
            'border-left:3px solid #06B6D4;border-radius:0 8px 8px 0;padding:1rem 1.5rem;margin-top:1rem">'
            '<p style="margin:0;font-size:0.9rem;color:#E2E8F0">'
            '<b style="color:#06B6D4">Comment lire ces graphiques</b><br>'
            '• <b>Axe vertical</b> : variation de la probabilité bot par rapport au point de départ (centré à 0). '
            'Une valeur négative = la feature pousse vers "humain".<br>'
            '• <b>Zone rouge</b> : là où les bots dominent — ratio faible, peu d\'activité.<br>'
            '• <b>Zone bleue</b> : là où les humains dominent — ratio plus élevé, comportement actif.<br>'
            '• <b>Ligne verticale dorée</b> : seuil de transition entre les deux zones.<br>'
            '• <b>Amplitude faible</b> : normale sur un modèle à 99% — le modèle décide par combinaison '
            'de 30 indicateurs, pas sur une seule variable isolée.'
            '</p></div>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    build_app()
