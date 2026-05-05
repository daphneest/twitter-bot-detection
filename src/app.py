from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import MODEL_METRICS_FILE, DATA_DIR, PLOTS_DIR


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_DIR / 'twitter_clean.parquet')


def build_app() -> None:
    st.set_page_config(page_title="Détection de bots Twitter", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Contexte", "Données", "Modèles", "Résultats"])

    if page == "Contexte":
        st.title("Détection de bots sur Twitter / X")
        st.markdown("""
        ### Objectif
        Classifier automatiquement les comptes Twitter en **humains** ou **bots**,
        à partir de leurs métadonnées publiques uniquement — sans accès à l'API.

        ### Pourquoi c'est important ?
        | Problème | Impact |
        |---|---|
        | Manipulation de l'opinion | Fausses tendances (hashtags artificiels) |
        | Désinformation | Amplification massive de fake news |
        | Spam & phishing | Millions de comptes frauduleux |
        | Élections | Ingérence et campagnes coordonnées |

        ### Dataset
        - **Source** : Zenodo 10.5281/zenodo.5574403
        - **Volume** : 8 386 comptes Twitter labellisés
        - **Features** : 69 métriques comportementales (activité, réseau, identité, temporel)
        - **Licence** : CC-BY 4.0
        """)

    elif page == "Données":
        df = load_data()
        st.title("Exploration des données")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Comptes total", f"{len(df):,}")
        col2.metric("Bots", f"{df['class_bot'].sum():,}")
        col3.metric("Humains", f"{(df['class_bot']==0).sum():,}")
        col4.metric("Taux de bots", f"{df['class_bot'].mean()*100:.1f}%")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Distribution : followers_count")
            fig = px.histogram(
                df[df['followers_count'] < df['followers_count'].quantile(0.95)],
                x='followers_count', color=df['class_bot'].map({0:'Humain',1:'Bot'}),
                color_discrete_map={'Humain':'steelblue','Bot':'crimson'},
                nbins=50, barmode='overlay', opacity=0.7
            )
            fig.update_layout(xaxis_title="Followers", yaxis_title="Comptes", legend_title="Type")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Âge du compte (jours)")
            fig2 = px.box(
                df, x=df['class_bot'].map({0:'Humain',1:'Bot'}), y='account_age',
                color=df['class_bot'].map({0:'Humain',1:'Bot'}),
                color_discrete_map={'Humain':'steelblue','Bot':'crimson'}
            )
            fig2.update_layout(xaxis_title="", yaxis_title="Âge (jours)", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Ratio friends/followers — signal discriminant clé")
        fig3 = px.histogram(
            df[df['friends_followers_ratio'] < df['friends_followers_ratio'].quantile(0.95)],
            x='friends_followers_ratio',
            color=df['class_bot'].map({0:'Humain',1:'Bot'}),
            color_discrete_map={'Humain':'steelblue','Bot':'crimson'},
            nbins=60, barmode='overlay', opacity=0.7
        )
        st.plotly_chart(fig3, use_container_width=True)

    elif page == "Modèles":
        st.title("Comparaison des modèles")
        st.markdown("""
        3 modèles supervisés entraînés et comparés en **cross-validation 5 folds**.
        Label : 0 = compte humain, 1 = bot.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**Régression Logistique**\nModèle linéaire de base. Rapide et interprétable.")
        with col2:
            st.info("**Random Forest**\nEnsemble d'arbres. Robuste aux valeurs aberrantes.")
        with col3:
            st.info("**XGBoost**\nGradient boosting. Meilleur modèle sélectionné.")

        st.subheader("Features — 4 catégories")
        st.markdown("""
        | Catégorie | Features |
        |---|---|
        | Activité | statuses_count, favourites_count, listed_count |
        | Réseau | followers_count, friends_count, friends_followers_ratio |
        | Identité | name_entropy, default_profile, description_length, name_contains_bot |
        | Temporel | account_age, statuses_account_age_ratio, followers_account_age_ratio |
        """)

    elif page == "Résultats":
        st.title("Résultats")

        if MODEL_METRICS_FILE.exists():
            st.subheader("Métriques sur le jeu de test")
            st.dataframe(pd.read_csv(MODEL_METRICS_FILE), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Radar des métriques")
            st.image(str(PLOTS_DIR / 'radar_metriques.png'), use_container_width=True)
        with col2:
            st.subheader("Comparaison CV 5 folds")
            st.image(str(PLOTS_DIR / 'comparaison_modeles.png'), use_container_width=True)

        st.subheader("Courbes ROC")
        st.image(str(PLOTS_DIR / 'roc_curves.png'), use_container_width=True)

        st.subheader("Matrices de confusion")
        st.image(str(PLOTS_DIR / 'confusion_matrices.png'), use_container_width=True)

        st.subheader("Importance des features — XGBoost")
        st.image(str(PLOTS_DIR / 'feature_importance.png'), use_container_width=True)


if __name__ == "__main__":
    build_app()
