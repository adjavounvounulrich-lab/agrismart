# =============================================================================
# AGRISMART — Application Streamlit de recommandation de culture agricole
# =============================================================================
# Auteurs     : Équipe AgroSmart
# Modèle      : XGBoost + SHAP
# Interface   : Streamlit
# Version     : 3.0 (Waterfall SHAP + Recontextualisation Bénin)
# =============================================================================
#
# STRUCTURE DU FICHIER :
#   1.  Importation des bibliothèques
#   2.  Configuration de la page
#   3.  CSS global
#   4.  Chargement du modèle ML réel
#   5.  Session State (mémoire de l'app)
#   6.  Constantes + correspondances (cultures et saisons béninoises)
#   7.  Sidebar + navigation
#   8.  show_home()      → Page d'accueil
#   9.  show_analysis()  → Page analyse + SHAP waterfall
#   10. show_chat()      → Page chat (prêt pour RASA)
#   11. show_history()   → Page historique
#   12. Routeur principal
# =============================================================================


# =============================================================================
# SECTION 1 — IMPORTATION DES BIBLIOTHÈQUES
# Chaque bibliothèque a un rôle précis dans l'application.
# =============================================================================

import streamlit as st         # Framework principal : transforme ce script en app web
import joblib                  # Charge les fichiers .pkl (modèle, encodeur, scaler)
import numpy as np             # Tableaux numériques pour la prédiction
import pandas as pd            # Tableaux de données pour l'historique
import shap                    # Calcul des valeurs SHAP (explicabilité du modèle)
import matplotlib.pyplot as plt # Nécessaire pour afficher le waterfall SHAP
                                # dans Streamlit via st.pyplot()
import random                  # Utilisé uniquement pour la simulation de secours
import xgboost                 # Nécessaire pour que joblib puisse reconstruire
                               # le modèle XGBoost depuis le fichier model.pkl.
                               # Sans cette bibliothèque installée, le chargement
                               # du modèle échouerait.
import time                    # Nécessaire pour l'effet de frappe dans le chat
import re                      # Nécessaire pour découper le texte en tokens (frappe)
from datetime import datetime  # Horodatage des analyses dans l'historique


# =============================================================================
# SECTION 2 — CONFIGURATION DE LA PAGE
# Cette commande DOIT être la toute première instruction Streamlit du fichier.
# Si elle n'est pas en première position, Streamlit génère une erreur fatale.
# =============================================================================

st.set_page_config(
    page_title="AgriSmart",           # Titre de l'onglet du navigateur
    page_icon="🌿",                   # Icône de l'onglet
    layout="wide",                    # Contenu sur toute la largeur disponible
    initial_sidebar_state="expanded"  # Sidebar ouverte par défaut
)


# =============================================================================
# SECTION 3 — CSS GLOBAL
# Tout le style visuel de l'application est regroupé ici en un seul bloc.
# Regrouper le CSS évite les conflits entre styles dispersés dans le code.
#
# RÈGLE IMPORTANTE : dans un st.markdown(), ne jamais mettre de commentaires
# HTML <!-- --> car Streamlit les affiche comme du texte brut.
# On commente uniquement avec # en Python, en dehors des guillemets.
# =============================================================================

st.markdown("""
<style>

/* Fond général de l'application : blanc cassé très léger */
.stApp {
    background-color: #f8fafc;
}

/* Sidebar : fond vert foncé, largeur fixe */
[data-testid="stSidebar"] {
    background-color: #1B4332 !important;
    min-width: 225px !important;
    max-width: 225px !important;
}

/* Boutons de navigation dans la sidebar :
   on leur retire fond et bordure pour qu'ils ressemblent
   à de vrais liens de menu et non à des boutons classiques */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: rgba(255,255,255,0.65) !important;
    text-align: left !important;
    width: 100% !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    transition: all 0.2s ease !important;
}

/* Survol : fond légèrement éclairé */
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #d1fae5 !important;
    transform: translateX(3px) !important;
}

/* Bouton actif (page actuellement ouverte) : fond illuminé */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(255,255,255,0.15) !important;
    color: #d1fae5 !important;
    font-weight: 600 !important;
    border-left: 3px solid #6ee7b7 !important;
}

/* Zone de contenu principale : largeur limitée pour une bonne lisibilité */
.block-container {
    max-width: 900px !important;
    padding-top: 0.5rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
}

/* Topbar : barre de titre en haut de chaque page */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 0;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 28px;
}
.topbar-title {
    font-size: 17px;
    font-weight: 700;
    color: #1a202c;
}
.topbar-badge {
    font-size: 11px;
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    color: #065f46;
    padding: 5px 14px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.03em;
}

/* Titre de section principal */
.section-title {
    font-size: 17px;
    font-weight: 700;
    color: #1B4332;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid #d1fae5;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Sous-titre de section */
.sous-titre {
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Carte blanche générique avec ombre douce */
.carte {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}

/* Carte de résultat de prédiction : fond vert clair */
.result-card {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #86efac;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 4px 16px rgba(45,106,79,0.12);
}
.result-emoji  { font-size: 64px; display: block; margin-bottom: 12px; }
.result-name   { font-size: 28px; font-weight: 800; color: #1B4332; }
.result-badge  {
    display: inline-block;
    background: #1B4332;
    color: white;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    margin-top: 8px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.result-desc   { font-size: 13px; color: #4B5563; margin-top: 10px; }

/* Cartes de la page d'accueil */
.hero {
    position: relative;
    border-radius: 18px;
    overflow: hidden;
    margin-bottom: 28px;
    height: 300px;
}
.hero-img {
    width: 100%; height: 100%;
    object-fit: cover; display: block;
    filter: brightness(0.50);
}
.hero-overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 0 48px;
}
.hero-title {
    font-size: 34px; font-weight: 800;
    color: white; margin: 0 0 10px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.5);
}
.hero-sub {
    font-size: 15px; color: rgba(255,255,255,0.9);
    margin: 0 0 22px; max-width: 500px;
    text-shadow: 0 1px 6px rgba(0,0,0,0.4);
    line-height: 1.6;
}

/* Cartes de statistiques (page d'accueil) */
.stat-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: transform 0.2s ease;
}
.stat-icon  { font-size: 32px; margin-bottom: 10px; }
.stat-value { font-size: 28px; font-weight: 800; color: #1B4332; }
.stat-label { font-size: 12px; color: #6B7280; margin-top: 4px; }

/* Cartes des étapes "Comment ça marche" */
.step-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.step-img {
    width: 100%; height: 145px;
    object-fit: cover; display: block;
}
.step-body { padding: 16px 18px; }
.step-num {
    width: 26px; height: 26px;
    border-radius: 50%;
    background: #1B4332; color: white;
    font-size: 12px; font-weight: 800;
    display: inline-flex; align-items: center;
    justify-content: center; margin-bottom: 10px;
}
.step-title { font-size: 13px; font-weight: 700; color: #111827; margin-bottom: 5px; }
.step-desc  { font-size: 12px; color: #6B7280; line-height: 1.5; }

/* Section contact */
.contact-section {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #bbf7d0;
    border-radius: 16px;
    padding: 30px 36px;
    margin-top: 8px;
}

/* Masque le séparateur rouge par défaut de Streamlit dans la sidebar */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12);
}

/* Style des messages du chat.
   line-height: 1.7 et word-break: break-word garantissent un rendu
   propre même sur les messages longs avec retours à la ligne (<br>). */
.chat-bot {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 0 14px 14px 14px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-size: 13px;
    color: #1B4332;
    max-width: 80%;
    line-height: 1.7;
    word-break: break-word;
}
.chat-user {
    background: #1B4332;
    border-radius: 14px 0 14px 14px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-size: 13px;
    color: white;
    max-width: 80%;
    margin-left: auto;
    text-align: right;
    line-height: 1.7;
}

/* Titre de section générique réutilisable */
.sh {
    font-size: 18px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 18px;
}

</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 4 — CHARGEMENT DU MODÈLE ML RÉEL
#
# @st.cache_resource est un décorateur Streamlit qui dit :
# "Charge ces fichiers une seule fois au démarrage, garde-les en mémoire,
#  et ne les recharge JAMAIS sauf si l'app redémarre."
# Sans ce décorateur, le modèle serait rechargé à chaque clic → très lent.
#
# La fonction retourne (None, None, None, None) en cas d'erreur.
# Dans ce cas, l'app bascule automatiquement sur la simulation.
# =============================================================================

@st.cache_resource
def charger_outils():
    """
    Charge les 4 fichiers .pkl nécessaires à la prédiction réelle.

    Retourne :
        modele    : le modèle XGBoost entraîné
        encoder_x : OrdinalEncoder + ColumnTransformer pour les variables catégorielles
        encoder_y : LabelEncoder pour décoder la culture prédite
        scaler_x  : StandardScaler pour normaliser les données numériques

    Si un fichier est manquant, retourne (None, None, None, None)
    et l'application utilisera la simulation à la place.
    """
    try:
        modele    = joblib.load("model.pkl")
        encoder_x = joblib.load("encoder_x.pkl")
        encoder_y = joblib.load("encoder_y.pkl")
        scaler_x  = joblib.load("scaler_x.pkl")
        return modele, encoder_x, encoder_y, scaler_x
    except FileNotFoundError:
        # Les fichiers .pkl ne sont pas trouvés → mode simulation activé
        return None, None, None, None


# Appel du chargement — les 4 outils sont disponibles dans toute l'app
modele, encoder_x, encoder_y, scaler_x = charger_outils()

# Indicateur booléen : True si le modèle réel est disponible
MODELE_REEL_DISPONIBLE = (modele is not None)


# =============================================================================
# SECTION 5 — SESSION STATE (MÉMOIRE DE L'APPLICATION)
#
# Streamlit réexécute TOUT le script du début à la fin à chaque interaction
# (clic, slider, saisie). Sans Session State, toutes les données seraient
# perdues à chaque interaction.
#
# st.session_state est un dictionnaire persistant qui survit aux réexécutions.
# "if not in" vérifie si la variable existe déjà avant de la créer,
# car si on la recrée à chaque fois, elle serait remise à zéro à chaque clic.
# =============================================================================

# Mémorise quelle page est actuellement affichée
if "page_active" not in st.session_state:
    st.session_state.page_active = "Accueil"

# Mémorise le résultat de la dernière analyse (culture + SHAP + paramètres)
if "derniere_analyse" not in st.session_state:
    st.session_state.derniere_analyse = None

# Mémorise toutes les analyses effectuées pendant la session
if "historique" not in st.session_state:
    st.session_state.historique = []

# Mémorise les messages du chat (liste de dicts {role, content})
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Mémorise si le message SHAP automatique a déjà été affiché dans le chat
if "chat_shap_affiche" not in st.session_state:
    st.session_state.chat_shap_affiche = False

# Mémorise si le dernier message bot doit être animé (effet de frappe).
# True = animer le prochain message bot affiché.
# Remis à False après l'animation pour éviter de rejouer sur les reruns suivants.
if "animer_dernier_message" not in st.session_state:
    st.session_state.animer_dernier_message = False


# =============================================================================
# SECTION 6 — CONSTANTES, CORRESPONDANCES ET FONCTIONS DE PRÉDICTION
#
# RECONTEXTUALISATION BÉNIN :
# Le modèle a été entraîné avec des noms de cultures et de saisons
# qui ne correspondent pas au contexte béninois (Blé, Orge, Millet,
# Kharif, Rabi, Zaid). On effectue ici une correspondance transparente :
#   - L'utilisateur voit et interagit avec les réalités béninoises
#   - Le modèle reçoit toujours les valeurs exactes de son entraînement
#   - Tout se passe en arrière-plan, invisible pour l'utilisateur
#
# On sépare deux modes de prédiction :
#   A) predire_avec_modele_reel() : utilise XGBoost + SHAP + encodeurs réels
#   B) simuler_prediction()       : génère un résultat aléatoire réaliste
#      (utile pendant le développement ou si les .pkl sont absents)
#
# La fonction principale predire() choisit automatiquement entre A et B.
# =============================================================================

# =============================================================================
# CORRESPONDANCE SAISONS BÉNIN → CODE MODÈLE
# L'utilisateur voit les vraies saisons du Bénin.
# En arrière-plan, on traduit vers les codes que le modèle connaît.
# Kharif = saison humide/pluies | Rabi = saison sèche | Zaid = intermédiaire
# =============================================================================

CORRESPONDANCE_SAISONS = {
    # Sud-Bénin — Climat Subéquatorial (4 saisons)
    "🌧️ Grande saison des pluies — Sud (Avr. à mi-Juil.)":  "Kharif",
    "☀️ Petite saison sèche — Sud (mi-Juil. à mi-Sep.)":    "Zaid",
    "🌦️ Petite saison des pluies — Sud (mi-Sep. à Oct.)":   "Kharif",
    "🌵 Grande saison sèche — Sud (Nov. à Mars)":           "Rabi",
    # Nord-Bénin — Climat Soudanien (2 saisons)
    "🌧️ Saison des pluies — Nord (Mai à Oct.)":             "Kharif",
    "🌵 Saison sèche — Nord (Nov. à Avr.)":                 "Rabi",
}

# =============================================================================
# CORRESPONDANCE CULTURES MODÈLE → CULTURES BÉNINOISES
# Mapping interne uniquement — totalement invisible pour l'utilisateur.
# La culture affichée est directement béninoise, sans aucune mention
# des noms d'origine utilisés pendant l'entraînement.
# =============================================================================

CORRESPONDANCE_CULTURES = {
    # Blé → Sorgho : même profil céréalier en conditions sèches
    "Blé":   "Sorgho",
    # Orge → Fonio : même tolérance aux sols pauvres, cycle court, Atacora
    "Orge":  "Fonio",
    # Millet → Mil : même plante, terminologie locale béninoise
    "Millet":"Mil",
}

# Liste des cultures affichées à l'utilisateur (noms béninois)
CULTURES = [
    "Riz", "Maïs", "Sorgho", "Coton", "Canne à sucre",
    "Fonio", "Mil", "Pomme de terre", "Légumineuses", "Tomate"
]

# Emojis associés à chaque culture pour l'affichage visuel
EMOJIS_CULTURES = {
    "Riz":           "🌾",
    "Maïs":          "🌽",
    "Sorgho":        "🌾",
    "Coton":         "🌿",
    "Canne à sucre": "🎋",
    "Fonio":         "🌿",
    "Mil":           "🌾",
    "Pomme de terre":"🥔",
    "Légumineuses":  "🫘",
    "Tomate":        "🍅",
}

# Descriptions courtes de chaque culture — contextualisées pour le Bénin
DESCRIPTIONS_CULTURES = {
    "Riz":           "Céréale semi-aquatique, idéale pour les zones humides du Bénin",
    "Maïs":          "Première céréale du Bénin, adaptée aux zones tropicales",
    "Sorgho":        "Céréale résistante à la sécheresse, cultivée dans le Nord-Bénin",
    "Coton":         "Culture de rente stratégique, pilier de l'économie béninoise",
    "Canne à sucre": "Culture sucrière tropicale, exige beaucoup d'eau",
    "Fonio":         "Céréale ancestrale africaine, cultivée dans l'Atacora (Nord-Bénin)",
    "Mil":           "Céréale semi-aride des zones Nord-Bénin (Borgou, Atacora, Alibori)",
    "Pomme de terre":"Tubercule productif, préfère les sols frais et bien drainés",
    "Légumineuses":  "Soja, Niébé ou Arachide selon votre zone — cultures majeures au Bénin",
    "Tomate":        "Culture maraîchère à haute valeur ajoutée, cultivée partout au Bénin",
}

# Noms des features dans l'ordre exact de X_combined
# Numériques en premier, puis catégorielles encodées
# Cet ordre correspond exactement à l'ordre du scaler pendant l'entraînement :
# Azote, Phosphore, Potassium, pH_du_sol, Temperature, Humidite_de_l_air,
# Pluviometrie, Vitesse_du_vent, Altitude, Engrais_utilise,
# Type_de_sol, Region, Saison, Culture_precedente
NOMS_FEATURES = [
    "Azote (N)", "Phosphore (P)", "Potassium (K)", "pH du sol",
    "Température", "Humidité", "Pluviométrie", "Vent",
    "Altitude", "Engrais",
    "Type de sol", "Région", "Saison", "Culture préc."
]


def simuler_prediction(params):
    """
    Génère une prédiction simulée réaliste pour le développement.

    Cette fonction est utilisée quand les fichiers .pkl sont absents.
    Sa structure est identique à predire_avec_modele_reel() pour faciliter
    le remplacement ultérieur.

    Paramètres :
        params (dict) : dictionnaire des valeurs saisies par l'utilisateur

    Retourne :
        dict avec keys : culture, emoji, description, confiance, shap_1d,
                         shap_explanation (None en simulation), mode
    """
    # Sélection aléatoire d'une culture dans la liste béninoise officielle
    culture = random.choice(CULTURES)

    # Simulation d'un score de confiance entre 72% et 97%
    confiance = round(random.uniform(0.72, 0.97), 2)

    # Génération de valeurs SHAP simulées mais réalistes
    # (nombres flottants positifs et négatifs comme de vrais SHAP)
    np.random.seed(42)
    shap_1d = np.random.uniform(-0.5, 0.8, len(NOMS_FEATURES))

    return {
        "culture":          culture,
        "emoji":            EMOJIS_CULTURES.get(culture, "🌱"),
        "description":      DESCRIPTIONS_CULTURES.get(culture, "Culture adaptée à votre parcelle"),
        "confiance":        confiance,
        "shap_1d":          shap_1d,
        "shap_explanation": None,   # Pas disponible en mode simulation
        "mode":             "simulation"
    }


def predire_avec_modele_reel(params, type_sol, region, saison, culture_prec):
    """
    Effectue la prédiction réelle avec XGBoost + SHAP.

    Pipeline exact :
        1. Construire DataFrame catégoriel avec les bons noms de colonnes
        2. Encoder les catégorielles avec OrdinalEncoder
        3. Combiner numériques + catégorielles encodées (numériques en premier)
        4. Normaliser avec StandardScaler
        5. Prédire avec XGBoost
        6. Calculer les probabilités pour le score de confiance
        7. Appliquer la correspondance béninoise (transparente pour l'utilisateur)
        8. Calculer les valeurs SHAP avec TreeExplainer
        9. Construire l'objet shap.Explanation pour le waterfall

    RECONTEXTUALISATION :
        Après la prédiction, si le modèle retourne Blé/Orge/Millet,
        on remplace silencieusement par Sorgho/Fonio/Mil.
        L'utilisateur ne voit jamais les noms d'origine.

    INTÉGRATION MODÈLE RÉEL :
        Si tu veux changer de modèle (RandomForest, LightGBM...),
        tu n'as à modifier que cette fonction. Le reste de l'app ne change pas.

    INTÉGRATION SHAP RÉEL :
        shap.TreeExplainer fonctionne avec XGBoost, LightGBM, RandomForest.
        Pour un modèle non-arbre (SVM, réseau de neurones), utilise
        shap.KernelExplainer à la place.

    Paramètres :
        params (dict) : valeurs numériques (N, P, K, pH, etc.)
        type_sol, region, saison, culture_prec (str) : variables catégorielles
        Note : saison est déjà traduit (Kharif/Rabi/Zaid) par CORRESPONDANCE_SAISONS

    Retourne :
        dict avec keys : culture, emoji, description, confiance,
                         shap_1d, shap_explanation, mode
    """
    # ÉTAPE 1 — Construire le DataFrame catégoriel
    # On utilise un DataFrame (pas un numpy array) car l'OrdinalEncoder
    # a été entraîné avec des noms de colonnes — il les exige impérativement.
    # Les noms doivent être EXACTEMENT identiques au dataset d'entraînement.
    X_cat = pd.DataFrame(
        [[type_sol, region, saison, culture_prec]],
        columns=["Type_de_sol", "Region", "Saison", "Culture_precedente"]
    )

    # ÉTAPE 2 — Construire le tableau numérique
    # L'ordre des colonnes doit correspondre à l'ordre du dataset d'entraînement :
    # Azote, Phosphore, Potassium, pH_du_sol, Temperature, Humidite_de_l_air,
    # Pluviometrie, Vitesse_du_vent, Altitude, Engrais_utilise
    X_num = np.array([[
        params["N"],           # Azote
        params["P"],           # Phosphore
        params["K"],           # Potassium
        params["ph"],          # pH du sol
        params["temperature"], # Température
        params["humidite"],    # Humidité de l'air
        params["pluvio"],      # Pluviométrie
        params["vent"],        # Vitesse du vent
        params["altitude"],    # Altitude
        params["engrais"]      # Engrais utilisé
    ]])

    # ÉTAPE 3 — Encoder les variables catégorielles
    # OrdinalEncoder transforme ex: "Limoneux" → 2.0 (indice dans l'ordre appris)
    X_cat_enc = encoder_x.transform(X_cat)

    # ÉTAPE 4 — Combiner numériques + catégorielles encodées
    # np.hstack colle les deux tableaux côte à côte horizontalement.
    # Les NUMÉRIQUES viennent EN PREMIER car c'est l'ordre du scaler
    # pendant l'entraînement (vérifié via scaler_x.feature_names_in_).
    X_combined = np.hstack([X_num, X_cat_enc])

    # ÉTAPE 5 — Normaliser avec StandardScaler
    # Ramène toutes les valeurs sur la même échelle que pendant l'entraînement.
    # Sans cette étape, le modèle donnerait de mauvaises prédictions.
    X_scaled = scaler_x.transform(X_combined)

    # ÉTAPE 6 — Prédire avec XGBoost
    # .flat[0] extrait le premier élément quelle que soit la forme du tableau
    # (évite les erreurs selon que y_pred est 1D ou 2D)
    y_pred_raw     = modele.predict(X_scaled)
    classe_predite = int(y_pred_raw.flat[0])

    # ÉTAPE 7 — Décoder l'indice prédit en nom de culture
    # encoder_y.classes_ est un tableau ex: ["Coton", "Légumineuses", "Maïs", ...]
    # On récupère directement le nom à l'indice prédit.
    # On utilise .classes_[indice] et non inverse_transform() qui cause des
    # erreurs avec certaines versions de LabelBinarizer.
    culture_modele = encoder_y.classes_[classe_predite]

    # ÉTAPE 8 — Appliquer la correspondance béninoise (transparente)
    # Si le modèle prédit Blé/Orge/Millet, on traduit silencieusement
    # vers Sorgho/Fonio/Mil. L'utilisateur ne voit jamais l'original.
    culture = CORRESPONDANCE_CULTURES.get(culture_modele, culture_modele)

    # ÉTAPE 9 — Calculer les probabilités pour le score de confiance
    # predict_proba retourne les probabilités pour chaque classe.
    # On prend la probabilité de la classe prédite comme score de confiance.
    try:
        probas    = modele.predict_proba(X_scaled)
        confiance = float(probas[0][classe_predite])
    except Exception:
        confiance = 0.0

    # ÉTAPE 10 — Calculer les valeurs SHAP avec TreeExplainer
    # On utilise la même logique que notre fonction get_shap_top3 du notebook.
    # TreeExplainer est optimisé pour XGBoost et les modèles à base d'arbres.
    explainer     = shap.TreeExplainer(modele)
    shap_vals_all = explainer.shap_values(X_scaled)

    # Extraction des valeurs SHAP pour la classe prédite.
    # On gère les deux formats possibles selon la version de SHAP :
    # - Liste de tableaux (une liste par classe)
    # - Tableau multidimensionnel 3D ou 2D
    if isinstance(shap_vals_all, list):
        # Format liste : on accède à la classe prédite directement
        shap_1d = shap_vals_all[classe_predite][0]
    else:
        shap_vals_arr = np.asarray(shap_vals_all)
        if shap_vals_arr.ndim == 3:
            # Format 3D [instance, feature, classe] → on prend la classe prédite
            shap_1d = shap_vals_arr[0, :, classe_predite]
        elif shap_vals_arr.ndim == 2:
            # Format 2D [instance, feature] → on prend la première instance
            shap_1d = shap_vals_arr[0]
        else:
            raise ValueError(f"Format SHAP inattendu : {shap_vals_arr.shape}")

    # ÉTAPE 11 — Construire l'objet shap.Explanation pour le waterfall
    # base_values = valeur de départ du modèle (sans aucune feature).
    # values      = contribution de chaque feature à la prédiction.
    # data        = valeurs réelles des features (après scaling).
    # feature_names = noms lisibles des features pour l'affichage.
    base_value = (
        explainer.expected_value[classe_predite]
        if hasattr(explainer.expected_value, '__len__')
        else explainer.expected_value
    )

    # On construit X_scaled sous forme de DataFrame avec les noms de colonnes
    # pour que SHAP affiche les bons noms sur le waterfall.
    X_scaled_df = pd.DataFrame(X_scaled, columns=NOMS_FEATURES)

    shap_explanation = shap.Explanation(
        values        = shap_1d,
        base_values   = base_value,
        data          = X_scaled_df.values[0],
        feature_names = NOMS_FEATURES
    )

    return {
        "culture":          culture,           # Nom béninois (Sorgho/Fonio/Mil si applicable)
        "emoji":            EMOJIS_CULTURES.get(culture, "🌱"),
        "description":      DESCRIPTIONS_CULTURES.get(culture, "Culture adaptée à votre parcelle"),
        "confiance":        confiance,
        "shap_1d":          shap_1d,           # Gardé pour le chat (message auto)
        "shap_explanation": shap_explanation,  # Pour le waterfall dans show_analysis()
        "mode":             "reel"
    }


def predire(params, type_sol, region, saison, culture_prec):
    """
    Fonction principale de prédiction — choisit automatiquement entre
    le modèle réel et la simulation selon la disponibilité des fichiers .pkl.

    C'est la seule fonction appelée depuis l'interface.
    """
    if MODELE_REEL_DISPONIBLE:
        # Les fichiers .pkl sont présents → on utilise XGBoost + SHAP réels
        return predire_avec_modele_reel(params, type_sol, region, saison, culture_prec)
    else:
        # Fichiers .pkl absents → on utilise la simulation de secours
        return simuler_prediction(params)


# =============================================================================
# SECTION 7 — SIDEBAR ET NAVIGATION
#
# La sidebar contient le logo AgriSmart et les 4 boutons de navigation.
# st.rerun() force Streamlit à relancer le script depuis le début
# pour afficher la nouvelle page immédiatement.
# =============================================================================

with st.sidebar:

    # Logo + nom AgriSmart en haut de la sidebar.
    # IMPORTANT : pas de commentaires HTML <!-- --> dans st.markdown().
    # Streamlit les afficherait comme du texte brut au lieu de les ignorer.
    st.markdown("""
    <div style="padding: 22px 16px 18px;
                border-bottom: 1px solid rgba(255,255,255,0.12);
                margin-bottom: 16px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:46px; height:46px;
                        background:rgba(255,255,255,0.18);
                        border-radius:10px;
                        display:flex; align-items:center;
                        justify-content:center;
                        font-size:24px; flex-shrink:0;">🌿</div>
            <div>
                <div style="font-size:21px; font-weight:800;
                            color:#d1fae5; line-height:1.2;">AgriSmart</div>
                <div style="font-size:11px;
                            color:rgba(255,255,255,0.5);
                            letter-spacing:0.05em;
                            margin-top:3px;">Bénin · IA Agricole</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Définition des pages dans l'ordre d'affichage
    nav_pages = [
        ("🏠", "Accueil"),
        ("🔬", "Analyse"),
        ("💬", "Chat"),
        ("📋", "Historique"),
    ]

    for icone, nom in nav_pages:
        # est_actif = True si ce bouton correspond à la page actuellement ouverte
        est_actif = (st.session_state.page_active == nom)

        # Crée le bouton avec le style "primary" (actif) ou "secondary" (inactif)
        if st.button(
            f"{icone}   {nom}",
            key=f"nav_{nom}",
            use_container_width=True,
            type="primary" if est_actif else "secondary"
        ):
            # Mémorise la nouvelle page choisie dans le Session State
            st.session_state.page_active = nom
            # Réinitialise le flag du message SHAP automatique du chat
            # pour qu'il s'affiche à nouveau si on revient sur le chat
            st.session_state.chat_shap_affiche = False
            # CORRECTION ANTI-CONFLIT DOM :
            # On coupe l'animation AVANT le rerun pour éviter que le navigateur
            # essaie de supprimer un élément encore en cours de modification.
            # Sans cette ligne, naviguer pendant l'animation provoque l'erreur
            # "removeChild" dans la console du navigateur.
            st.session_state.animer_dernier_message = False
            # Force Streamlit à recharger le script → affiche la nouvelle page
            st.rerun()

    # Indicateur de mode en bas de la sidebar
    st.markdown("<br>", unsafe_allow_html=True)
    if MODELE_REEL_DISPONIBLE:
        st.markdown("""
        <div style="padding:10px 14px; background:rgba(110,231,183,0.15);
                    border-radius:8px; text-align:center;">
            <div style="font-size:11px; color:#6ee7b7; font-weight:600;">
                ✅ Modèle XGBoost chargé
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:10px 14px; background:rgba(251,191,36,0.15);
                    border-radius:8px; text-align:center;">
            <div style="font-size:11px; color:#fbbf24; font-weight:600;">
                ⚙️ Mode simulation actif
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SECTION 8 — PAGE D'ACCUEIL
# =============================================================================

def show_home():
    """
    Affiche la page d'accueil avec :
    - Section héro avec photo agricole
    - Chiffres clés
    - Comment ça marche (3 étapes)
    - FAQ interactive
    - Section contact avec formulaire
    """

    # CSS spécifique à la page d'accueil uniquement
    st.markdown("""
    <style>
    .faq-titre { font-size: 18px; font-weight: 700; color: #111827; margin-bottom: 16px; }
    </style>
    """, unsafe_allow_html=True)

    # --- SECTION HÉRO ---
    # Grande bannière avec photo agricole et texte par-dessus.
    # L'URL Unsplash est gratuite et libre de droits.
    st.markdown("""
    <div class="hero">
        <img class="hero-img"
             src="https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1200&q=80"
             alt="Champ agricole au Bénin"/>
        <div class="hero-overlay">
            <div class="hero-title">🌿 AgriSmart</div>
            <div class="hero-sub">
                L'intelligence artificielle au service de l'agriculture béninoise.
                Obtenez en quelques secondes la culture la mieux adaptée
                à votre parcelle, selon votre région et votre saison.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bouton fonctionnel pour naviguer vers la page Analyse.
    # Le bouton HTML dans le héro est décoratif — celui-ci est le vrai.
    if st.button("📊 Commencer l'analyse →", type="primary"):
        st.session_state.page_active = "Analyse"
        st.rerun()

    st.write("")

    # --- CHIFFRES CLÉS ---
    # Ces 3 cartes crédibilisent la plateforme.
    # Adapte les valeurs selon tes vrais résultats de test.
    st.markdown('<div class="sh">📈 AgriSmart en chiffres</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">🌽</div>
            <div class="stat-value">10</div>
            <div class="stat-label">Cultures recommandables</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        # Remplace 94% par ta vraie précision (accuracy_score sur ton test set)
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-value">94%</div>
            <div class="stat-label">Précision du modèle XGBoost</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">📍</div>
            <div class="stat-value">5</div>
            <div class="stat-label">Régions couvertes au Bénin</div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("---")

    # --- COMMENT ÇA MARCHE ---
    # 3 étapes illustrées avec photos Unsplash gratuites.
    st.markdown('<div class="sh">⚙️ Comment ça marche ?</div>', unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3, gap="medium")

    with e1:
        st.markdown("""
        <div class="step-card">
            <img class="step-img"
                 src="https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=600&q=80"
                 alt="Saisie des paramètres"/>
            <div class="step-body">
                <div class="step-num">1</div>
                <div class="step-title">Renseignez votre parcelle</div>
                <div class="step-desc">
                    Entrez les données de votre sol (N, P, K, pH)
                    et les conditions climatiques de votre zone.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with e2:
        st.markdown("""
        <div class="step-card">
            <img class="step-img"
                 src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=600&q=80"
                 alt="Analyse IA"/>
            <div class="step-body">
                <div class="step-num">2</div>
                <div class="step-title">L'IA analyse les données</div>
                <div class="step-desc">
                    Notre modèle XGBoost analyse vos paramètres
                    et identifie la culture la plus adaptée.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with e3:
        st.markdown("""
        <div class="step-card">
            <img class="step-img"
                 src="https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=600&q=80"
                 alt="Résultat"/>
            <div class="step-body">
                <div class="step-num">3</div>
                <div class="step-title">Recevez la recommandation</div>
                <div class="step-desc">
                    Obtenez la culture recommandée avec une explication
                    visuelle SHAP waterfall des facteurs décisifs.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("---")

    # --- FAQ ---
    # st.expander crée un accordéon dépliable — parfait pour les FAQ.
    # L'utilisateur clique sur la question pour voir la réponse.
    st.markdown('<div class="sh">❓ Questions fréquentes</div>', unsafe_allow_html=True)

    with st.expander("📌 Quelles données dois-je avoir pour utiliser AgriSmart ?"):
        st.write("""
        Vous avez besoin de deux types de données :

        **Données sol** : taux d'azote (N), phosphore (P), potassium (K),
        pH du sol, type de sol, altitude, engrais utilisé.

        **Données climat** : température moyenne, humidité de l'air,
        pluviométrie annuelle, vitesse du vent, région, saison, culture précédente.

        Ces données sont disponibles auprès de votre centre agricole local
        ou via une analyse de sol en laboratoire.
        """)

    with st.expander("📌 Comment le modèle fait-il ses recommandations ?"):
        st.write("""
        AgriSmart utilise un modèle **XGBoost** (Extreme Gradient Boosting),
        entraîné sur des données agro-climatiques couvrant 10 cultures.

        XGBoost analyse la combinaison de vos 14 paramètres et identifie
        la culture dont le profil correspond le mieux à votre parcelle.
        """)

    with st.expander("📌 Que signifie le graphique waterfall SHAP dans les résultats ?"):
        st.write("""
        Le graphique **waterfall SHAP** (SHapley Additive exPlanations) montre
        comment chaque paramètre contribue à la prédiction finale :

        🔴 **Barres rouges** : ce paramètre pousse vers la culture recommandée.

        🔵 **Barres bleues** : ce paramètre joue contre cette culture.

        **E[f(x)]** = prédiction de base du modèle sans aucune information.
        **f(x)** = prédiction finale avec vos données réelles.
        Chaque barre montre comment un paramètre fait évoluer cette prédiction.
        """)

    with st.expander("📌 Les saisons proposées correspondent-elles aux saisons du Bénin ?"):
        st.write("""
        Oui. AgriSmart utilise les **vraies saisons agricoles du Bénin**.

        **Sud-Bénin (Climat Subéquatorial — 4 saisons)**
        • Grande saison des pluies (Avril à mi-Juillet)
        • Petite saison sèche (mi-Juillet à mi-Septembre)
        • Petite saison des pluies (mi-Septembre à Octobre)
        • Grande saison sèche (Novembre à Mars)

        **Nord-Bénin (Climat Soudanien — 2 saisons)**
        • Saison des pluies (Mai à Octobre)
        • Saison sèche (Novembre à Avril)
        """)

    with st.expander("📌 Les recommandations sont-elles valables pour tout le Bénin ?"):
        st.write("""
        Oui. Le modèle couvre **5 régions** : Nord, Sud, Est, Ouest et Centre.
        Il intègre les spécificités climatiques de chaque zone.

        Les recommandations sont un **outil d'aide à la décision**, pas un
        substitut à l'expertise d'un agronome local.
        """)

    with st.expander("📌 Mes données sont-elles sauvegardées ?"):
        st.write("""
        Non. AgriSmart ne sauvegarde aucune donnée sur un serveur externe.
        L'historique visible dans l'onglet **Historique** est uniquement stocké
        en mémoire locale pendant votre session. Il disparaît à la fermeture.
        """)

    st.write("")
    st.markdown("---")

    # --- SECTION CONTACT ---
    st.markdown("""
    <div class="contact-section">
        <div style="font-size:18px; font-weight:700; color:#1B4332; margin-bottom:8px;">
            📬 Nous contacter
        </div>
        <div style="font-size:13px; color:#4B5563; margin-bottom:20px;">
            Une question, un bug ou une suggestion d'amélioration ?
            Écrivez-nous, nous répondons sous 24h.
        </div>
        <div style="display:flex; flex-direction:column; gap:8px; margin-bottom:20px;">
            <div style="font-size:13px; color:#374151;">
                📍 Université d'Abomey-Calavi, Bénin
            </div>
            <div style="font-size:13px; color:#374151;">
                ✉️ agrismart@laboiacaebzongo.bj
            </div>
            <div style="font-size:13px; color:#374151;">
                📞 +229 99 43 03 26
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Formulaire de contact.
    # st.form groupe les champs : Streamlit n'exécute rien avant que
    # l'utilisateur clique sur le bouton "Envoyer".
    # Sans st.form, l'app se rechargerait à chaque frappe dans les champs.
    with st.form("formulaire_contact"):
        st.markdown("**✍️ Envoyer un message**")
        col_nom, col_email = st.columns(2)
        with col_nom:
            nom   = st.text_input("Votre nom")
        with col_email:
            email = st.text_input("Votre email")
        message = st.text_area("Votre message", height=100)
        envoye  = st.form_submit_button(
            "Envoyer ✉️", type="primary", use_container_width=True
        )

    if envoye:
        if nom and email and message:
            st.success(
                f"✅ Merci {nom} ! Votre message a bien été reçu. "
                f"Nous vous répondrons à **{email}**."
            )
        else:
            st.warning("⚠️ Merci de remplir tous les champs avant d'envoyer.")


# =============================================================================
# SECTION 9 — PAGE ANALYSE
# =============================================================================

def show_analysis():
    """
    Affiche la page d'analyse avec :
    - Formulaire en deux colonnes (sol + climat)
    - Bouton de prédiction
    - Carte de résultat
    - Graphique waterfall SHAP d'explicabilité
    - Bouton "Discuter de cette recommandation" → accès direct au chat
    """

    # Topbar avec titre et badge technologique
    st.markdown("""
    <div class="topbar">
        <span class="topbar-title">🔬 Analyse de parcelle</span>
        <span class="topbar-badge">XGBoost · SHAP</span>
    </div>
    """, unsafe_allow_html=True)

    # Titre de section imposant
    st.markdown(
        '<div class="section-title">🌍 Paramètres du sol et du climat</div>',
        unsafe_allow_html=True
    )

    # --- FORMULAIRE EN DEUX COLONNES ---
    # Colonne gauche : données du sol
    # Colonne droite : données climatiques
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="sous-titre">🌱 Composition du sol</div>',
                    unsafe_allow_html=True)

        # Sliders pour les variables numériques du sol.
        # format="%d kg/ha" affiche l'unité directement sur le curseur.
        N        = st.slider("Azote (N)",        20,   160,  90,  format="%d kg/ha")
        P        = st.slider("Phosphore (P)",     10,    90,  46,  format="%d kg/ha")
        K        = st.slider("Potassium (K)",     10,   120,  62,  format="%d kg/ha")
        ph       = st.slider("pH du sol",        4.5,   8.5,  6.5, step=0.1)
        engrais  = st.slider("Engrais utilisé",   50,   350, 120,  format="%d kg/ha")
        altitude = st.slider("Altitude",           0,  2500, 200,  format="%d m")

        # Listes déroulantes pour les variables catégorielles du sol
        type_sol = st.selectbox("Type de sol",
                    ["Limoneux", "Sableux", "Argileux", "Limono-argileux"])
        region   = st.selectbox("Région",
                    ["Nord", "Sud", "Est", "Ouest", "Centre"])

    with col2:
        st.markdown('<div class="sous-titre">🌤️ Conditions climatiques</div>',
                    unsafe_allow_html=True)

        # Sliders pour les variables climatiques
        temperature  = st.slider("Température",       10.0,  40.0, 26.0,
                                 step=0.5, format="%.1f °C")
        humidite     = st.slider("Humidité de l'air", 30.0,  90.0, 65.0,
                                 step=0.5, format="%.1f %%")
        pluvio       = st.slider("Pluviométrie",     200.0, 3000.0, 900.0,
                                 step=10.0, format="%.0f mm")
        vent         = st.slider("Vitesse du vent",    1.0,  20.0,  8.0,
                                 step=0.5, format="%.1f km/h")

        # Guide des saisons béninoises — aide l'utilisateur à choisir
        with st.expander("🗺️ Quelle saison choisir selon votre région ?"):
            col_nord, col_sud = st.columns(2)
            with col_nord:
                st.markdown("""
                **🌍 Nord-Bénin**
                *(Alibori, Atacora, Borgou, Donga)*

                | Période | Saison |
                |---|---|
                | Mai → Octobre | Saison des pluies |
                | Novembre → Avril | Saison sèche |
                """)
            with col_sud:
                st.markdown("""
                **🌊 Sud-Bénin**
                *(Atlantique, Littoral, Mono, Ouémé, Plateau, Zou, Collines)*

                | Période | Saison |
                |---|---|
                | Avr. → mi-Juil. | Grande saison des pluies |
                | mi-Juil. → mi-Sep. | Petite saison sèche |
                | mi-Sep. → Oct. | Petite saison des pluies |
                | Nov. → Mars | Grande saison sèche |
                """)

        # Selectbox avec les vraies saisons béninoises.
        # La traduction vers Kharif/Rabi/Zaid se fait en arrière-plan
        # via CORRESPONDANCE_SAISONS — invisible pour l'utilisateur.
        saison_benin = st.selectbox(
            "Saison actuelle",
            list(CORRESPONDANCE_SAISONS.keys()),
            help="Choisissez la saison correspondant à votre zone géographique"
        )
        # Traduction silencieuse vers le code attendu par le modèle
        saison = CORRESPONDANCE_SAISONS[saison_benin]

        culture_prec = st.selectbox(
            "Culture précédente",
            ["Coton", "Riz", "Maïs", "Légumineuses", "Sorgho", "Légumes"]
        )

    st.write("")

    # --- BOUTON DE PRÉDICTION ---
    # Tout le pipeline de traitement ML se déclenche quand on clique ici.
    if st.button("🌾 Recommandation de culture",
                 use_container_width=True, type="primary"):

        # Regroupement des paramètres numériques dans un dictionnaire.
        # Ce format facilite la transmission entre les fonctions.
        params = {
            "N": N, "P": P, "K": K, "ph": ph,
            "temperature": temperature, "humidite": humidite,
            "pluvio": pluvio, "vent": vent,
            "altitude": altitude, "engrais": engrais
        }

        # Appel de la fonction principale de prédiction.
        # Elle choisit automatiquement entre le modèle réel et la simulation.
        # La recontextualisation béninoise est déjà appliquée à l'intérieur.
        with st.spinner("⏳ Analyse en cours..."):
            resultat = predire(params, type_sol, region, saison, culture_prec)

        # Calcul du top 3 SHAP en texte pour transmission à RASA.
        # RASA ne recalcule rien — il lit ce texte depuis les slots.
        if resultat.get("shap_1d") is not None:
            shap_vals = resultat["shap_1d"]
            top3_idx  = np.argsort(np.abs(shap_vals))[::-1][:3]
            lignes    = []
            for i in top3_idx:
                val       = float(shap_vals[i])
                direction = "favorise" if val >= 0 else "joue contre"
                lignes.append(
                    f"  • {NOMS_FEATURES[i]} ({val:+.3f}) : {direction} cette culture"
                )
            resultat["top3_shap"] = "\n".join(lignes)
        else:
            resultat["top3_shap"] = ""

        # Sauvegarde du résultat complet dans la mémoire de l'application.
        # On y ajoute aussi les paramètres et la date pour l'historique.
        resultat["params"]       = params
        resultat["type_sol"]     = type_sol
        resultat["region"]       = region
        resultat["saison_benin"] = saison_benin  # On stocke le nom béninois pour l'affichage
        resultat["saison"]       = saison
        resultat["culture_prec"] = culture_prec
        resultat["date"]         = datetime.now().strftime("%d/%m/%Y %H:%M")

        st.session_state.derniere_analyse = resultat

        # Ajout d'une copie dans l'historique (copie = évite les références partagées)
        st.session_state.historique.append(resultat.copy())

        # Réinitialise le chat pour que le message SHAP automatique
        # se génère avec les nouvelles données
        st.session_state.chat_messages    = []
        st.session_state.chat_shap_affiche = False

    # --- AFFICHAGE DU RÉSULTAT ---
    # Ce bloc s'affiche uniquement si une prédiction a déjà été effectuée.
    if st.session_state.derniere_analyse:
        a = st.session_state.derniere_analyse

        # Carte principale avec emoji, nom de culture et badge
        st.markdown(f"""
        <div class="result-card">
            <span class="result-emoji">{a['emoji']}</span>
            <div class="result-name">{a['culture']}</div>
            <div class="result-badge">✅ RECOMMANDÉ</div>
            <div class="result-desc">{a['description']}</div>
            <div style="font-size:13px; color:#374151; margin-top:12px;">
                Confiance du modèle :
                <strong style="color:#1B4332;">
                    {int(a['confiance']*100)}%
                </strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- BOUTON "DISCUTER DE CETTE RECOMMANDATION" ---
        # Permet à l'utilisateur de naviguer directement vers le chat
        # sans revenir manuellement dans l'onglet Chat.
        # Le contexte de la prédiction est déjà mémorisé dans session_state.
        if st.button(
            "💬 Discuter de cette recommandation",
            type="secondary",
            use_container_width=True,
            key="btn_discuter_analyse"
        ):
            # Réinitialise le chat pour générer un nouveau message contextuel
            st.session_state.chat_messages     = []
            st.session_state.chat_shap_affiche = False
            # Active l'animation pour le premier message automatique du chat
            st.session_state.animer_dernier_message = True
            # Navigue vers la page Chat
            st.session_state.page_active = "Chat"
            st.rerun()

        # --- WATERFALL SHAP ---
        st.markdown(
            '<div class="sous-titre">📊 Influence de chaque caractéristique</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='font-size:12px; color:#6B7280; margin-bottom:14px;'>"
            "🔴 Barres rouges = poussent vers cette culture &nbsp;|&nbsp; "
            "🔵 Barres bleues = jouent contre cette culture</p>",
            unsafe_allow_html=True
        )

        # Vérification que shap_explanation est disponible (modèle réel uniquement).
        # En mode simulation, shap_explanation vaut None — on affiche un message.
        if "shap_explanation" in a and a["shap_explanation"] is not None:

            # Création de la figure matplotlib pour le waterfall.
            # show=False est OBLIGATOIRE dans Streamlit : sans ça, SHAP essaie
            # d'afficher la figure directement ce qui cause une erreur.
            # On laisse Streamlit gérer l'affichage via st.pyplot().
            fig, ax = plt.subplots(figsize=(8, 5))

            shap.plots.waterfall(
                a["shap_explanation"],
                max_display=14,  # Affiche toutes les 14 features
                show=False       # OBLIGATOIRE pour Streamlit
            )

            # bbox_inches='tight' évite que les labels soient coupés sur les bords
            st.pyplot(fig, bbox_inches='tight')

            # Fermeture de la figure pour libérer la mémoire.
            # Sans plt.close(), les figures s'accumulent en mémoire à chaque clic.
            plt.close(fig)

        else:
            # Mode simulation : pas de shap_explanation disponible
            st.info(
                "📊 Le waterfall SHAP est disponible uniquement avec le modèle réel. "
                "Chargez les fichiers .pkl pour voir l'explication complète."
            )


# =============================================================================
# SECTION 10 — PAGE CHAT
#
# INTÉGRATION RASA :
# La connexion se fait via deux appels HTTP :
#   1. POST /conversations/{sender_id}/tracker/events  → injecte les slots
#   2. POST /webhooks/rest/webhook                     → envoie le message
#
# Si RASA n'est pas lancé, l'app bascule sur la simulation locale.
# =============================================================================

def afficher_avec_frappe(contenu_brut, placeholder):
    """
    Affiche un message bot avec un effet de frappe mot par mot.

    APPROCHE :
    On découpe le texte en tokens (mots + sauts de ligne séparés).
    À chaque token, on reconvertit les \\n accumulés en <br> avant le rendu.
    Cela garantit que :
      - les balises HTML restent complètes à chaque étape (jamais <b partiel)
      - les retours à la ligne sont visibles progressivement
      - l'effet de frappe est fluide et lisible

    PROTECTION ANTI-CONFLIT DOM :
    Si l'utilisateur navigue vers une autre page pendant l'animation,
    Streamlit lève une StopException interne. Le try/except l'intercepte
    proprement, évitant l'erreur removeChild dans le navigateur.

    Paramètres :
        contenu_brut (str)  : texte brut avec \\n comme séparateurs de ligne
        placeholder         : st.empty() dans lequel afficher progressivement
    """
    # Découpage en tokens : mots normaux + marqueurs de saut de ligne (\n)
    # re.split(r'(\n)') garde les \n comme éléments séparés de la liste
    tokens = re.split(r'(\n)', contenu_brut)

    texte_progressif = ""

    try:
        for token in tokens:
            if token == "":
                continue

            texte_progressif += token

            # Conversion des \n accumulés en <br> pour le rendu HTML.
            # On fait cette conversion à CHAQUE itération pour que les sauts
            # de ligne soient visibles au fur et à mesure de l'animation.
            texte_html = texte_progressif.replace("\n", "<br>")

            placeholder.markdown(
                f'<div class="chat-bot">🤖 {texte_html}</div>',
                unsafe_allow_html=True
            )

            if token == "\n":
                # Pause légèrement plus longue aux sauts de ligne
                time.sleep(0.06)
            else:
                # Délai par mot : 0.018s → environ 55 mots/seconde
                time.sleep(0.018)

    except Exception:
        # Si Streamlit interrompt l'animation (navigation, rerun),
        # on affiche le texte complet instantanément et on sort proprement.
        # Cela évite l'erreur DOM removeChild dans le navigateur.
        texte_html = texte_progressif.replace("\n", "<br>")
        placeholder.markdown(
            f'<div class="chat-bot">🤖 {texte_html}</div>',
            unsafe_allow_html=True
        )


def generer_reponse_bot(message_utilisateur, derniere_analyse):
    """
    Envoie le message à RASA et retourne la réponse.

    FONCTIONNEMENT :
    1. Injection des slots via l'API tracker/events (méthode officielle RASA)
    2. Envoi du message via /webhooks/rest/webhook
    3. Si RASA n'est pas disponible, bascule sur la simulation locale

    CORRECTION CRITIQUE — Transmission des slots :
    L'ancienne méthode (metadata dans /set_slots) ne fonctionne pas dans RASA.
    La seule méthode correcte est l'API tracker/events qui définit les slots
    directement dans le tracker RASA avant d'envoyer le message.

    Paramètres :
        message_utilisateur (str)  : message tapé par l'utilisateur
        derniere_analyse (dict)    : résultat de la dernière prédiction

    Retourne :
        str : réponse du bot (RASA ou simulation)
    """
    try:
        import requests
    except ImportError:
        return generer_reponse_simulee(message_utilisateur, derniere_analyse)

    # Identifiant de session — doit être identique dans les deux appels
    SENDER_ID      = "agrismart_user"
    # Endpoint principal : envoi des messages utilisateur
    RASA_URL       = f"http://localhost:5005/webhooks/rest/webhook"
    # Endpoint tracker : définition directe des slots (méthode officielle RASA)
    RASA_SLOTS_URL = f"http://localhost:5005/conversations/{SENDER_ID}/tracker/events"

    try:
        # ÉTAPE 1 — Injecter les slots dans le tracker RASA via l'API officielle.
        # Cette méthode est la seule qui fonctionne réellement pour transmettre
        # des données de Streamlit vers RASA sans passer par le NLU.
        if derniere_analyse and "params" in derniere_analyse:
            p = derniere_analyse["params"]

            # Chaque événement de type "slot" définit la valeur d'un slot
            # directement dans la mémoire du tracker RASA.
            events = [
                {"event": "slot", "name": "culture_predite",
                 "value": str(derniere_analyse.get("culture", ""))},
                {"event": "slot", "name": "confiance",
                 "value": str(int(derniere_analyse.get("confiance", 0) * 100))},
                {"event": "slot", "name": "top3_shap",
                 "value": str(derniere_analyse.get("top3_shap", ""))},
                {"event": "slot", "name": "N",
                 "value": str(p.get("N", 90))},
                {"event": "slot", "name": "P",
                 "value": str(p.get("P", 46))},
                {"event": "slot", "name": "K",
                 "value": str(p.get("K", 62))},
                {"event": "slot", "name": "ph",
                 "value": str(p.get("ph", 6.5))},
                {"event": "slot", "name": "temperature",
                 "value": str(p.get("temperature", 26.0))},
                {"event": "slot", "name": "humidite",
                 "value": str(p.get("humidite", 65.0))},
                {"event": "slot", "name": "pluvio",
                 "value": str(p.get("pluvio", 900.0))},
                {"event": "slot", "name": "vent",
                 "value": str(p.get("vent", 8.0))},
                {"event": "slot", "name": "altitude",
                 "value": str(p.get("altitude", 200))},
                {"event": "slot", "name": "engrais",
                 "value": str(p.get("engrais", 120))},
                {"event": "slot", "name": "type_sol",
                 "value": str(derniere_analyse.get("type_sol", "Limoneux"))},
                {"event": "slot", "name": "region",
                 "value": str(derniere_analyse.get("region", "Sud"))},
                # saison contient le code modèle (Kharif/Rabi/Zaid)
                {"event": "slot", "name": "saison",
                 "value": str(derniere_analyse.get("saison", "Kharif"))},
                {"event": "slot", "name": "culture_prec",
                 "value": str(derniere_analyse.get("culture_prec", "Maïs"))},
            ]

            # Envoi des slots au tracker RASA (timeout=3s pour ne pas bloquer)
            requests.post(RASA_SLOTS_URL, json=events, timeout=3)

        # ÉTAPE 2 — Envoyer le message de l'utilisateur à RASA
        payload      = {"sender": SENDER_ID, "message": message_utilisateur}
        reponse_http = requests.post(RASA_URL, json=payload, timeout=5)

        # ÉTAPE 3 — Traiter la réponse RASA
        # RASA retourne une liste de messages — on les concatène avec \n\n
        messages = reponse_http.json()
        if messages:
            textes = [m.get("text", "") for m in messages if m.get("text")]
            return "\n\n".join(textes) if textes else "🤔 Pas de réponse de RASA."
        return "🤔 RASA n'a pas généré de réponse pour cette question."

    except requests.exceptions.ConnectionError:
        # RASA n'est pas démarré → bascule sur la simulation locale
        return (
            "⚠️ *[Mode hors-ligne — RASA non connecté]*\n\n"
            + generer_reponse_simulee(message_utilisateur, derniere_analyse)
        )

    except requests.exceptions.Timeout:
        return "⏳ RASA met trop de temps à répondre. Réessayez dans quelques secondes."

    except Exception as e:
        return (
            f"⚠️ *[Erreur : {str(e)}]*\n\n"
            + generer_reponse_simulee(message_utilisateur, derniere_analyse)
        )


def generer_reponse_simulee(message_utilisateur, derniere_analyse):
    """
    Simulation locale de secours — utilisée si RASA n'est pas disponible.
    """
    msg = message_utilisateur.lower()

    if any(mot in msg for mot in ["pourquoi", "raison", "explication", "comment", "facteur", "paramètre", "variable", "top 3", "top3"]):
        if derniere_analyse:
            culture  = derniere_analyse["culture"]
            top3     = derniere_analyse.get("top3_shap", "")
            conf     = int(derniere_analyse.get("confiance", 0) * 100)
            desc     = DESCRIPTIONS_CULTURES.get(culture, "Culture adaptée à votre parcelle")
            reponse  = (
                f"🌿 Le modèle recommande **{culture}** avec une confiance de **{conf}%**.\n\n"
                f"📋 {desc}\n\n"
            )
            if top3:
                reponse += f"📊 Les 3 facteurs les plus décisifs :\n{top3}\n\n"
            reponse += (
                f"Les barres 🔴 rouges du graphique waterfall (page Analyse) montrent\n"
                f"ce qui a le plus poussé vers cette recommandation."
            )
            return reponse

    if any(mot in msg for mot in ["alternative", "autre", "option", "différent"]):
        alternatives = random.sample(CULTURES, 3)
        return (
            f"🌱 D'autres cultures pourraient convenir à votre parcelle :\n\n"
            f"• **{alternatives[0]}** : {DESCRIPTIONS_CULTURES.get(alternatives[0], '')}\n"
            f"• **{alternatives[1]}** : {DESCRIPTIONS_CULTURES.get(alternatives[1], '')}\n"
            f"• **{alternatives[2]}** : {DESCRIPTIONS_CULTURES.get(alternatives[2], '')}\n\n"
            f"Modifiez les paramètres dans la page **Analyse** pour explorer ces alternatives."
        )

    if any(mot in msg for mot in ["conseil", "améliorer", "optimiser", "augmenter"]):
        return (
            "🌾 Conseils pour optimiser votre parcelle :\n\n"
            "🧪 Gestion du sol :\n"
            "• Maintenez un pH entre 5.5 et 7.0\n"
            "• Apportez de l'azote (N) si la valeur est inférieure à 40 kg/ha\n"
            "• Un taux de potassium (K) équilibré améliore la résistance\n\n"
            "🌧️ Gestion de l'eau :\n"
            "• Adaptez l'irrigation selon les besoins de la culture\n"
            "• Évitez les excès d'eau qui favorisent les maladies fongiques\n\n"
            "🔄 Rotation des cultures :\n"
            "• Les légumineuses enrichissent naturellement le sol en azote\n"
            "• Évitez de planter la même culture deux saisons consécutives"
        )

    if any(mot in msg for mot in ["bonjour", "salut", "bonsoir", "hello"]):
        return (
            "👋 Bonjour ! Je suis l'assistant AgriSmart.\n\n"
            "Je peux vous aider à :\n"
            "• 🌾 Comprendre votre recommandation de culture\n"
            "• 🔍 Expliquer les facteurs décisifs\n"
            "• 🌱 Proposer des alternatives\n"
            "• 📋 Donner les bonnes pratiques agricoles\n\n"
            "Que souhaitez-vous savoir ?"
        )

    if any(mot in msg for mot in ["shap", "waterfall", "graphique", "barre"]):
        return (
            "📊 Le graphique **waterfall SHAP** montre comment chaque paramètre\n"
            "contribue à la prédiction finale.\n\n"
            "🔴 Barres rouges → poussent VERS la culture recommandée\n"
            "🔵 Barres bleues → jouent CONTRE cette culture\n\n"
            "E[f(x)] = prédiction de base du modèle sans données\n"
            "f(x) = prédiction finale avec vos données réelles\n\n"
            "Plus une barre est longue, plus son influence est importante."
        )

    return (
        "🤔 Je n'ai pas bien compris votre question.\n\n"
        "Vous pouvez me demander :\n"
        "• *'Pourquoi cette culture ?'* → explication SHAP\n"
        "• *'Quelles alternatives ?'* → autres cultures possibles\n"
        "• *'Conseils agricoles'* → recommandations pratiques"
    )


def show_chat():
    """
    Affiche la page de chat avec :
    - Message automatique contextuel au chargement
    - Historique des messages avec formatage HTML correct (\\n → <br>)
    - Effet de frappe sur le dernier message bot
    - Champ de saisie + bouton d'envoi
    """

    # Topbar avec titre et badge RASA
    st.markdown("""
    <div class="topbar">
        <span class="topbar-title">💬 Assistant AgriSmart</span>
        <span class="topbar-badge">RASA Ready</span>
    </div>
    """, unsafe_allow_html=True)

    # --- MESSAGE AUTOMATIQUE CONTEXTUEL AU CHARGEMENT ---
    # Si une analyse a été effectuée et que le message n'a pas encore été affiché,
    # on génère un message riche qui nomme la culture, ses caractéristiques
    # et les facteurs décisifs. Ce message informe immédiatement l'utilisateur
    # sans qu'il ait à poser de questions.
    if (st.session_state.derniere_analyse and
            not st.session_state.chat_shap_affiche):

        a        = st.session_state.derniere_analyse
        culture  = a["culture"]
        shap_1d  = a["shap_1d"]
        noms     = NOMS_FEATURES
        conf     = int(a.get("confiance", 0) * 100)
        desc     = DESCRIPTIONS_CULTURES.get(culture, "Culture adaptée à votre parcelle")

        # Identification des 2 features positives les plus influentes
        indices_pos = [i for i in np.argsort(shap_1d)[::-1] if shap_1d[i] > 0][:2]

        # Construction du message automatique enrichi :
        # on nomme explicitement la culture, sa description, sa confiance
        # et les facteurs décisifs — bien plus informatif que l'ancienne version.
        if len(indices_pos) >= 2:
            f1 = noms[indices_pos[0]]
            f2 = noms[indices_pos[1]]
            message_auto = (
                f"🌿 Analyse terminée !\n\n"
                f"Le modèle recommande **{culture}** pour votre parcelle "
                f"avec une confiance de **{conf}%**.\n\n"
                f"📋 {desc}\n\n"
                f"📊 Les deux facteurs les plus déterminants sont :\n"
                f"• **{f1}** — favorise fortement cette culture\n"
                f"• **{f2}** — correspond bien aux exigences de {culture}\n\n"
                f"Posez-moi vos questions : pourquoi ce choix, quelles alternatives, "
                f"ou demandez le guide de production complet."
            )
        else:
            message_auto = (
                f"🌿 Analyse terminée !\n\n"
                f"Le modèle recommande **{culture}** pour votre parcelle "
                f"avec une confiance de **{conf}%**.\n\n"
                f"📋 {desc}\n\n"
                f"Posez-moi vos questions sur cette recommandation !"
            )

        # Ajout du message à l'historique
        st.session_state.chat_messages.append({
            "role":    "bot",
            "content": message_auto
        })
        # Marque ce message comme affiché pour ne pas le répéter
        st.session_state.chat_shap_affiche = True
        # Active l'animation pour ce premier message automatique
        st.session_state.animer_dernier_message = True

    # Si aucune analyse n'a été faite, message d'invitation
    elif not st.session_state.derniere_analyse:
        st.info(
            "💡 Effectuez d'abord une analyse dans la page **🔬 Analyse** "
            "pour activer l'assistant contextuel."
        )

    # --- AFFICHAGE DE L'HISTORIQUE DES MESSAGES ---
    # On parcourt tous les messages.
    # Pour le dernier message bot avec le flag d'animation actif,
    # on joue l'effet de frappe. Pour tous les autres, affichage instantané.
    for idx, msg in enumerate(st.session_state.chat_messages):

        if msg["role"] == "bot":

            est_dernier = (idx == len(st.session_state.chat_messages) - 1)

            if est_dernier and st.session_state.animer_dernier_message:
                # Dernier message bot + animation demandée →
                # crée un emplacement vide et anime le texte progressivement.
                # afficher_avec_frappe() gère la conversion \n → <br> en interne.
                placeholder = st.empty()
                afficher_avec_frappe(msg["content"], placeholder)
                # Désactive le flag après animation pour éviter de rejouer
                # sur les reruns suivants (scroll, autre interaction, etc.)
                st.session_state.animer_dernier_message = False

            else:
                # Messages anciens ou sans animation :
                # conversion \n → <br> pour que les sauts de ligne soient visibles.
                contenu_html = msg["content"].replace("\n", "<br>")
                st.markdown(
                    f'<div class="chat-bot">🤖 {contenu_html}</div>',
                    unsafe_allow_html=True
                )

        else:
            # Message utilisateur : conversion \n → <br> par cohérence
            contenu_html = msg["content"].replace("\n", "<br>")
            st.markdown(
                f'<div class="chat-user">{contenu_html} 👤</div>',
                unsafe_allow_html=True
            )

    # --- FORMULAIRE DE SAISIE DU CHAT ---
    # st.form évite que la page se recharge à chaque frappe dans le champ texte.
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Message",
                placeholder="Posez votre question sur la recommandation...",
                label_visibility="collapsed"
            )
        with col_btn:
            envoyer = st.form_submit_button("Envoyer", use_container_width=True)

    if envoyer and user_input.strip():
        # Ajout du message utilisateur à l'historique
        st.session_state.chat_messages.append({
            "role":    "user",
            "content": user_input.strip()
        })

        # Génération de la réponse via RASA ou simulation
        reponse = generer_reponse_bot(
            user_input.strip(),
            st.session_state.derniere_analyse
        )

        # Ajout de la réponse à l'historique
        st.session_state.chat_messages.append({
            "role":    "bot",
            "content": reponse
        })

        # Active l'animation pour ce nouveau message bot
        st.session_state.animer_dernier_message = True

        # Force le rechargement pour afficher les nouveaux messages
        st.rerun()


# =============================================================================
# SECTION 11 — PAGE HISTORIQUE
# =============================================================================

def show_history():
    """
    Affiche la page d'historique avec :
    - Liste de toutes les analyses de la session
    - Détails expandables pour chaque analyse
    - Bouton "Recharger l'analyse" → affiche le résultat dans la page Analyse
    - Bouton "Discuter avec le chatbot" → ouvre le chat avec le contexte de cette analyse
    """

    # Topbar
    st.markdown("""
    <div class="topbar">
        <span class="topbar-title">📋 Historique des analyses</span>
    </div>
    """, unsafe_allow_html=True)

    # Cas où aucune analyse n'a encore été effectuée
    if not st.session_state.historique:
        st.info(
            "📋 Aucune analyse pour l'instant. "
            "Allez dans **🔬 Analyse** pour commencer."
        )
        return

    # Compteur d'analyses affiché en haut
    nb = len(st.session_state.historique)
    st.markdown(
        f"<p style='font-size:13px; color:#6B7280; margin-bottom:16px;'>"
        f"{nb} analyse(s) effectuée(s) pendant cette session.</p>",
        unsafe_allow_html=True
    )

    # Affichage du plus récent au plus ancien (reversed)
    for idx, analyse in enumerate(reversed(st.session_state.historique)):

        # Index réel dans la liste (pour identifier l'analyse correctement)
        idx_reel = len(st.session_state.historique) - 1 - idx

        # Titre de l'accordéon : emoji + culture + date
        titre = f"{analyse['emoji']} {analyse['culture']} — {analyse['date']}"

        with st.expander(titre):

            p = analyse["params"]

            # Affichage des paramètres principaux en deux colonnes
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"🌡️ Température : **{p['temperature']} °C**")
                st.write(f"💧 Humidité : **{p['humidite']} %**")
                st.write(f"🌧️ Pluviométrie : **{p['pluvio']} mm**")
                st.write(f"🌍 Région : **{analyse.get('region', 'N/A')}**")
            with col_b:
                st.write(f"🧪 Azote (N) : **{p['N']} kg/ha**")
                st.write(f"🧪 Phosphore (P) : **{p['P']} kg/ha**")
                st.write(f"🧪 Potassium (K) : **{p['K']} kg/ha**")
                # Affichage de la saison béninoise (pas le code interne)
                st.write(f"🌿 Saison : **{analyse.get('saison_benin', analyse.get('saison', 'N/A'))}**")

            # Score de confiance
            confiance = analyse.get("confiance", 0)
            st.write(f"🎯 Confiance du modèle : **{int(confiance*100)}%**")

            # Deux boutons côte à côte : recharger l'analyse + discuter avec le chatbot
            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                # Bouton pour recharger cette analyse dans la page Analyse
                # pour revoir le waterfall SHAP complet.
                if st.button(
                    "🔄 Recharger l'analyse",
                    key=f"reload_{idx_reel}",
                    type="primary",
                    use_container_width=True
                ):
                    # On remet cette analyse comme analyse active
                    st.session_state.derniere_analyse = analyse
                    # Réinitialise le chat pour qu'il explique cette analyse
                    st.session_state.chat_messages    = []
                    st.session_state.chat_shap_affiche = False
                    # Navigue vers la page Analyse pour voir le résultat
                    st.session_state.page_active = "Analyse"
                    st.rerun()

            with btn_col2:
                # Bouton pour ouvrir le chat directement avec le contexte
                # de cette analyse spécifique, sans navigation manuelle.
                if st.button(
                    "💬 Discuter avec le chatbot",
                    key=f"chat_{idx_reel}",
                    type="secondary",
                    use_container_width=True
                ):
                    # On charge cette analyse comme analyse active
                    st.session_state.derniere_analyse = analyse
                    # Réinitialise le chat pour générer un nouveau message contextuel
                    st.session_state.chat_messages    = []
                    st.session_state.chat_shap_affiche = False
                    # Active l'animation pour le premier message automatique
                    st.session_state.animer_dernier_message = True
                    # Navigue vers le chat
                    st.session_state.page_active = "Chat"
                    st.rerun()


# =============================================================================
# SECTION 12 — ROUTEUR PRINCIPAL
#
# C'est ici que Streamlit décide quelle page afficher selon la valeur
# de st.session_state.page_active. C'est l'équivalent d'un router
# dans une application web classique (React Router, Vue Router...).
#
# Chaque elif correspond à une page. L'ajout d'une nouvelle page
# se fait en ajoutant un elif + une fonction show_nouvelle_page().
# =============================================================================

if st.session_state.page_active == "Accueil":
    show_home()

elif st.session_state.page_active == "Analyse":
    show_analysis()

elif st.session_state.page_active == "Chat":
    show_chat()

elif st.session_state.page_active == "Historique":
    show_history()

# Sécurité : si page_active contient une valeur inattendue,
# on affiche la page d'accueil par défaut
else:
    st.session_state.page_active = "Accueil"
    show_home()
