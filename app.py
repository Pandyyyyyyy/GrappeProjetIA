"""
Application Streamlit principale pour le système de recommandation œnologique
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import List, Dict, Optional
from data_loader import WineDataLoader
from semantic_search import SemanticWineSearch
from genai_integration import WineGenAI
from data_analysis import WineDataAnalysis
from visualizations import WineVisualizations
from food_pairing_matcher import FoodPairingMatcher
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="Grappe - Recommandation Œnologique",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé - Design professionnel et coloré
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables de couleurs */
    :root {
        --wine-red: #8B0000;
        --wine-burgundy: #722F37;
        --wine-gold: #D4AF37;
        --wine-cream: #F5F1E8;
        --wine-dark: #2C1810;
        --wine-light: #FFF5E6;
        --gradient-primary: linear-gradient(135deg, #8B0000 0%, #722F37 50%, #A0522D 100%);
        --gradient-gold: linear-gradient(135deg, #D4AF37 0%, #F4D03F 100%);
        --gradient-light: linear-gradient(135deg, #FFF5E6 0%, #F5F1E8 100%);
    }
    
    /* Masquer la barre blanche Streamlit en haut */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    
    /* Style global */
    .stApp {
        background: var(--gradient-light);
        font-family: 'Inter', sans-serif;
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Ajuster le padding pour compenser la barre supprimée */
    .main .block-container {
        padding-top: 2rem !important;
    }
    
    /* Header principal */
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 700;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(139, 0, 0, 0.1);
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #722F37;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Sidebar stylisée */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4A2C2A 0%, #8B4A4A 100%);
        color: white;
    }
    
    /* Tous les textes de la sidebar en blanc/crème */
    [data-testid="stSidebar"] * {
        color: #F5F1E8 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown label,
    [data-testid="stSidebar"] .stMarkdown span {
        color: #F5F1E8 !important;
        font-family: 'Playfair Display', serif;
    }
    
    /* Labels des inputs */
    [data-testid="stSidebar"] label {
        color: #F5F1E8 !important;
    }
    
    /* Textes des selectbox et inputs */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #F5F1E8 !important;
    }
    
    /* Inputs et selectbox avec fond clair */
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stNumberInput > div > div > input,
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #2C1810 !important;
    }
    
    /* Checkbox avec texte visible */
    [data-testid="stSidebar"] .stCheckbox {
        color: #F5F1E8 !important;
    }
    
    [data-testid="stSidebar"] .stCheckbox label {
        color: #F5F1E8 !important;
    }
    
    /* Slider labels */
    [data-testid="stSidebar"] .stSlider label {
        color: #F5F1E8 !important;
    }
    
    /* Help text */
    [data-testid="stSidebar"] [data-testid="stTooltipIcon"] {
        color: #F5F1E8 !important;
    }
    
    /* Expander header */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #F5F1E8 !important;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader p {
        color: #F5F1E8 !important;
    }
    
    /* Réduire les espacements dans la sidebar pour tout afficher sans scroll */
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0.3rem !important;
    }
    
    [data-testid="stSidebar"] h3 {
        margin-bottom: 0.5rem !important;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stSidebar"] .stSlider {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
    }
    
    [data-testid="stSidebar"] .stCheckbox {
        margin-bottom: 0.2rem !important;
    }
    
    [data-testid="stSidebar"] .stCaption {
        margin-top: 0.2rem !important;
        font-size: 0.85rem !important;
    }
    
    [data-testid="stSidebar"] hr {
        margin: 0.5rem 0 !important;
    }
    
    /* Cartes de vin */
    .wine-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(139, 0, 0, 0.15);
        margin-bottom: 2rem;
        border: 2px solid transparent;
        background-image: linear-gradient(white, white), var(--gradient-primary);
        background-origin: border-box;
        background-clip: padding-box, border-box;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .wine-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(139, 0, 0, 0.25);
    }
    
    /* Badge de score */
    .score-badge {
        background: var(--gradient-gold);
        color: #2C1810;
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4);
        display: inline-block;
        text-align: center;
        min-width: 120px;
    }
    
    /* Badge de type de vin */
    .wine-type-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .type-rouge {
        background: linear-gradient(135deg, #8B0000, #A0522D);
        color: white;
    }
    
    .type-blanc {
        background: linear-gradient(135deg, #F5DEB3, #FFE4B5);
        color: #2C1810;
    }
    
    .type-rose {
        background: linear-gradient(135deg, #FFB6C1, #FFC0CB);
        color: #722F37;
    }
    
    .type-bulles {
        background: linear-gradient(135deg, #FFF8DC, #F0E68C);
        color: #2C1810;
    }
    
    .type-liquoreux {
        background: var(--gradient-gold);
        color: #2C1810;
    }
    
    /* Boutons stylisés */
    .stButton>button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 0, 0, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 0, 0, 0.4);
    }
    
    /* Inputs stylisés */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #E8DCC6;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #8B0000;
        box-shadow: 0 0 0 3px rgba(139, 0, 0, 0.1);
    }
    
    /* Sliders stylisés */
    .stSlider>div>div>div {
        background: var(--gradient-primary);
    }
    
    /* Tabs stylisés */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        color: #722F37;
        border: 2px solid #E8DCC6;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary);
        color: white;
        border-color: #8B0000;
    }
    
    /* Expanders stylisés */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #F5F1E8, white);
        border-radius: 10px;
        font-weight: 600;
        color: #722F37;
    }
    
    /* Messages de succès/erreur */
    .stSuccess {
        background: linear-gradient(90deg, #D4EDDA, #C3E6CB);
        border-left: 4px solid #28A745;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stWarning {
        background: linear-gradient(90deg, #FFF3CD, #FFE69C);
        border-left: 4px solid #FFC107;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stInfo {
        background: linear-gradient(90deg, #D1ECF1, #BEE5EB);
        border-left: 4px solid #17A2B8;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Prix stylisé */
    .wine-price {
        font-size: 1.5rem;
        font-weight: 700;
        color: #8B0000;
        background: var(--gradient-gold);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    /* Mots-clés stylisés */
    .wine-keywords {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .keyword-tag {
        background: #E8DCC6;
        color: #722F37;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Section d'accords */
    .food-pairing {
        background: linear-gradient(135deg, #FFF5E6 0%, #F5F1E8 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #D4AF37;
        margin: 1rem 0;
    }
    
    /* Animation de chargement */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .loading-shimmer {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 1000px 100%;
        animation: shimmer 2s infinite;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialisation de la session
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = None
if 'semantic_search' not in st.session_state:
    st.session_state.semantic_search = None
    # Initialiser GenAI (optionnel)
    # Utilise Google Gemini 2.5 Flash par défaut (gratuit)
    # Alternative: WineGenAI(provider="openai") pour OpenAI
    if 'genai' not in st.session_state:
        # Essayer Gemini d'abord (gratuit), sinon OpenAI si disponible
        gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if gemini_key:
            st.session_state.genai = WineGenAI(provider="gemini", api_key=gemini_key)
        else:
            # Fallback sur OpenAI si Gemini non configuré
            st.session_state.genai = WineGenAI(provider="openai")
if 'embeddings_computed' not in st.session_state:
    st.session_state.embeddings_computed = False
if 'food_pairing_matcher' not in st.session_state:
    st.session_state.food_pairing_matcher = FoodPairingMatcher()

def initialize_system():
    """Initialise le système de recommandation"""
    csv_path = "Projet IA BDD Vins - BDD Vins.csv"
    
    # Charger les données
    if st.session_state.data_loader is None:
        with st.spinner("Chargement des données de vins..."):
            st.session_state.data_loader = WineDataLoader(csv_path)
            # ÉTAPE 1 : Charger la BDD
            st.session_state.data_loader.load_data()
            # ÉTAPE 2 : Vérifier la BDD (validation)
            st.session_state.data_loader.validate_data()
            # ÉTAPE 3 : Fusionner les 3 dernières colonnes et préprocesser
            wines = st.session_state.data_loader.preprocess_data()
            st.session_state.wines = wines
    
    # Initialiser la recherche sémantique
    if st.session_state.semantic_search is None:
        st.session_state.semantic_search = SemanticWineSearch()
        st.session_state.semantic_search.load_model()
    else:
        # Vérifier que la méthode search_similar_in_wines existe (pour forcer le rechargement si nécessaire)
        if not hasattr(st.session_state.semantic_search, 'search_similar_in_wines'):
            # Recharger le module si la méthode n'existe pas
            import importlib
            import semantic_search
            importlib.reload(semantic_search)
            # Réinitialiser avec le modèle rechargé
            old_model = st.session_state.semantic_search.model
            old_embeddings = st.session_state.semantic_search.wine_embeddings
            old_wines = st.session_state.semantic_search.wines
            st.session_state.semantic_search = semantic_search.SemanticWineSearch()
            st.session_state.semantic_search.model = old_model
            st.session_state.semantic_search.wine_embeddings = old_embeddings
            st.session_state.semantic_search.wines = old_wines
        # S'assurer que les wines sont à jour
        if st.session_state.semantic_search.wines != st.session_state.wines:
            st.session_state.semantic_search.wines = st.session_state.wines
    
    # Calculer les embeddings si nécessaire
    # Note: Si vous modifiez la structure des données, supprimez wine_embeddings.pkl pour recalculer
    if not st.session_state.embeddings_computed:
        with st.spinner("Calcul des embeddings sémantiques (cela peut prendre quelques instants)..."):
            # Vérifier si les embeddings existent et sont à jour
            embeddings_file = "wine_embeddings.pkl"
            if os.path.exists(embeddings_file):
                # Vérifier la date de modification du CSV vs embeddings
                csv_path = "Projet IA BDD Vins - BDD Vins.csv"
                if os.path.exists(csv_path):
                    csv_mtime = os.path.getmtime(csv_path)
                    emb_mtime = os.path.getmtime(embeddings_file)
                    # Si le CSV est plus récent, recalculer
                    if csv_mtime > emb_mtime:
                        st.info("⚠️ La base de données a été modifiée. Recalcul des embeddings...")
                        st.session_state.semantic_search.compute_embeddings(
                            st.session_state.wines,
                            force_recompute=True
                        )
                    else:
                        st.session_state.semantic_search.compute_embeddings(
                            st.session_state.wines,
                            force_recompute=False
                        )
                else:
                    st.session_state.semantic_search.compute_embeddings(
                        st.session_state.wines,
                        force_recompute=False
                    )
            else:
                st.session_state.semantic_search.compute_embeddings(
                    st.session_state.wines,
                    force_recompute=False
                )
            st.session_state.embeddings_computed = True

def main():
    """Fonction principale de l'application"""
    
    # En-tête stylisé
    st.markdown('<h1 class="main-header">🍷 Grappe</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Votre Sommelier Intelligent - Recommandations personnalisées par IA</p>', unsafe_allow_html=True)
    
    # Initialiser le système
    initialize_system()
    
    # Sidebar pour les filtres et préférences - Version compacte
    with st.sidebar:
        st.markdown("### 🎯 Préférences")
        
        # Filtres de base - Version compacte
        st.markdown("**🍷 Type de Vin**")
        
        # Types de base uniquement - En 2 colonnes pour économiser l'espace
        basic_wine_types = ["Rouge", "Blanc", "Rosé", "Bulles"]
        selected_types = []
        
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("Rouge", key="type_Rouge", value=False):
                selected_types.append("Rouge")
            if st.checkbox("Blanc", key="type_Blanc", value=False):
                selected_types.append("Blanc")
        with col2:
            if st.checkbox("Rosé", key="type_Rosé", value=False):
                selected_types.append("Rosé")
            if st.checkbox("Bulles", key="type_Bulles", value=False):
                selected_types.append("Bulles")
        
        # Si aucun type sélectionné, considérer "Tous"
        if not selected_types:
            selected_type = "Tous"
        else:
            selected_type = selected_types[0] if len(selected_types) == 1 else "Multiple"
        
        # Budget - Version compacte
        st.markdown("**💰 Budget**")
        budget_max = st.slider(
            "Budget max (€)",
            min_value=0,
            max_value=200,
            value=100,
            step=5,
            help="Budget maximum"
        )
        # Afficher le budget de manière compacte
        if budget_max > 0:
            st.caption(f"💰 {budget_max}€")
        else:
            st.caption("💰 Aucun filtre")
        
        # Région retirée - valeur par défaut
        selected_region = "Toutes"
        
        # Échelle d'intensité aromatique - Version compacte
        st.markdown("**🌸 Intensité**")
        intensite_aromatique = st.select_slider(
            "Intensité aromatique",
            options=["Léger", "Moyen", "Intense", "Fort"],
            value="Moyen",
            help="De léger à fort"
        )
        
        # Convertir en valeur numérique pour le traitement (1-4)
        intensite_map = {"Léger": 1, "Moyen": 2, "Intense": 3, "Fort": 4}
        intensite_aromatique_num = intensite_map[intensite_aromatique]
        
        # Valeurs par défaut pour les autres paramètres (non utilisés dans le nouveau questionnaire)
        astringence = 3
        acidite = 3
        
        # Options techniques (cachées par défaut ou simplifiées)
        cepage = ""  # Retiré du questionnaire principal
        enrich_query = True  # Activé par défaut
        use_genai_justification = True  # Activé par défaut
    
    # Zone principale
    st.markdown("### 🔍 Recherche de vin")
    st.markdown("---")
    
    # Onglets
    tab1, tab2 = st.tabs(["📝 Recherche libre", "📊 Analyses & Statistiques"])
    
    with tab1:
        st.markdown("#### 💬 Questionnaire de Recherche")
        st.markdown("")
        
        # Question ouverte 1 : Occasion principale
        user_query = st.text_area(
            "1️⃣ Décrivez votre occasion, votre repas ou l'ambiance recherchée",
            placeholder="Ex: Je cherche un vin pour un dîner romantique en hiver avec un plat de canard...",
            height=100,
            help="Décrivez librement : l'occasion, le plat, l'ambiance, la saison..."
        )
        
        # Question ouverte 2 : Préférences particulières (optionnelle)
        preferences_query = st.text_area(
            "2️⃣ Vos préférences particulières (optionnel)",
            placeholder="Ex: J'aime les vins fruités, je préfère les vins bio, je cherche quelque chose d'original...",
            height=80,
            help="Décrivez vos préférences gustatives, vos envies particulières..."
        )
        
        # Fusionner les deux questions ouvertes
        if preferences_query.strip():
            full_user_query = f"{user_query.strip()}. {preferences_query.strip()}"
        else:
            full_user_query = user_query.strip()
        
        col_slider, col_btn = st.columns([2, 1])
        with col_slider:
            top_n = st.slider("Nombre de recommandations", min_value=1, max_value=10, value=3)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🍷 Trouver mon vin", type="primary", use_container_width=True):
            if not user_query.strip():
                st.warning("Veuillez répondre au moins à la première question.")
            else:
                # Gérer la sélection multiple de types
                type_filter = selected_type if selected_type != "Multiple" else None
                if selected_types and len(selected_types) > 1:
                    # Si plusieurs types sélectionnés, ne pas filtrer strictement
                    type_filter = None
                
                search_wines(full_user_query, top_n, type_filter, selected_region, 
                           budget_max, astringence, acidite, intensite_aromatique_num,
                           cepage, enrich_query, use_genai_justification, selected_types)
    
    with tab2:
        display_analytics_tab()

def display_analytics_tab():
    """Affiche l'onglet d'analyses et statistiques"""
    st.markdown("#### 📊 Analyses & Statistiques de la Base de Données")
    st.markdown("")
    
    if 'wines' not in st.session_state or not st.session_state.wines:
        st.warning("Les données ne sont pas encore chargées. Veuillez patienter...")
        return
    
    # Initialiser les analyseurs
    analyzer = WineDataAnalysis(st.session_state.wines)
    visualizer = WineVisualizations(st.session_state.wines)
    
    # Section KPIs
    st.markdown("### 📈 Indicateurs Clés (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_wines = len(st.session_state.wines)
        st.metric("🍷 Total Vins", total_wines)
    
    with col2:
        prix_valid = [w['prix'] for w in st.session_state.wines if w['prix'] > 0]
        prix_moyen = sum(prix_valid) / len(prix_valid) if prix_valid else 0
        st.metric("💰 Prix Moyen", f"{prix_moyen:.2f}€")
    
    with col3:
        unique_types = len(set(w['type'] for w in st.session_state.wines))
        st.metric("🏷️ Types de Vins", unique_types)
    
    with col4:
        unique_regions = len(set(w['region'] for w in st.session_state.wines))
        st.metric("📍 Régions", unique_regions)
    
    st.markdown("---")
    
    # Statistiques descriptives
    st.markdown("### 📊 Statistiques Descriptives")
    
    stats = analyzer.get_descriptive_statistics()
    if stats:
        st.markdown("#### Prix - Statistiques")
        stats_df = pd.DataFrame({
            'Métrique': ['Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum', 'Q1', 'Q3', 'IQR'],
            'Valeur (€)': [
                f"{stats.get('mean', 0):.2f}",
                f"{stats.get('median', 0):.2f}",
                f"{stats.get('std', 0):.2f}",
                f"{stats.get('min', 0):.2f}",
                f"{stats.get('max', 0):.2f}",
                f"{stats.get('q25', 0):.2f}",
                f"{stats.get('q75', 0):.2f}",
                f"{stats.get('iqr', 0):.2f}"
            ]
        })
        st.dataframe(stats_df, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # Visualisations - 4 graphiques pertinents uniquement
    st.markdown("### 📈 Visualisations")
    
    # Graphique 1 : Distribution par Type (camembert)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribution par Type")
        fig_type_pie = visualizer.plot_type_distribution()
        st.plotly_chart(fig_type_pie, width='stretch')
    
    # Graphique 2 : Prix Moyen par Type
    with col2:
        st.markdown("#### Prix Moyen par Type")
        fig_type_bar = visualizer.plot_price_by_type()
        st.plotly_chart(fig_type_bar, width='stretch')
    
    # Graphique 3 : Distribution des Prix par Type (Box Plot)
    st.markdown("#### Distribution des Prix par Type (Box Plot)")
    fig_box = visualizer.plot_price_boxplot_by_type()
    st.plotly_chart(fig_box, width='stretch')
    
    # Graphique 4 : Top Régions
    st.markdown("#### Top Régions")
    fig_region = visualizer.plot_region_distribution(top_n=15)
    st.plotly_chart(fig_region, width='stretch')

def extract_wine_characteristics(wines: List[Dict]) -> Dict[str, float]:
    """
    Extrait les caractéristiques sensorielles moyennes des vins
    depuis leurs descriptions et mots-clés
    
    Args:
        wines: Liste des dictionnaires de vins
        
    Returns:
        Dictionnaire avec les scores moyens (0-1) pour chaque caractéristique
    """
    if not wines:
        return {}
    
    characteristics = {
        'tanins': [],
        'acidite': [],
        'corps': [],
        'intensite_aromatique': [],
        'longueur': [],
        'complexite': []
    }
    
    for wine in wines:
        # Combiner mots-clés et description
        text = (wine.get('mots_cles', '') + ' ' + wine.get('description_narrative', '')).lower()
        
        # TANINS (0-1)
        tanins_score = 0.5  # Par défaut
        if any(word in text for word in ['tannique', 'tanins', 'structure', 'masse', 'charpenté', 'puissant']):
            tanins_score = 0.8
        elif any(word in text for word in ['tanins présents', 'structuré', 'corsé']):
            tanins_score = 0.7
        elif any(word in text for word in ['léger', 'souple', 'tendre', 'soyeux', 'tanins fondus']):
            tanins_score = 0.3
        elif any(word in text for word in ['peu de tanins', 'délicat', 'fraîcheur']):
            tanins_score = 0.2
        characteristics['tanins'].append(tanins_score)
        
        # ACIDITÉ (0-1)
        acidite_score = 0.5  # Par défaut
        if any(word in text for word in ['vif', 'nerveux', 'acidité', 'fraîcheur', 'minéral']):
            acidite_score = 0.8
        elif any(word in text for word in ['équilibré', 'harmonieux', 'bien équilibré']):
            acidite_score = 0.5
        elif any(word in text for word in ['rond', 'gras', 'doux', 'moelleux', 'sans acidité']):
            acidite_score = 0.3
        characteristics['acidite'].append(acidite_score)
        
        # CORPS (0-1)
        corps_score = 0.5  # Par défaut
        if any(word in text for word in ['puissant', 'corsé', 'charpenté', 'généreux', 'ample']):
            corps_score = 0.8
        elif any(word in text for word in ['moyen', 'équilibré', 'harmonieux']):
            corps_score = 0.5
        elif any(word in text for word in ['léger', 'délicat', 'finesse', 'élégant']):
            corps_score = 0.3
        characteristics['corps'].append(corps_score)
        
        # INTENSITÉ AROMATIQUE (0-1)
        intensite_score = 0.5  # Par défaut
        if any(word in text for word in ['intense', 'expressif', 'aromatique', 'explosif', 'puissant']):
            intensite_score = 0.8
        elif any(word in text for word in ['fruité', 'fruité', 'arômes', 'nez']):
            intensite_score = 0.6
        elif any(word in text for word in ['subtil', 'délicat', 'discret', 'léger']):
            intensite_score = 0.3
        characteristics['intensite_aromatique'].append(intensite_score)
        
        # LONGUEUR (0-1)
        longueur_score = 0.5  # Par défaut
        if any(word in text for word in ['longue', 'persistance', 'finale longue', 'rétro-olfaction']):
            longueur_score = 0.8
        elif any(word in text for word in ['finale', 'persistance moyenne']):
            longueur_score = 0.5
        elif any(word in text for word in ['courte', 'finale courte']):
            longueur_score = 0.3
        characteristics['longueur'].append(longueur_score)
        
        # COMPLEXITÉ (0-1)
        complexite_score = 0.5  # Par défaut
        if any(word in text for word in ['complexe', 'nuancé', 'multiples arômes', 'riche']):
            complexite_score = 0.8
        elif any(word in text for word in ['fruité', 'fruité', 'arômes']):
            complexite_score = 0.6
        elif any(word in text for word in ['simple', 'direct', 'facile']):
            complexite_score = 0.3
        characteristics['complexite'].append(complexite_score)
    
    # Calculer les moyennes
    avg_characteristics = {
        'Tanins': sum(characteristics['tanins']) / len(characteristics['tanins']) if characteristics['tanins'] else 0.5,
        'Acidité': sum(characteristics['acidite']) / len(characteristics['acidite']) if characteristics['acidite'] else 0.5,
        'Corps': sum(characteristics['corps']) / len(characteristics['corps']) if characteristics['corps'] else 0.5,
        'Intensité Aromatique': sum(characteristics['intensite_aromatique']) / len(characteristics['intensite_aromatique']) if characteristics['intensite_aromatique'] else 0.5,
        'Longueur': sum(characteristics['longueur']) / len(characteristics['longueur']) if characteristics['longueur'] else 0.5,
        'Complexité': sum(characteristics['complexite']) / len(characteristics['complexite']) if characteristics['complexite'] else 0.5
    }
    
    return avg_characteristics

def create_radar_chart(characteristics: Dict[str, float]) -> go.Figure:
    """
    Crée un graphique en radar (araignée) des caractéristiques des vins
    
    Args:
        characteristics: Dictionnaire avec les scores moyens (0-1) pour chaque caractéristique
        
    Returns:
        Figure Plotly
    """
    categories = list(characteristics.keys())
    values = [characteristics[cat] * 100 for cat in categories]  # Convertir en pourcentage
    
    # Ajouter le premier point à la fin pour fermer le polygone
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='Caractéristiques moyennes',
        line=dict(color='#8B0000', width=3),
        fillcolor='rgba(139, 0, 0, 0.3)',
        marker=dict(size=8, color='#8B0000')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=12, color='#2C1810'),
                gridcolor='rgba(139, 0, 0, 0.2)',
                linecolor='#8B0000'
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color='#2C1810', family='Playfair Display'),
                linecolor='#8B0000',
                rotation=90
            )
        ),
        title={
            'text': '🕷️ Profil Sensoriel Moyen des Vins Recommandés',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Playfair Display', 'color': '#2C1810'}
        },
        height=500,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Playfair Display', color='#2C1810')
    )
    
    return fig

def search_wines(
    user_query: str,
    top_n: int,
    selected_type: str,
    selected_region: str,
    budget_max: float,
    astringence: int,
    acidite: int,
    intensite_aromatique: int,
    cepage: str,
    enrich_query: bool,
    use_genai_justification: bool,
    selected_types: list = None
):
    """Effectue la recherche et affiche les résultats"""
    
    with st.spinner("Recherche en cours..."):
        # Enrichir la requête si demandé (seulement si requête < 5 mots)
        genai_available = (hasattr(st.session_state.genai, 'client') and st.session_state.genai.client) or \
                         (hasattr(st.session_state.genai, 'genai_client') and st.session_state.genai.genai_client)
        if enrich_query and genai_available:
            with st.spinner("Enrichissement de la requête avec l'IA..."):
                enriched_query = st.session_state.genai.enrich_user_query(user_query)
                if enriched_query != user_query:
                    st.info(f"Requête enrichie: *{enriched_query}*")
                    query_to_use = enriched_query
                else:
                    query_to_use = user_query
        else:
            query_to_use = user_query
        
        # Préparer les filtres
        filters = {}
        # Gérer la sélection multiple de types
        # IMPORTANT : Ne définir filters['type'] que si exactement UN type est sélectionné
        # Si aucune case n'est cochée OU si plusieurs types sont sélectionnés, ne PAS définir filters['type']
        # Cela permet à la recherche sémantique de proposer tous les types pertinents
        if selected_types and len(selected_types) == 1:
            # Un seul type sélectionné via checkbox : on peut l'utiliser pour le scoring mais pas pour filtrer strictement
            # Le filtrage strict se fait plus bas dans le code
            filters['type'] = selected_types[0]
        # Si aucune case cochée ou plusieurs types sélectionnés, filters['type'] reste None
        
        if selected_region != "Toutes":
            filters['region'] = selected_region
        if budget_max > 0:
            filters['budget_max'] = budget_max
        if cepage:
            filters['cepage'] = cepage
        
        # Profil gustatif
        taste_profile = {
            'astringence': astringence,
            'acidite': acidite,
            'intensite_aromatique': intensite_aromatique
        }
        
        # Extraire le plat de la requête pour les accords mets-vins
        dish_info = st.session_state.food_pairing_matcher.extract_dish_from_query(query_to_use)
        
        # Construire la requête de recherche en incluant les préférences gustatives
        query_parts = [query_to_use]
        
        # Ajouter l'intensité aromatique à la requête pour que SBERT puisse matcher
        if intensite_aromatique == 4:  # Fort
            query_parts.append("vin fort puissant intense aromatique explosif")
        elif intensite_aromatique == 3:  # Intense
            query_parts.append("vin intense aromatique expressif")
        elif intensite_aromatique == 2:  # Moyen
            query_parts.append("vin modéré équilibré")
        elif intensite_aromatique == 1:  # Léger
            query_parts.append("vin léger subtil délicat discret")
        
        query_for_search = " ".join(query_parts)
        
        # Enrichir UNIQUEMENT si un plat est détecté (ajouter des termes d'accords)
        if dish_info.get('meat_category') or dish_info.get('dish'):
            query_for_search = st.session_state.food_pairing_matcher.enhance_query_with_pairing(query_for_search)
        
        # DÉTECTION AUTOMATIQUE DU TYPE DE VIN si aucun type n'est sélectionné
        # IMPORTANT : Si aucune case n'est cochée, NE PAS FILTRER par type
        if not filters.get('type') and not selected_types:
            # Ne pas filtrer par type - laisser la recherche sémantique proposer tous les types pertinents
            pass
        
        # FILTRAGE PAR TYPE UNIQUEMENT SI UNE CASE EST COCHÉE
        # Si aucune case n'est cochée, ne PAS filtrer - proposer tous les types pertinents
        wines_to_search = st.session_state.wines
        # Filtrer uniquement si l'utilisateur a explicitement coché au moins une case
        if selected_types and len(selected_types) > 0:
            # Si plusieurs types sélectionnés, ne pas filtrer strictement (laisser la recherche sémantique)
            if len(selected_types) == 1:
                # Un seul type sélectionné : filtrer strictement
                filter_type = selected_types[0].lower()
                wines_to_search = [
                    wine for wine in wines_to_search 
                    if wine.get('type', '').lower() == filter_type
                ]
                if not wines_to_search:
                    st.warning(f"Aucun vin de type '{selected_types[0]}' trouvé dans la base de données.")
                    return
            # Si plusieurs types sélectionnés, ne pas filtrer - laisser la recherche sémantique proposer parmi ces types
        
        # Recherche sémantique basée sur le profil textuel complet
        # Utilisation de la similarité sémantique des embeddings pour mesurer la couverture
        # Rechercher uniquement dans les vins filtrés
        try:
            # Utiliser la nouvelle méthode avec filtrage préalable
            # Augmenter le top_k pour avoir plus de vins à évaluer avec le calcul par blocs
            # Le calcul par blocs (avec poids élevé sur Accords_Mets) permettra de mieux filtrer
            # Recherche sémantique : utiliser la requête simplifiée
            # Réduire top_k pour avoir des résultats plus pertinents
            semantic_results = st.session_state.semantic_search.search_similar_in_wines(
                query_for_search,
                wines_to_search,
                top_k=min(50, len(wines_to_search))  # Top 50 pour avoir des résultats plus pertinents
            )
        except AttributeError:
            # Fallback si la méthode n'existe pas encore (cache Streamlit)
            # Filtrer manuellement après la recherche
            semantic_results_all = st.session_state.semantic_search.search_similar(
                query_for_search,
                top_k=min(50, len(st.session_state.wines))
            )
            # Filtrer par type après la recherche
            if filters.get('type'):
                filter_type = filters['type'].lower()
                semantic_results = [
                    (wine, score) for wine, score in semantic_results_all
                    if wine.get('type', '').lower() == filter_type
                ]
            else:
                semantic_results = semantic_results_all
        
        # EF2.2 et EF2.3 : Calcul simple basé uniquement sur la similarité cosinus
        # Pas de calculs complexes, juste la similarité cosinus entre vecteurs SBERT
        scored_wines = []
        
        for wine, semantic_score in semantic_results:
            # Filtrer les résultats avec une similarité trop faible (seuil minimum)
            # Les scores de similarité cosinus sont généralement entre 0.2 et 0.9
            # On garde seulement ceux avec une similarité >= 0.3 pour avoir des résultats pertinents
            if semantic_score < 0.3:
                continue  # Ignorer les vins avec une similarité trop faible
            
            # Appliquer les filtres stricts (budget, type si sélectionné)
            # Mais garder le score sémantique comme score principal
            final_score = semantic_score
            
            # Pénalité pour budget dépassé
            if filters.get('budget_max') and wine['prix'] > filters.get('budget_max', float('inf')):
                final_score *= 0.5  # Réduire de moitié si budget dépassé
            
            # DÉTECTION DES NÉGATIONS dans les descriptions
            # Si le vin dit explicitement "ce n'est pas un vin d'apéro", le pénaliser fortement
            wine_full_text = (wine.get('description_narrative', '') + " " + wine.get('mots_cles', '') + " " + wine.get('accords_mets', '')).lower()
            
            # Détecter les négations courantes (patterns regex)
            negation_patterns = [
                r"ce n'?est pas",
                r"n'?est pas",
                r"n'?est aucun",
                r"n'?est point",
                r"ne pas",
                r"pas un",
                r"pas de",
                r"jamais",
                r"aucun"
            ]
            
            # Contextes à vérifier selon la requête utilisateur
            query_lower = user_query.lower()
            contexts_to_check = []
            
            # Détecter le contexte recherché dans la requête
            if any(word in query_lower for word in ['apéro', 'apero', 'apéritif', 'aperitif']):
                contexts_to_check = ['apéro', 'apero', 'apéritif', 'aperitif']
            elif any(word in query_lower for word in ['viande rouge', 'bœuf', 'boeuf', 'entrecôte', 'steak']):
                contexts_to_check = ['viande rouge', 'bœuf', 'boeuf', 'entrecôte', 'steak']
            elif any(word in query_lower for word in ['viande blanche', 'poulet', 'volaille']):
                contexts_to_check = ['viande blanche', 'poulet', 'volaille']
            elif any(word in query_lower for word in ['poisson', 'saumon', 'bar']):
                contexts_to_check = ['poisson', 'saumon', 'bar']
            
            # Vérifier si un contexte recherché est mentionné dans une négation
            for context in contexts_to_check:
                # Chercher le contexte dans le texte
                if context in wine_full_text:
                    # Vérifier si c'est dans une phrase négative
                    # Chercher dans une fenêtre de 60 caractères avant et après le mot
                    context_pos = wine_full_text.find(context)
                    if context_pos != -1:
                        # Extraire une fenêtre autour du contexte (plus large pour capturer les négations)
                        start = max(0, context_pos - 60)
                        end = min(len(wine_full_text), context_pos + len(context) + 60)
                        context_window = wine_full_text[start:end]
                        
                        # Vérifier si une négation est présente dans cette fenêtre
                        for pattern in negation_patterns:
                            if re.search(pattern, context_window, re.IGNORECASE):
                                # Le vin dit explicitement que ce n'est PAS pour ce contexte
                                # Pénalité très forte : réduire à 5% du score original
                                final_score *= 0.05
                                break
            
            # PÉNALITÉ FORTE pour non-correspondance des accords mets-vins
            # Si l'utilisateur cherche de la viande rouge, pénaliser fortement les vins qui mentionnent du poulet
            if dish_info.get('meat_category') == 'viande_rouge':
                wine_accords = wine.get('accords_mets', '').lower()
                # Mots-clés incompatibles (viande blanche)
                incompatible_keywords = ['poulet', 'poularde', 'chapon', 'dinde', 'volaille', 'volailles', 'viande blanche']
                compatible_keywords = ['bœuf', 'boeuf', 'entrecôte', 'steak', 'agneau', 'gigot', 'mouton', 'veau', 'gibier', 'viande rouge', 'sanglier', 'canard', 'magret']
                
                has_incompatible = any(kw in wine_accords for kw in incompatible_keywords)
                has_compatible = any(kw in wine_accords for kw in compatible_keywords)
                
                if has_incompatible:
                    if has_compatible:
                        # Le vin mentionne les deux : pénalité modérée (mais quand même pénalité)
                        final_score *= 0.5  # Réduire de moitié
                    else:
                        # Le vin mentionne SEULEMENT du poulet/volaille : pénalité très forte
                        final_score *= 0.15  # Réduire à 15% du score original
                elif has_compatible:
                    # Le vin mentionne SEULEMENT de la viande rouge : bonus léger
                    final_score *= 1.1  # Augmenter de 10%
                    final_score = min(1.0, final_score)  # Ne pas dépasser 1.0
            
            # Pénalité pour non-correspondance de l'intensité aromatique
            # Vérifier si le vin correspond à l'intensité demandée
            wine_text = (wine.get('mots_cles', '') + " " + wine.get('description_narrative', '')).lower()
            
            if intensite_aromatique == 4:  # Fort demandé
                # Pénaliser si le vin est décrit comme léger ou subtil
                if any(word in wine_text for word in ['léger', 'subtil', 'délicat', 'discret', 'fin']):
                    final_score *= 0.4  # Forte pénalité
                # Bonus si le vin est décrit comme fort ou intense
                elif any(word in wine_text for word in ['fort', 'puissant', 'intense', 'explosif', 'aromatique', 'expressif']):
                    final_score *= 1.1  # Petit bonus (max 1.0 après normalisation)
            elif intensite_aromatique == 3:  # Intense demandé
                # Pénaliser si le vin est décrit comme léger ou discret
                if any(word in wine_text for word in ['léger', 'subtil', 'délicat', 'discret']):
                    final_score *= 0.5  # Pénalité modérée
            elif intensite_aromatique == 1:  # Léger demandé
                # Pénaliser si le vin est décrit comme fort ou puissant
                if any(word in wine_text for word in ['fort', 'puissant', 'intense', 'explosif']):
                    final_score *= 0.4  # Forte pénalité
                # Bonus si le vin est décrit comme léger ou subtil
                elif any(word in wine_text for word in ['léger', 'subtil', 'délicat', 'discret', 'fin']):
                    final_score *= 1.1  # Petit bonus
            
            # Normaliser le score final entre 0 et 1
            final_score = min(1.0, final_score)
            
            # Le score final est la similarité cosinus (0-1) avec ajustements
            # C'est conforme à EF2.2 (SBERT) et EF2.3 (Similarité Cosinus)
            scored_wines.append((wine, final_score, semantic_score))
        
        # Trier par score de similarité cosinus décroissant
        scored_wines.sort(key=lambda x: x[1], reverse=True)
        
        # Prendre le top N (triés par similarité cosinus décroissante)
        top_wines = scored_wines[:top_n]
        
        # Préparer les données pour le graphique de similarité cosinus
        # Prendre les top 15 pour le graphique
        top_for_chart = min(15, len(semantic_results))
        
        # Afficher les scores bruts (non normalisés) pour montrer la vraie similarité cosinus
        # IMPORTANT : Les scores de similarité cosinus avec SBERT sont généralement entre 0.3-0.7
        # pour des textes similaires mais pas identiques. C'est NORMAL et attendu.
        # Un score de 0.4-0.6 indique une bonne correspondance sémantique.
        # La normalisation min-max masquerait ces informations importantes.
        chart_data = []
        for wine, semantic_score in semantic_results[:top_for_chart]:
            chart_data.append({
                'nom': wine['nom'],
                'type': wine['type'],
                'score_cosinus': semantic_score,  # Score brut (0-1), pas de normalisation
                'score_original': semantic_score
            })
        
        # Afficher les résultats
        if top_wines:
            # Messages informatifs retirés pour simplifier l'interface
            # Affichage direct des recommandations
            
            # EF4.3 : Synthèse de Profil (un seul appel API pour la sortie finale)
            # Vérifier qu'un client GenAI est disponible (client ou genai_client)
            genai_available = (hasattr(st.session_state.genai, 'client') and st.session_state.genai.client) or \
                             (hasattr(st.session_state.genai, 'genai_client') and st.session_state.genai.genai_client)
            if use_genai_justification and genai_available:
                # Calculer la moyenne des scores de similarité cosinus
                avg_semantic = sum(score for _, score, _ in top_wines) / len(top_wines) if top_wines else 0
                with st.spinner("Génération de votre profil œnologique..."):
                    profile_summary = st.session_state.genai.generate_profile_summary(
                        query_for_search,
                        [w for w, _, _ in top_wines],
                        avg_semantic
                    )
                    st.markdown("---")
                    st.markdown("### 📋 Votre Profil Œnologique")
                    st.markdown(f'<div style="background: linear-gradient(135deg, #F5F1E8 0%, #FFF5E6 100%); padding: 1.5rem; border-radius: 15px; border-left: 5px solid #D4AF37; color: #2C1810; line-height: 1.8; font-style: italic;">{profile_summary}</div>', unsafe_allow_html=True)
                    st.markdown("---")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            for idx, wine_data in enumerate(top_wines, 1):
                # Structure simplifiée : (wine, final_score, semantic_score)
                wine, final_score, semantic_score = wine_data
                # Déterminer la classe CSS selon le type de vin
                wine_type_lower = wine['type'].lower()
                type_class = "type-rouge"
                if "blanc" in wine_type_lower:
                    type_class = "type-blanc"
                elif "rosé" in wine_type_lower or "rose" in wine_type_lower:
                    type_class = "type-rose"
                elif "bulle" in wine_type_lower or "pétillant" in wine_type_lower or "champagne" in wine_type_lower:
                    type_class = "type-bulles"
                elif "liquoreux" in wine_type_lower or "moelleux" in wine_type_lower:
                    type_class = "type-liquoreux"
                
                with st.container():
                    # Carte principale
                    st.markdown(f"""
                    <div class="wine-card">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                            <div style="flex: 1;">
                                <h2 style="color: #2C1810; margin-bottom: 0.5rem; font-family: 'Playfair Display', serif;">
                                    🍷 {idx}. {wine['nom']}
                                </h2>
                                <div style="margin: 1rem 0;">
                                    <span class="wine-type-badge {type_class}">{wine['type']}</span>
                                    <span style="color: #722F37; font-weight: 600;">📍 {wine['region']}</span>
                                </div>
                                <div style="margin: 0.5rem 0;">
                                    <span style="color: #722F37;">🍇 {wine['cepages']}</span>
                                </div>
                                <div class="wine-price">{wine['prix_str']}</div>
                            </div>
                            <div style="text-align: center;">
                                <div class="score-badge">⭐ {int(semantic_score * 100)}%</div>
                            </div>
                        </div>
                        <p style="color: #555; font-style: italic; line-height: 1.6; margin: 1rem 0;">
                            {wine['description_narrative']}
                        </p>
                        <div class="wine-keywords">
                            {''.join([f'<span class="keyword-tag">{kw.strip()}</span>' for kw in wine['mots_cles'].split(',') if kw.strip()])}
                        </div>
                        <div class="food-pairing">
                            <strong style="color: #722F37;">🍽️ Accords mets:</strong> {wine['accords_mets']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Justification GenAI (avec cache automatique)
                    genai_available = (hasattr(st.session_state.genai, 'client') and st.session_state.genai.client) or \
                                     (hasattr(st.session_state.genai, 'genai_client') and st.session_state.genai.genai_client)
                    if use_genai_justification and genai_available:
                        with st.expander(f"💡 Pourquoi ce vin? (Généré par IA)", expanded=False):
                            justification = st.session_state.genai.generate_recommendation_justification(
                                wine,
                                user_query,
                                semantic_score
                            )
                            st.markdown(f'<div style="background: #F5F1E8; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #D4AF37; color: #2C1810; line-height: 1.8;">{justification}</div>', unsafe_allow_html=True)
                    
                    # Analyse accord mets-vins
                    if wine['accords_mets']:
                        with st.expander("🍽️ Analyse de l'accord mets-vins", expanded=False):
                            # Utiliser le plat extrait de la requête
                            dish_for_analysis = dish_info.get('dish') if dish_info.get('dish') else None
                            
                            # Analyse de l'accord mets-vins générée par IA
                            analysis = st.session_state.genai.generate_food_pairing_analysis(
                                wine,
                                dish_for_analysis
                            )
                            st.markdown(f'<div style="background: #FFF5E6; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #8B0000; color: #2C1810; line-height: 1.8;">{analysis}</div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
            
            
            # Section Data Viz - Visualisations des résultats de recherche
            st.markdown("---")
            st.markdown("### 📊 Data Visualisation - Analyse des Résultats")
            
            if top_wines:
                # Préparer les données pour les graphiques
                wines_data = []
                for wine_data in top_wines:
                    # Structure simplifiée : (wine, final_score, semantic_score)
                    wine, final_score, semantic_score = wine_data
                    
                    wines_data.append({
                        'nom': wine['nom'][:30] + '...' if len(wine['nom']) > 30 else wine['nom'],
                        'type': wine['type'],
                        'prix': wine['prix'],
                        'semantic_score': semantic_score,  # Score de similarité cosinus
                        'region': wine['region']
                    })
                
                wines_df = pd.DataFrame(wines_data)
                
                # Graphique 1 : Similarité cosinus des vins recommandés
                st.markdown("#### 📈 Similarité Cosinus des Vins Recommandés")
                fig_coverage = px.bar(
                    wines_df,
                    x='nom',
                    y='semantic_score',
                    color='type',
                    title='Similarité Cosinus par Vin',
                    labels={'semantic_score': 'Similarité Cosinus', 'nom': 'Vin'},
                    color_discrete_map={
                        'Rouge': '#8B0000',
                        'Blanc': '#F5DEB3',
                        'Rosé': '#FFB6C1',
                        'Bulles': '#FFF8DC'
                    }
                )
                fig_coverage.update_layout(
                    xaxis_tickangle=-45,
                    height=400,
                    showlegend=True,
                    template='plotly_white'
                )
                fig_coverage.update_traces(texttemplate='%{y:.2%}', textposition='outside')
                st.plotly_chart(fig_coverage, width='stretch')
                
                # Graphique 2 : Prix des vins recommandés
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 💰 Prix des Vins Recommandés")
                    fig_price = px.bar(
                        wines_df,
                        x='nom',
                        y='prix',
                        color='type',
                        title='Prix par Vin',
                        labels={'prix': 'Prix (€)', 'nom': 'Vin'},
                        color_discrete_map={
                            'Rouge': '#8B0000',
                            'Blanc': '#F5DEB3',
                            'Rosé': '#FFB6C1',
                            'Bulles': '#FFF8DC'
                        }
                    )
                    fig_price.update_layout(
                        xaxis_tickangle=-45,
                        height=350,
                        showlegend=True,
                        template='plotly_white'
                    )
                    fig_price.update_traces(texttemplate='%{y:.0f}€', textposition='outside')
                    st.plotly_chart(fig_price, width='stretch')
                
                with col2:
                    st.markdown("#### 🍷 Répartition par Type")
                    type_counts = wines_df['type'].value_counts()
                    fig_type = px.pie(
                        values=type_counts.values,
                        names=type_counts.index,
                        title='Distribution des Types Recommandés',
                        color_discrete_map={
                            'Rouge': '#8B0000',
                            'Blanc': '#F5DEB3',
                            'Rosé': '#FFB6C1',
                            'Bulles': '#FFF8DC'
                        }
                    )
                    fig_type.update_layout(height=350, template='plotly_white')
                    fig_type.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_type, width='stretch')
                
                # Graphique 3 : Relation Prix vs Score de Couverture
                st.markdown("#### 📊 Relation Prix vs Score de Couverture")
                fig_scatter = px.scatter(
                    wines_df,
                    x='prix',
                    y='semantic_score',
                    size='semantic_score',
                    color='type',
                    hover_name='nom',
                    title='Prix vs Similarité Cosinus',
                    labels={'prix': 'Prix (€)', 'semantic_score': 'Similarité Cosinus', 'type': 'Type'},
                    color_discrete_map={
                        'Rouge': '#8B0000',
                        'Blanc': '#F5DEB3',
                        'Rosé': '#FFB6C1',
                        'Bulles': '#FFF8DC'
                    }
                )
                fig_scatter.update_layout(
                    height=400,
                    showlegend=True,
                    template='plotly_white'
                )
                fig_scatter.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig_scatter, width='stretch')
            
            # Carte de France avec les vins recommandés
            st.markdown("---")
            st.markdown("### 🗺️ Carte de France - Localisation des Vins Recommandés")
            st.markdown("Carte interactive montrant la répartition géographique des vins recommandés avec leurs informations (région, cépage, prix)")
            
            if top_wines:
                # Préparer les données pour la carte (inclure le score)
                wines_for_map = []
                for wine_data in top_wines:
                    wine, final_score, semantic_score = wine_data
                    wine_with_score = wine.copy()
                    wine_with_score['score'] = f"{semantic_score*100:.1f}%"
                    wines_for_map.append(wine_with_score)
                
                # Créer la carte
                visualizer = WineVisualizations(wines_for_map)
                fig_map = visualizer.create_france_wine_map(wines_for_map)
                st.plotly_chart(fig_map, width='stretch')
            
            # Graphique en radar (araignée) des caractéristiques moyennes
            st.markdown("---")
            st.markdown("### 🕷️ Profil Sensoriel des Vins Recommandés")
            st.markdown("Graphique en radar montrant les caractéristiques moyennes des vins recommandés")
            
            if top_wines:
                # Extraire les caractéristiques des vins recommandés
                characteristics = extract_wine_characteristics([w for w, _, _ in top_wines])
                
                if characteristics:
                    fig_radar = create_radar_chart(characteristics)
                    st.plotly_chart(fig_radar, width='stretch')
            
        else:
            st.warning("Aucun vin ne correspond à vos critères. Essayez d'élargir vos filtres.")

if __name__ == "__main__":
    main()
