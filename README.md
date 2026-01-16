# 🍷 Grappe - Système de Recommandation Œnologique Intelligent

Système intelligent de recommandation de vins basé sur l'analyse sémantique (SBERT) et l'IA générative. Le système analyse les préférences gustatives et contextuelles de l'utilisateur pour proposer des vins personnalisés avec justifications générées par IA.

---

## 🎯 Vue d'ensemble

**Grappe** est un système de recommandation de vins qui combine :
- **Recherche sémantique** (SBERT) : Comprend le sens des requêtes utilisateur
- **Filtrage intelligent** : Exclusion automatique des vins inappropriés
- **IA générative** : Enrichissement de requêtes et justifications personnalisées

### Problématique résolue
Choisir un vin adapté à un repas ou une occasion est complexe. Ce système automatise cette sélection en analysant sémantiquement les besoins de l'utilisateur et en proposant des recommandations justifiées.

---

## 🚀 Fonctionnalités Principales

### 🔍 Recherche Sémantique
- **Embeddings SBERT** : Modèle multilingue (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Similarité cosinus** : Calcul de proximité entre requête et descriptions de vins
- **Fusion intelligente** : Combinaison pondérée de description, mots-clés et accords mets-vins

### 🎯 Filtrage Intelligent
- **Détection automatique** : Type de plat (viande rouge/blanche, poisson, fromage)
- **Exclusion contextuelle** : Viande rouge → exclut rosés et blancs
- **Détection de négations** : "Ce n'est pas un vin d'apéro" → exclusion automatique
- **Filtres utilisateur** : Type, budget, région, intensité aromatique

### 🤖 Intégration GenAI
- **Enrichissement de requêtes** : Expansion automatique des requêtes courtes (< 5 mots)
- **Justifications personnalisées** : Notes de dégustation expliquant chaque recommandation
- **Analyse pédagogique** : Explications des accords mets-vins
- **Cache intelligent** : Réduction des coûts API

### 📊 Visualisations
- Graphiques de similarité cosinus
- Répartition des prix
- Carte de France interactive
- Profil sensoriel en radar

---

## 📋 Structure du Projet

```
Grappe/
├── app.py                      # Application Streamlit principale
├── data_loader.py              # Chargement et traitement du CSV
├── semantic_search.py          # Embeddings SBERT et recherche sémantique
├── genai_integration.py        # Intégration OpenAI/Gemini pour GenAI
├── food_pairing_matcher.py     # Détection et matching des plats
├── data_analysis.py             # Analyses statistiques
├── visualizations.py           # Graphiques Plotly
├── genai_cache.py              # Système de cache pour GenAI
├── requirements.txt            # Dépendances Python
├── Projet IA BDD Vins - BDD Vins.csv  # Base de données (200 vins)
├── wine_embeddings.pkl         # Embeddings pré-calculés (généré)
├── genai_cache.db              # Cache GenAI (généré)
├── README.md                   # Ce fichier
├── EXPLICATION_TECHNIQUE.md    # Documentation technique détaillée
├── EXPLICATION_FLUX.md         # Explication du flux de recherche
└── STRUCTURE_BDD_OPTIMALE.md   # Guide de structure de BDD
```

---

## 🛠️ Installation

### Prérequis
- Python 3.8 ou supérieur
- pip
- Connexion internet (pour télécharger le modèle SBERT la première fois)

### Étapes d'installation

1. **Cloner ou télécharger le projet**

2. **Créer un environnement virtuel** (recommandé)
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configuration de l'API GenAI** (optionnel)
   - Créer un fichier `.env` à la racine du projet
   - Ajouter votre clé API :
   ```
   OPENAI_API_KEY=your_api_key_here
   # OU
   GEMINI_API_KEY=your_api_key_here
   ```
   - Obtenez votre clé sur :
     - OpenAI : [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
     - Google Gemini : [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
   
   **Note** : L'application fonctionne sans clé API, mais les fonctionnalités GenAI (enrichissement et justifications) seront désactivées.

---

## 💻 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

### Première utilisation

Lors du premier lancement, le système va :
1. Charger les données depuis le CSV (200 vins)
2. Télécharger le modèle SBERT (première fois uniquement, ~100 MB)
3. Calculer les embeddings pour tous les vins (2-3 minutes)
4. Sauvegarder les embeddings dans `wine_embeddings.pkl` pour réutilisation

**Note** : Les embeddings sont calculés une seule fois. Les recherches suivantes seront instantanées (< 1 seconde).

### Utilisation de l'interface

#### Onglet "Recherche libre"
1. **Décrivez votre recherche** dans la zone de texte :
   - Occasion : "dîner romantique", "apéro entre amis"
   - Plat : "côte de bœuf", "saumon", "fromage frais"
   - Ambiance : "hiver", "été", "célébration"
   
2. **Ajustez les filtres** dans la barre latérale :
   - Type de vin (Rouge, Blanc, Rosé, Bulles)
   - Budget maximum
   - Intensité aromatique (Léger, Moyen, Intense, Fort)
   
3. **Cliquez sur "🍷 Trouver mon vin"**

4. **Consultez les recommandations** :
   - Cartes de vins avec scores de similarité
   - Justifications IA (expandable)
   - Analyse des accords mets-vins
   - Graphiques de visualisation

#### Onglet "Analyses & Statistiques"
- KPIs de la base de données
- Statistiques descriptives
- Graphiques de répartition (prix, régions, types)
- Profil sensoriel moyen

---

## 🔬 Architecture Technique

### Flux de traitement

```
Requête Utilisateur
    ↓
[1] Extraction du plat (FoodPairingMatcher)
    → Détecte : viande rouge/blanche, poisson, fromage
    ↓
[2] Enrichissement IA (si requête < 5 mots)
    → Ajoute termes œnologiques pertinents
    ↓
[3] Vectorisation SBERT de la requête
    → Vecteur 384 dimensions
    ↓
[4] Calcul Similarité Cosinus avec tous les vins
    → Top 50 résultats par similarité
    ↓
[5] Filtrage et Pénalités
    → Exclusion vins inappropriés
    → Bonus/malus selon contexte
    ↓
[6] Tri et Sélection Top N
    → Recommandations finales
    ↓
[7] Génération justifications IA
    → Explications personnalisées
```

### Technologies utilisées

- **SBERT** : `paraphrase-multilingual-MiniLM-L12-v2`
  - Multilingue (français, anglais)
  - 384 dimensions (rapide)
  - Optimisé pour similarité sémantique

- **GenAI** : GPT-4o-mini (OpenAI) ou Gemini (Google)
  - Enrichissement de requêtes
  - Génération de justifications
  - Cache pour optimiser les coûts

- **Streamlit** : Interface utilisateur
- **Plotly** : Visualisations interactives
- **Pandas** : Manipulation de données

### Scoring d'affinité

Le score final combine :
1. **Score sémantique** (base) : Similarité cosinus (0-1)
2. **Pénalités de filtres** : Budget dépassé (-50%), type incompatible (exclusion)
3. **Ajustements contextuels** :
   - Bonus si accords compatibles (+10-15%)
   - Pénalité si accords incompatibles (-40-60%)
   - Exclusion si explicitement incompatible
4. **Ajustements gustatifs** : Selon intensité aromatique demandée

---

## 📊 Base de Données

### Structure du CSV

Le fichier CSV doit contenir les colonnes suivantes :

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `ID` | Nombre | Identifiant unique | `1` |
| `Nom_du_Vin` | Texte | Nom du vin | `"Château Margaux 2018"` |
| `Type` | Texte | Type de vin | `Rouge`, `Blanc`, `Rosé`, `Bulles` |
| `Region` | Texte | Région d'origine | `Bordeaux`, `Bourgogne` |
| `Cepages` | Texte | Cépages utilisés | `"Cabernet Sauvignon, Merlot"` |
| `Prix` | Texte | Prix formaté | `"€45,00"` |
| `Description_Narrative` | Texte | Description textuelle | `"Un vin corsé avec des tanins..."` |
| `Mots_Cles` | Texte | Mots-clés (virgules) | `"fruité, tanins, corsé"` |
| `Accords_Mets` | Texte | Accords mets-vins | `"Bœuf, entrecôte, agneau"` |

### Données actuelles
- **200 vins** français
- **10 régions** : Bordeaux, Bourgogne, Champagne, Loire, Rhône, Provence, Alsace, Sud-Ouest, Languedoc, Beaujolais
- **5 types** : Rouge, Blanc, Rosé, Bulles, Liquoreux
- **Prix** : De 6€ à 1200€

---

## 🎨 Fonctionnalités Avancées

### Filtrage Intelligent

#### Détection automatique du contexte
- **Viande rouge** → Exclut rosés et blancs, priorise rouges avec accords "bœuf", "entrecôte"
- **Viande blanche** → Exclut rouges corsés, garde blancs, rosés, rouges légers
- **Poisson** → Exclut tous les rouges, garde blancs, rosés, bulles
- **Apéro** → Bonus pour vins "léger", "désaltérant", "soif"
- **Fromage frais** → Bonus pour vins avec "charcuterie", "apéro"

#### Détection de négations
Le système détecte et exclut automatiquement les vins qui disent explicitement :
- "Ce n'est pas un vin d'apéro"
- "Pas pour viande blanche"
- "À éviter avec poisson"

### Enrichissement IA

Si la requête est trop courte (< 5 mots), l'IA enrichit automatiquement :
- **Input** : "vin apéro"
- **Output** : "vin d'apéritif léger et désaltérant, adapté pour un moment convivial"

### Justifications Personnalisées

Chaque recommandation inclut :
- **Note de dégustation** : Pourquoi ce vin correspond à votre recherche
- **Analyse des accords** : Explication pédagogique des accords mets-vins
- **Conseils pratiques** : Température de service, moment idéal

---

## 🔧 Configuration Avancée

### Modèle SBERT

Par défaut : `paraphrase-multilingual-MiniLM-L12-v2` (rapide, 384 dimensions)

Pour un modèle plus précis (mais plus lent), modifiez dans `semantic_search.py` :
```python
self.model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")  # 768 dimensions
```

### Modèle GenAI

Par défaut : `gpt-4o-mini` (économique)

Pour GPT-4 (plus performant mais plus cher), modifiez dans `genai_integration.py` :
```python
self.model = "gpt-4"
```

Pour utiliser Gemini au lieu d'OpenAI :
```python
# Dans genai_integration.py, utilisez Gemini
from google import genai
```

---

## 📈 Performance

### Métriques
- **Temps de recherche** : < 1 seconde pour 200 vins
- **Embeddings** : 384 dimensions par vin
- **Précision** : Filtrage intelligent exclut les vins inappropriés
- **Couverture** : 200 vins, 10 régions, 5 types

### Optimisations
- **Embeddings pré-calculés** : Calculés une fois, sauvegardés dans `wine_embeddings.pkl`
- **Cache GenAI** : Réduction des appels API (sauvegarde dans `genai_cache.db`)
- **Filtrage précoce** : Exclusion des vins inappropriés avant calcul de similarité

---

## 🐛 Dépannage

### Erreur lors du chargement du modèle SBERT
- **Cause** : Première utilisation, téléchargement du modèle
- **Solution** : Vérifiez votre connexion internet, attendez le téléchargement (~100 MB)
- **Note** : Le modèle sera mis en cache pour les utilisations suivantes

### Erreur "OPENAI_API_KEY not found"
- **Cause** : Clé API non configurée
- **Solution** : Créez un fichier `.env` avec `OPENAI_API_KEY=your_key`
- **Alternative** : L'application fonctionne sans GenAI (fonctionnalités limitées)

### L'application est lente
- **Cause** : Calcul des embeddings en cours
- **Solution** : Attendez la fin du calcul (2-3 minutes la première fois)
- **Note** : Les embeddings sont sauvegardés dans `wine_embeddings.pkl`

### Aucun résultat trouvé
- **Cause** : Filtres trop stricts ou requête trop spécifique
- **Solution** : 
  - Élargissez les filtres (budget, type)
  - Simplifiez la requête
  - Vérifiez que la base de données est chargée

### Les justifications IA ne s'affichent pas
- **Cause** : Clé API non configurée ou erreur API
- **Solution** : Vérifiez votre clé API dans `.env`
- **Alternative** : Les recommandations fonctionnent sans justifications IA

---

## 📝 Notes Importantes

- **Embeddings** : Calculés une fois et sauvegardés. Si vous modifiez le CSV, supprimez `wine_embeddings.pkl` pour recalculer.
- **Cache GenAI** : Les justifications sont mises en cache pour éviter les appels API répétés.
- **Fonctionnement sans API** : L'application fonctionne sans clé API, mais les fonctionnalités GenAI seront désactivées.
- **Encodage** : Le CSV doit être en UTF-8 avec BOM pour gérer les accents français.

---

## 🎯 Exemples d'Utilisation

### Exemple 1 : Viande Rouge
```
Requête : "Je cherche un vin pour une côte de bœuf"
→ Détecte : viande rouge
→ Exclut : rosés, blancs
→ Recommande : Vins rouges avec accords "bœuf", "entrecôte"
→ Score : 70-80%
```

### Exemple 2 : Apéro Fromage
```
Requête : "Je veux vin apero fromage frais"
→ Détecte : apéro + fromage
→ Bonus : Vins avec "charcuterie", "soif", "désaltérant"
→ Exclut : Vins "pas d'apéro"
→ Score : 65-75%
```

### Exemple 3 : Poisson
```
Requête : "Vin pour accompagner un saumon"
→ Détecte : poisson
→ Exclut : Tous les rouges
→ Recommande : Blancs, rosés avec accords "poisson", "fruits de mer"
→ Score : 70-80%
```

---

## 🔬 Documentation Technique

Pour plus de détails techniques, consultez :
- **EXPLICATION_TECHNIQUE.md** : Architecture détaillée, SBERT, scoring
- **EXPLICATION_FLUX.md** : Flux de recherche étape par étape
- **STRUCTURE_BDD_OPTIMALE.md** : Guide pour créer une base de données optimale

---

## 📊 Métriques d'Évaluation

Le système inclut des métriques formelles d'évaluation (module `evaluation_metrics.py`) :

- **Précision@K** : Proportion de vins pertinents dans les K premiers résultats
- **Recall@K** : Proportion de vins pertinents retrouvés dans les K premiers
- **NDCG@K** : Normalized Discounted Cumulative Gain (qualité du ranking)
- **MRR** : Mean Reciprocal Rank (position du premier vin pertinent)
- **Qualité justifications** : Métriques pour évaluer les justifications IA (longueur, cohérence, explication, conseils)

Pour plus de détails, consultez :
- **VERIFICATION_COMPETENCES.md** : Vérification complète des compétences de la grille A.5
- **OBJECTIFS_PERFORMANCE.md** : Objectifs de performance et mesures actuelles
- **evaluation_metrics.py** : Implémentation des métriques

---

## 🚀 Améliorations Futures

- [x] Métriques d'évaluation quantitative (précision@K, recall@K) ✅
- [ ] Dataset de validation avec labels (en cours)
- [ ] Tests utilisateurs (A/B testing)
- [ ] Comparaison avec baselines (TF-IDF, Word2Vec)
- [ ] Interface mobile responsive
- [ ] Recommandations basées sur l'historique utilisateur

---

## 📄 Licence

Ce projet est développé dans le cadre d'un projet académique sur l'IA générative.

---

## 👥 Auteur

Projet développé pour le cours d'IA Générative - Mastère.

---

**Bon dégustation ! 🍷**
