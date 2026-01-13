# 🍷 Grappe - Système de Recommandation Œnologique

Système intelligent de recommandation de vins basé sur l'analyse sémantique (SBERT) et l'IA générative. Le système analyse les préférences gustatives et contextuelles de l'utilisateur pour proposer des vins personnalisés avec justifications générées par IA.

## 🎯 Fonctionnalités

### Entrées utilisateur
- **Recherche textuelle libre** : Description de l'occasion, du repas ou de l'ambiance recherchée
- **Profil gustatif** : Auto-déclaration sur échelles (astringence/tanins, acidité, intensité aromatique)
- **Questions guidées** : Interface interactive pour affiner la recherche (occasion, saison, plat, émotion)
- **Filtres** : Type de vin, région, budget, cépage

### Analyse sémantique
- **Embeddings SBERT** : Utilisation de modèles multilingues pour l'analyse sémantique
- **Similarité cosinus** : Calcul de proximité entre requête utilisateur et descriptions de vins
- **Scoring d'affinité** : Combinaison de similarité sémantique, filtres stricts et profil gustatif

### Intégration GenAI
- **Enrichissement de requêtes** : Amélioration automatique des descriptions utilisateur trop courtes
- **Justifications personnalisées** : Notes de dégustation expliquant pourquoi chaque vin est recommandé
- **Analyse pédagogique** : Synthèses sur les accords mets-vins

## 📋 Structure du projet

```
Grappe/
├── app.py                      # Application Streamlit principale
├── data_loader.py              # Chargement et traitement du CSV
├── semantic_search.py          # Embeddings SBERT et recherche sémantique
├── scoring.py                  # Système de scoring d'affinité
├── genai_integration.py        # Intégration OpenAI pour GenAI
├── requirements.txt            # Dépendances Python
├── .env.example                # Exemple de fichier d'environnement
├── Projet IA BDD Vins - BDD Vins.csv  # Base de données des vins
└── README.md                   # Ce fichier
```

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

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

4. **Configuration de l'API OpenAI** (optionnel, pour les fonctionnalités GenAI)
   - Créer un fichier `.env` à la racine du projet
   - Ajouter votre clé API OpenAI :
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   - Obtenez votre clé sur [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   
   **Note** : L'application fonctionne sans clé API, mais les fonctionnalités GenAI (enrichissement et justifications) seront désactivées.

## 💻 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

### Première utilisation

Lors du premier lancement, le système va :
1. Charger les données depuis le CSV
2. Télécharger le modèle SBERT (première fois uniquement)
3. Calculer les embeddings pour tous les vins (peut prendre quelques minutes)
4. Sauvegarder les embeddings pour les utilisations suivantes

### Utilisation de l'interface

#### Onglet "Recherche libre"
1. Décrivez votre recherche dans la zone de texte (occasion, repas, ambiance)
2. Ajustez les filtres dans la barre latérale (type, région, budget)
3. Définissez votre profil gustatif avec les sliders
4. Cliquez sur "Trouver mon vin"
5. Consultez les recommandations avec leurs justifications

#### Onglet "Questions guidées"
1. Répondez aux questions (occasion, saison, plat, émotion)
2. Les filtres et le profil gustatif s'appliquent également
3. Cliquez sur "Rechercher avec questions guidées"

## 🔧 Configuration avancée

### Modèle SBERT

Par défaut, le système utilise `paraphrase-multilingual-MiniLM-L12-v2` (rapide et multilingue).

Pour utiliser un modèle plus précis (mais plus lent), modifiez dans `semantic_search.py` :
```python
self.model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
```

### Modèle GenAI

Par défaut, le système utilise `gpt-4o-mini` (économique).

Pour utiliser GPT-4, modifiez dans `genai_integration.py` :
```python
self.model = "gpt-4"
```

## 📊 Structure des données

Le fichier CSV doit contenir les colonnes suivantes :
- `ID` : Identifiant unique
- `Nom_du_Vin` : Nom du vin
- `Type` : Type de vin (Rouge, Blanc, Rosé, Bulles, etc.)
- `Region` : Région d'origine
- `Cepages` : Cépages utilisés
- `Prix` : Prix au format "€XX,XX"
- `Description_Narrative` : Description textuelle du vin
- `Mots_Cles` : Mots-clés sensoriels (séparés par des virgules)
- `Accords_Mets` : Suggestions d'accords mets-vins

## 🎨 Fonctionnalités techniques

### Analyse sémantique
- **Embeddings** : Représentation vectorielle des descriptions de vins
- **Similarité cosinus** : Mesure de proximité entre requête et vins
- **Fusion de champs** : Combinaison de description, mots-clés et accords pour enrichir la recherche

### Scoring d'affinité
Le score final combine :
- **Score sémantique** (base) : Similarité cosinus entre requête et vin
- **Pénalités de filtres** : Si le vin ne respecte pas les contraintes strictes (budget, type, région)
- **Ajustements gustatifs** : Bonus/malus selon le profil gustatif de l'utilisateur

### GenAI
- **Enrichissement** : Expansion de requêtes courtes avec contexte œnologique
- **Justification** : Génération de notes de dégustation personnalisées
- **Pédagogie** : Explications des accords mets-vins

## 🐛 Dépannage

### Erreur lors du chargement du modèle SBERT
- Vérifiez votre connexion internet (premier téléchargement)
- Le modèle sera mis en cache pour les utilisations suivantes

### Erreur "OPENAI_API_KEY not found"
- Créez un fichier `.env` avec votre clé API
- Ou désactivez les fonctionnalités GenAI dans l'interface

### L'application est lente
- Le calcul des embeddings peut prendre quelques minutes la première fois
- Les embeddings sont sauvegardés dans `wine_embeddings.pkl` pour réutilisation
- Utilisez un modèle SBERT plus petit pour de meilleures performances

## 📝 Notes

- Les embeddings sont calculés une fois et sauvegardés pour accélérer les recherches suivantes
- Si vous modifiez le CSV, supprimez `wine_embeddings.pkl` pour recalculer les embeddings
- Le système fonctionne sans API OpenAI, mais avec des fonctionnalités limitées

## 📄 Licence

Ce projet est développé dans le cadre d'un projet académique sur l'IA générative.

## 👥 Auteur

Projet développé pour le cours d'IA Générative.

---

**Bon dégustation ! 🍷**
