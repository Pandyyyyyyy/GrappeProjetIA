"""
Module de matching des accords mets-vins
"""
import re
import unicodedata
from typing import Dict, List, Optional, Tuple


class FoodPairingMatcher:
    """Classe pour matcher les plats recherchés avec les accords mets-vins"""
    
    # Catégories de viandes et leurs correspondances
    MEAT_CATEGORIES = {
        'viande_rouge': [
            'viande rouge', 'bœuf', 'boeuf', 'entrecôte', 'côte de bœuf', 'côte de boeuf',
            'steak', 'bavette', 'rumsteck', 'onglet', 'tournedos', 'filet', 'rognons',
            'gigot', 'agneau', 'mouton', 'côtelette', 'carré', 'épaule',
            'veau', 'osso buco', 'jarret', 'côte de veau',
            'sanglier', 'chevreuil', 'cerf', 'gibier', 'faisan', 'perdrix', 'lièvre',
            'canard', 'magret', 'confit', 'cuisse de canard'
        ],
        'viande_blanche': [
            'viande blanche', 'poulet', 'poularde', 'chapon', 'coq', 'pintade',
            'dinde', 'dindonneau', 'turkey',
            'lapin', 'lapin de garenne',
            'porc', 'côtelette de porc', 'rôti de porc', 'échine', 'travers',
            'jambon', 'jambon cru', 'prosciutto', 'serrano',
            'volaille', 'volailles'
        ],
        'poisson': [
            'poisson', 'saumon', 'truite', 'thon', 'maquereau', 'sardine',
            'bar', 'loup', 'daurade', 'dorade', 'sole', 'turbot', 'saint-pierre',
            'cabillaud', 'morue', 'colin', 'lieu', 'merlan',
            'rouget', 'saint-jacques', 'coquille saint-jacques',
            'homard', 'langouste', 'crabe', 'tourteau',
            'huîtres', 'moules', 'coquillages', 'fruits de mer', 'crustacés'
        ],
        'fromage': [
            'fromage', 'fromages', 'chèvre', 'brebis', 'vache',
            'comté', 'roquefort', 'brie', 'camembert', 'munster',
            'chèvre frais', 'bleu', 'fourme', 'reblochon', 'morbier'
        ]
    }
    
    # Mots-clés qui indiquent un type de plat
    DISH_KEYWORDS = {
        'grillé': ['grillé', 'grillade', 'barbecue', 'bbq', 'braisé'],
        'en sauce': ['sauce', 'sauté', 'mijoté', 'braisé', 'daube', 'ragoût'],
        'fumé': ['fumé', 'smoked'],
        'épicé': ['épicé', 'piquant', 'curry', 'tajine', 'couscous']
    }
    
    def __init__(self):
        """Initialise le matcher"""
        pass
    
    def extract_dish_from_query(self, query: str) -> Dict[str, any]:
        """
        Extrait le plat de la requête utilisateur
        
        Args:
            query: Requête textuelle
            
        Returns:
            Dictionnaire avec le plat extrait et ses caractéristiques
        """
        query_lower = query.lower()
        
        extracted = {
            'dish': None,
            'meat_category': None,
            'cooking_method': None,
            'keywords': []
        }
        
        # Chercher les catégories de viande (recherche améliorée avec normalisation)
        # Normaliser la requête pour gérer les accents et variations (ex: "côte de bœuf" vs "cote de boeuf")
        query_normalized = unicodedata.normalize('NFKD', query_lower).encode('ascii', 'ignore').decode('ascii')
        
        for category, keywords in self.MEAT_CATEGORIES.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                keyword_normalized = unicodedata.normalize('NFKD', keyword_lower).encode('ascii', 'ignore').decode('ascii')
                
                # Recherche exacte (avec accents) - priorité
                if keyword_lower in query_lower:
                    extracted['meat_category'] = category
                    extracted['dish'] = keyword
                    break
                # Recherche normalisée (sans accents) pour "côte de bœuf" vs "cote de boeuf"
                elif keyword_normalized in query_normalized and len(keyword_normalized) > 2:
                    # Vérifier que le mot normalisé fait au moins 3 caractères pour éviter les faux positifs
                    extracted['meat_category'] = category
                    extracted['dish'] = keyword
                    break
            if extracted['meat_category']:
                break
        
        # Chercher les méthodes de cuisson
        for method, keywords in self.DISH_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    extracted['cooking_method'] = method
                    break
            if extracted['cooking_method']:
                break
        
        # Extraire d'autres mots-clés de plat
        dish_patterns = [
            r'\b(?:avec|pour|accompagner|manger)\s+([^,\.]+?)(?:,|\.|$)',
            r'\b(?:plat|repas|mets)\s+(?:de|du|des)?\s*([^,\.]+?)(?:,|\.|$)',
        ]
        
        for pattern in dish_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                extracted['keywords'].extend([m.strip() for m in matches])
        
        return extracted
    
    def check_food_pairing_compatibility(
        self,
        wine_accords: str,
        user_dish: Dict[str, any]
    ) -> Tuple[float, str]:
        """
        Vérifie la compatibilité entre les accords du vin et le plat recherché
        
        Args:
            wine_accords: Chaîne des accords mets-vins du vin
            user_dish: Dictionnaire du plat recherché
            
        Returns:
            Tuple (score_compatibilité, raison)
        """
        if not wine_accords or not user_dish.get('meat_category'):
            return 0.5, "Pas d'information suffisante"
        
        wine_accords_lower = wine_accords.lower()
        meat_category = user_dish.get('meat_category')
        
        # Score de compatibilité
        compatibility_score = 0.5  # Score neutre par défaut
        reason = ""
        
        # Vérifier la compatibilité selon la catégorie de viande
        if meat_category == 'viande_rouge':
            # Mots-clés compatibles avec viande rouge
            compatible_keywords = [
                'bœuf', 'boeuf', 'entrecôte', 'steak', 'bavette', 'côte de bœuf',
                'agneau', 'gigot', 'mouton', 'veau', 'gibier', 'sanglier',
                'canard', 'magret', 'confit', 'viande rouge', 'viandes rouges'
            ]
            
            # Mots-clés incompatibles (viande blanche)
            incompatible_keywords = [
                'poulet', 'poularde', 'chapon', 'dinde', 'volaille', 'volailles',
                'viande blanche', 'viandes blanches', 'poisson', 'saumon'
            ]
            
            # Vérifier les compatibles
            compatible_found = any(kw in wine_accords_lower for kw in compatible_keywords)
            incompatible_found = any(kw in wine_accords_lower for kw in incompatible_keywords)
            
            if compatible_found and not incompatible_found:
                compatibility_score = 1.0
                reason = "Accord parfait avec viande rouge"
            elif compatible_found and incompatible_found:
                compatibility_score = 0.6
                reason = "Accord mixte (mentionne aussi viande blanche)"
            elif incompatible_found:
                compatibility_score = 0.2  # Pénalité forte
                reason = "Accord principalement pour viande blanche"
            else:
                compatibility_score = 0.5
                reason = "Accord neutre"
        
        elif meat_category == 'viande_blanche':
            compatible_keywords = [
                'poulet', 'poularde', 'chapon', 'dinde', 'volaille', 'volailles',
                'viande blanche', 'viandes blanches', 'porc', 'lapin'
            ]
            
            incompatible_keywords = [
                'bœuf', 'boeuf', 'entrecôte', 'steak', 'agneau', 'gigot',
                'viande rouge', 'gibier', 'sanglier'
            ]
            
            compatible_found = any(kw in wine_accords_lower for kw in compatible_keywords)
            incompatible_found = any(kw in wine_accords_lower for kw in incompatible_keywords)
            
            if compatible_found and not incompatible_found:
                compatibility_score = 1.0
                reason = "Accord parfait avec viande blanche"
            elif compatible_found and incompatible_found:
                compatibility_score = 0.6
                reason = "Accord mixte"
            elif incompatible_found:
                compatibility_score = 0.2
                reason = "Accord principalement pour viande rouge"
            else:
                compatibility_score = 0.5
                reason = "Accord neutre"
        
        elif meat_category == 'poisson':
            compatible_keywords = [
                'poisson', 'saumon', 'truite', 'thon', 'bar', 'loup', 'sole',
                'fruits de mer', 'coquillages', 'crustacés', 'huîtres', 'moules'
            ]
            
            incompatible_keywords = [
                'bœuf', 'boeuf', 'steak', 'viande rouge', 'agneau', 'gibier'
            ]
            
            compatible_found = any(kw in wine_accords_lower for kw in compatible_keywords)
            incompatible_found = any(kw in wine_accords_lower for kw in incompatible_keywords)
            
            if compatible_found and not incompatible_found:
                compatibility_score = 1.0
                reason = "Accord parfait avec poisson"
            elif incompatible_found:
                compatibility_score = 0.2
                reason = "Accord principalement pour viande"
            else:
                compatibility_score = 0.5
                reason = "Accord neutre"
        
        # Bonus si le plat exact est mentionné
        if user_dish.get('dish'):
            dish_name = user_dish['dish'].lower()
            if dish_name in wine_accords_lower:
                compatibility_score = min(1.0, compatibility_score + 0.2)
                reason += f" (mentionne {dish_name})"
        
        return compatibility_score, reason
    
    def suggest_wine_type_from_context(self, query: str) -> Optional[str]:
        """
        Suggère automatiquement un type de vin basé sur le contexte de la requête
        (plat, occasion, etc.)
        
        Args:
            query: Requête textuelle de l'utilisateur
            
        Returns:
            Type de vin suggéré ('Rouge', 'Blanc', 'Rosé', 'Bulles') ou None
        """
        query_lower = query.lower()
        
        # Détection par catégorie de viande/plat
        if any(keyword in query_lower for keyword in self.MEAT_CATEGORIES['viande_rouge']):
            return 'Rouge'
        
        if any(keyword in query_lower for keyword in ['barbecue', 'bbq', 'grillade', 'grillé']):
            # Barbecue = généralement viande rouge
            if any(keyword in query_lower for keyword in ['entrecôte', 'steak', 'bœuf', 'boeuf', 'viande rouge']):
                return 'Rouge'
            # Sinon, peut être blanc/rosé pour poulet/poisson au barbecue
            if any(keyword in query_lower for keyword in ['poulet', 'poisson', 'saumon']):
                return 'Blanc'
            # Par défaut barbecue = rouge
            return 'Rouge'
        
        if any(keyword in query_lower for keyword in self.MEAT_CATEGORIES['viande_blanche']):
            # Viande blanche = Blanc ou Rosé
            if 'rosé' in query_lower:
                return 'Rosé'
            return 'Blanc'
        
        if any(keyword in query_lower for keyword in self.MEAT_CATEGORIES['poisson']):
            return 'Blanc'
        
        if any(keyword in query_lower for keyword in self.MEAT_CATEGORIES['fromage']):
            # Fromage = peut être rouge ou blanc selon le contexte
            if 'rouge' in query_lower:
                return 'Rouge'
            return 'Blanc'
        
        # Détection par occasion - peut suggérer plusieurs types
        if any(keyword in query_lower for keyword in ['apéritif', 'apéro', 'apéritif dînatoire']):
            # Pour apéritif, plusieurs types sont possibles
            if any(keyword in query_lower for keyword in ['été', 'piscine', 'chaud', 'frais', 'rosé']):
                # Contexte estival = rosé, bulles ou blanc
                return None  # Ne pas filtrer strictement, laisser la recherche sémantique proposer plusieurs types
            return 'Bulles'  # Par défaut pour apéritif
        
        if any(keyword in query_lower for keyword in ['été', 'piscine', 'plage', 'terrasse', 'chaud']):
            # Contexte estival = rosé, bulles ou blanc - ne pas filtrer strictement
            return None
        
        if any(keyword in query_lower for keyword in ['dîner romantique', 'romantique']):
            # Dîner romantique = généralement rouge, mais peut être blanc selon le plat
            if any(keyword in query_lower for keyword in ['poisson', 'saumon', 'blanc']):
                return 'Blanc'
            return 'Rouge'  # Par défaut
        
        if any(keyword in query_lower for keyword in ['soirée', 'fête', 'célébration']):
            return 'Bulles'
        
        # Si aucun indice, ne pas suggérer (None)
        return None
    
    def enhance_query_with_pairing(self, query: str) -> str:
        """
        Enrichit la requête avec des termes d'accords mets-vins
        
        Args:
            query: Requête originale
            
        Returns:
            Requête enrichie
        """
        dish_info = self.extract_dish_from_query(query)
        
        if dish_info.get('meat_category'):
            # Ajouter des termes pertinents selon la catégorie
            # Répéter les termes importants pour donner plus de poids dans SBERT
            if dish_info['meat_category'] == 'viande_rouge':
                # Enrichir avec des termes très spécifiques à la viande rouge
                # Répéter plusieurs fois pour donner plus de poids dans l'embedding
                query += " accord viande rouge bœuf entrecôte steak agneau gigot mouton veau gibier sanglier canard magret"
                query += " accord viande rouge bœuf entrecôte steak agneau"
                query += " bœuf entrecôte steak agneau veau gibier"
            elif dish_info['meat_category'] == 'viande_blanche':
                query += " accord viande blanche poulet volaille dinde porc lapin"
                query += " accord viande blanche poulet volaille"
                query += " poulet volaille dinde porc"
            elif dish_info['meat_category'] == 'poisson':
                query += " accord poisson fruits de mer saumon bar sole"
                query += " accord poisson fruits de mer"
                query += " poisson saumon bar sole"
        
        return query
