"""
Module de calcul des scores pour le système de recommandation de vins
Gère les filtres, pénalités et bonus pour affiner les recommandations
"""
import re
from typing import List, Dict, Tuple, Optional


class WineScorer:
    """Classe pour calculer les scores finaux des vins avec filtres, pénalités et bonus"""
    
    def __init__(self):
        """Initialise le scorer"""
        pass
    
    def calculate_scores(
        self,
        semantic_results: List[Tuple[Dict, float]],
        user_query: str,
        dish_info: Dict,
        preferred_type: Optional[str],
        filters: Dict,
        intensite_aromatique: int,
        top_n: int
    ) -> List[Tuple[Dict, float, float]]:
        """
        Calcule les scores finaux pour chaque vin avec filtres, pénalités et bonus
        """
        scored_wines = []
        query_lower = user_query.lower()
        query_lower_type = user_query.lower()
        
        for wine, semantic_score in semantic_results:
            if semantic_score < 0.4:
                continue
            
            if preferred_type:
                wine_type = wine.get('type', '').strip()
                if wine_type.lower() != preferred_type.lower():
                    continue
            
            final_score = semantic_score
            
            if filters.get('budget_max') and wine['prix'] > filters.get('budget_max', float('inf')):
                final_score *= 0.5
            
            wine_full_text = (wine.get('description_narrative', '') + " " + 
                            wine.get('mots_cles', '') + " " + 
                            wine.get('accords_mets', '')).lower()
            
            final_score = self._check_negations(wine_full_text, query_lower, final_score)
            
            if dish_info.get('meat_category') == 'viande_rouge':
                if self._should_exclude_for_red_meat(wine, query_lower_type):
                    continue
                final_score = self._apply_red_meat_scoring(wine, wine_full_text, final_score)
            
            elif dish_info.get('meat_category') == 'viande_blanche':
                if self._should_exclude_for_white_meat(wine):
                    continue
                final_score = self._apply_white_meat_scoring(wine, final_score)
            
            elif dish_info.get('meat_category') == 'poisson':
                if self._should_exclude_for_fish(wine):
                    continue
                final_score = self._apply_fish_scoring(wine, final_score)
            
            if any(word in query_lower_type for word in ['apéro', 'apero', 'apéritif', 'aperitif']):
                if self._should_exclude_for_apero(wine, query_lower_type, preferred_type):
                    continue
                final_score = self._apply_apero_scoring(wine, query_lower, query_lower_type, final_score)
            
            if any(word in query_lower for word in ['fromage frais', 'fromages frais', 'chèvre frais', 'fromage', 'fromages']):
                final_score = self._apply_cheese_scoring(wine, query_lower, final_score)
            
            final_score = self._apply_intensity_scoring(wine, intensite_aromatique, final_score)
            
            final_score = self._apply_taste_preferences_scoring(wine, query_lower, final_score)
            
            final_score = min(1.0, final_score)
            if final_score < 0.2:
                continue
            
            scored_wines.append((wine, final_score, semantic_score))
        
        scored_wines.sort(key=lambda x: x[1], reverse=True)
        return scored_wines[:top_n]
    
    def _check_negations(self, wine_full_text: str, query_lower: str, final_score: float) -> float:
        """Vérifie les négations dans les descriptions"""
        negation_patterns = [
            r"ce n'?est pas", r"n'?est pas", r"n'?est aucun", r"n'?est point",
            r"ne pas", r"pas un", r"pas de", r"jamais", r"aucun"
        ]
        
        contexts_to_check = []
        if any(word in query_lower for word in ['apéro', 'apero', 'apéritif', 'aperitif']):
            contexts_to_check = ['apéro', 'apero', 'apéritif', 'aperitif']
        elif any(word in query_lower for word in ['viande rouge', 'bœuf', 'boeuf', 'entrecôte', 'steak']):
            contexts_to_check = ['viande rouge', 'bœuf', 'boeuf', 'entrecôte', 'steak']
        elif any(word in query_lower for word in ['viande blanche', 'poulet', 'volaille']):
            contexts_to_check = ['viande blanche', 'poulet', 'volaille']
        elif any(word in query_lower for word in ['poisson', 'saumon', 'bar']):
            contexts_to_check = ['poisson', 'saumon', 'bar']
        
        for context in contexts_to_check:
            if context in wine_full_text:
                context_pos = wine_full_text.find(context)
                if context_pos != -1:
                    start = max(0, context_pos - 60)
                    end = min(len(wine_full_text), context_pos + len(context) + 60)
                    context_window = wine_full_text[start:end]
                    
                    for pattern in negation_patterns:
                        if re.search(pattern, context_window, re.IGNORECASE):
                            final_score *= 0.05
                            break
        
        return final_score
    
    def _should_exclude_for_red_meat(self, wine: Dict, query_lower_type: str) -> bool:
        """Vérifie si le vin doit être exclu pour viande rouge"""
        wine_type_lower = wine.get('type', '').lower()
        
        if 'rosé' in wine_type_lower or 'rose' in wine_type_lower or 'blanc' in wine_type_lower:
            return True
        
        wine_accords_check = wine.get('accords_mets', '').lower()
        wine_desc_check = wine.get('description_narrative', '').lower()
        wine_full_check = wine_accords_check + " " + wine_desc_check
        
        if 'veau' in wine_full_check and 'bœuf' not in wine_full_check and 'boeuf' not in wine_full_check and \
           'entrecôte' not in wine_full_check and 'steak' not in wine_full_check and \
           'agneau' not in wine_full_check and 'gigot' not in wine_full_check:
            return True
        
        return False
    
    def _apply_red_meat_scoring(self, wine: Dict, wine_full_text: str, final_score: float) -> float:
        """Applique le scoring pour viande rouge"""
        wine_accords = wine.get('accords_mets', '').lower()
        wine_description = wine.get('description_narrative', '').lower()
        wine_full = wine_accords + " " + wine_description
        
        incompatible_keywords = [
            'poulet', 'poularde', 'chapon', 'dinde', 'volaille', 'volailles', 
            'viande blanche', 'viandes blanches', 'porc',
            'poisson', 'saumon', 'truite', 'thon', 'bar', 'loup', 'sole', 'turbot',
            'fruits de mer', 'coquillages', 'crustacés', 'huîtres', 'moules', 'rouget'
        ]
        compatible_keywords = [
            'bœuf', 'boeuf', 'entrecôte', 'steak', 'bavette', 'rumsteck', 'onglet',
            'agneau', 'gigot', 'mouton', 'côtelette', 'carré', 'épaule',
            'gibier', 'viande rouge', 'viandes rouges', 'sanglier', 'chevreuil', 'cerf',
            'canard', 'magret', 'confit', 'côte de bœuf', 'côte de boeuf'
        ]
        
        explicit_incompatible_phrases = [
            'pour accompagner un poisson', 'pour poisson', 'avec poisson',
            'pour viande blanche', 'avec viande blanche', 'viande blanche sans',
            'accompagner un poisson', 'accompagner une viande blanche'
        ]
        
        has_explicit_incompatible_phrase = any(phrase in wine_full for phrase in explicit_incompatible_phrases)
        has_incompatible = any(kw in wine_full for kw in incompatible_keywords)
        has_compatible = any(kw in wine_full for kw in compatible_keywords)
        
        if has_explicit_incompatible_phrase:
            return 0.0
        if has_incompatible and not has_compatible:
            return 0.0
        elif not has_compatible and not has_incompatible:
            final_score *= 0.6
        elif has_incompatible and has_compatible:
            final_score *= 0.3
        elif has_compatible:
            final_score *= 1.15
            final_score = min(1.0, final_score)
        
        return final_score
    
    def _should_exclude_for_white_meat(self, wine: Dict) -> bool:
        """Vérifie si le vin doit être exclu pour viande blanche"""
        wine_type_lower = wine.get('type', '').lower()
        if 'rouge' in wine_type_lower:
            wine_text_check = (wine.get('mots_cles', '') + " " + wine.get('description_narrative', '')).lower()
            if any(word in wine_text_check for word in ['corsé', 'charpenté', 'puissant', 'tanins', 'structuré']):
                return True
        return False
    
    def _apply_white_meat_scoring(self, wine: Dict, final_score: float) -> float:
        """Applique le scoring pour viande blanche"""
        wine_accords = wine.get('accords_mets', '').lower()
        compatible_keywords = ['poulet', 'poularde', 'chapon', 'dinde', 'volaille', 'volailles', 
                              'viande blanche', 'viandes blanches', 'porc', 'lapin']
        incompatible_keywords = ['bœuf', 'boeuf', 'entrecôte', 'steak', 'agneau', 'gigot', 
                                'mouton', 'veau', 'gibier', 'viande rouge', 'viandes rouges', 'sanglier']
        
        has_compatible = any(kw in wine_accords for kw in compatible_keywords)
        has_incompatible = any(kw in wine_accords for kw in incompatible_keywords)
        
        if has_incompatible and not has_compatible:
            return 0.0
        elif has_incompatible and has_compatible:
            final_score *= 0.4
        elif has_compatible:
            final_score *= 1.1
            final_score = min(1.0, final_score)
        
        return final_score
    
    def _should_exclude_for_fish(self, wine: Dict) -> bool:
        """Vérifie si le vin doit être exclu pour poisson"""
        wine_type_lower = wine.get('type', '').lower()
        if 'rouge' in wine_type_lower:
            return True
        return False
    
    def _apply_fish_scoring(self, wine: Dict, final_score: float) -> float:
        """Applique le scoring pour poisson"""
        wine_accords = wine.get('accords_mets', '').lower()
        compatible_keywords = ['poisson', 'saumon', 'truite', 'thon', 'bar', 'loup', 'sole', 
                              'fruits de mer', 'coquillages', 'crustacés', 'huîtres', 'moules']
        incompatible_keywords = ['bœuf', 'boeuf', 'steak', 'viande rouge', 'viandes rouges', 
                                'agneau', 'gibier', 'poulet', 'volaille']
        
        has_compatible = any(kw in wine_accords for kw in compatible_keywords)
        has_incompatible = any(kw in wine_accords for kw in incompatible_keywords)
        
        if has_incompatible and not has_compatible:
            return 0.0
        elif has_incompatible and has_compatible:
            final_score *= 0.4
        elif has_compatible:
            final_score *= 1.1
            final_score = min(1.0, final_score)
        
        return final_score
    
    def _should_exclude_for_apero(self, wine: Dict, query_lower_type: str, preferred_type: Optional[str]) -> bool:
        """Vérifie si le vin doit être exclu pour apéro"""
        wine_type_lower = wine.get('type', '').lower()
        wine_text_apero = (wine.get('description_narrative', '') + " " + wine.get('mots_cles', '')).lower()
        
        if preferred_type and preferred_type.lower() != 'rouge':
            if 'rouge' in wine_type_lower:
                return True
        elif 'rouge' in wine_type_lower:
            if any(word in wine_text_apero for word in ['corsé', 'corse', 'charpenté', 'charpente', 
                                                         'puissant', 'généreux', 'structuré', 'tanins', 
                                                         'corps', 'mâche', 'mache', 'matière']):
                return True
        
        if any(word in query_lower_type for word in ['frais', 'fraiche', 'fraîche', 'léger', 'leger', 'légers', 'legers']):
            if any(word in wine_text_apero for word in ['corsé', 'corse', 'charpenté', 'charpente', 
                                                         'puissant', 'généreux', 'structuré', 'corps', 
                                                         'mâche', 'mache']):
                return True
        
        wine_full_text_apero = (wine.get('description_narrative', '') + " " + 
                               wine.get('accords_mets', '') + " " + 
                               wine.get('mots_cles', '')).lower()
        if any(phrase in wine_full_text_apero for phrase in [
            "ce n'est pas un vin d'apéro", "ce n'est pas un vin d'apero",
            "pas un vin d'apéro", "pas un vin d'apero",
            "pas un petit vin d'apéro", "pas un petit vin d'apero"
        ]):
            return True
        
        return False
    
    def _apply_apero_scoring(self, wine: Dict, query_lower: str, query_lower_type: str, final_score: float) -> float:
        """Applique le scoring pour apéro"""
        wine_full_text_apero = (wine.get('description_narrative', '') + " " + 
                               wine.get('accords_mets', '') + " " + 
                               wine.get('mots_cles', '')).lower()
        wine_type_lower = wine.get('type', '').lower()
        
        if 'blanc' in wine_type_lower or 'rosé' in wine_type_lower or 'rose' in wine_type_lower or 'bulle' in wine_type_lower:
            final_score *= 1.15
            final_score = min(1.0, final_score)
        
        meal_dishes = ['cassoulet', 'dinde', 'tarte', 'tartes', 'rôti', 'roti', 'gigot', 'entrecôte', 'steak', 
                      'côte de bœuf', 'cote de boeuf', 'côte de boeuf', 'cote de bœuf', 'bœuf', 'boeuf',
                      'canard', 'magret', 'poulet rôti', 'poulet roti', 'poularde', 'chapon', 'plat', 'plats',
                      'repas', 'dîner', 'diner', 'déjeuner', 'dejeuner', 'menu', 'recette', 'recettes']
        
        apero_foods = ['fromage', 'fromages', 'charcuterie', 'charcuteries', 'tapas', 'amuse-bouches', 
                      'amuse bouches', 'amuses-bouches', 'cacahuètes', 'cacahuetes', 'olives', 'biscuits',
                      'biscuit', 'chips', 'noix', 'noisettes', 'amandes', 'saucisson', 'saucissons',
                      'jambon', 'jambons', 'pâté', 'pate', 'pâtés', 'pates', 'rillettes', 'terrine']
        
        has_meal_dish = any(word in wine_full_text_apero for word in meal_dishes)
        has_apero_food = any(word in wine_full_text_apero for word in apero_foods)
        has_specific_dish = any(word in wine_full_text_apero for word in ['huîtres', 'huitres', 'crevettes', 
                                                                          'fruits de mer', 'coquillages', 
                                                                          'poisson', 'saumon', 'bar', 'sole'])
        has_apero_mention = any(word in wine_full_text_apero for word in ['apéro', 'apero', 'apéritif', 
                                                                          'aperitif', 'soif', 'désaltérant', 
                                                                          'desalterant', 'dimanche midi'])
        
        query_has_specific_dish = any(word in query_lower_type for word in ['huîtres', 'huitres', 'crevettes', 
                                                                           'fruits de mer', 'coquillages', 
                                                                           'poisson', 'saumon', 'bar', 'sole', 'plateau'])
        
        if any(word in wine_full_text_apero for word in ['apéro', 'apero', 'apéritif', 'aperitif', 
                                                          'soif', 'désaltérant', 'desalterant']):
            final_score *= 1.5
            final_score = min(1.0, final_score)
        elif has_apero_food:
            final_score *= 1.3
            final_score = min(1.0, final_score)
        elif any(word in query_lower for word in ['frais', 'fraiche', 'fraîche', 'fruité', 'fruite', 'fruit']) and \
             any(word in wine_full_text_apero for word in ['frais', 'fraiche', 'fraîche', 'fruité', 'fruite', 
                                                           'fruit', 'léger', 'leger', 'vif', 'citronné']):
            final_score *= 1.4
            final_score = min(1.0, final_score)
        elif not query_has_specific_dish:
            if has_meal_dish and not has_apero_mention:
                return 0.0
            if has_apero_food:
                final_score *= 1.3
                final_score = min(1.0, final_score)
            if has_apero_mention:
                final_score *= 1.4
                final_score = min(1.0, final_score)
            elif has_specific_dish:
                final_score *= 0.5
        elif not query_has_specific_dish and has_specific_dish:
            return 0.0
        elif any(word in wine_full_text_apero for word in ['léger', 'leger', 'frais', 'fraiche', 'fraîche', 
                                                           'simple', 'efficace', 'citronné', 'vif']):
            final_score *= 1.2
            final_score = min(1.0, final_score)
        
        return final_score
    
    def _apply_cheese_scoring(self, wine: Dict, query_lower: str, final_score: float) -> float:
        """Applique le scoring pour fromage"""
        wine_full_text_fromage = (wine.get('accords_mets', '') + " " + 
                                  wine.get('description_narrative', '')).lower()
        
        fromage_frais_keywords = ['fromage frais', 'fromages frais', 'chèvre frais', 'fromage de chèvre', 
                                 'fromages de chèvre', 'charcuterie', 'apéro', 'apero', 'apéritif']
        fromage_affine_keywords = ['roquefort', 'bleu', 'comté', 'fromage affiné', 'fromages affinés', 
                                  'fromage fort', 'fromages forts']
        
        has_fromage_frais = any(kw in wine_full_text_fromage for kw in fromage_frais_keywords)
        has_fromage_affine = any(kw in wine_full_text_fromage for kw in fromage_affine_keywords)
        
        if has_fromage_affine and not has_fromage_frais:
            final_score *= 0.5
        elif has_fromage_frais:
            final_score *= 1.15
            final_score = min(1.0, final_score)
        
        return final_score
    
    def _apply_intensity_scoring(self, wine: Dict, intensite_aromatique: int, final_score: float) -> float:
        """Applique le scoring selon l'intensité aromatique"""
        wine_text = (wine.get('mots_cles', '') + " " + wine.get('description_narrative', '')).lower()
        
        if intensite_aromatique == 4:
            if any(word in wine_text for word in ['léger', 'subtil', 'délicat', 'discret', 'fin']):
                final_score *= 0.4
            elif any(word in wine_text for word in ['fort', 'puissant', 'intense', 'explosif', 'aromatique', 'expressif']):
                final_score *= 1.1
        elif intensite_aromatique == 3:
            if any(word in wine_text for word in ['léger', 'subtil', 'délicat', 'discret']):
                final_score *= 0.5
        elif intensite_aromatique == 1:
            if any(word in wine_text for word in ['fort', 'puissant', 'intense', 'explosif']):
                final_score *= 0.4
            elif any(word in wine_text for word in ['léger', 'subtil', 'délicat', 'discret', 'fin']):
                final_score *= 1.1
        
        return final_score
    
    def _apply_taste_preferences_scoring(self, wine: Dict, query_lower: str, final_score: float) -> float:
        """Applique le scoring selon les préférences gustatives"""
        wine_text_prefs = (wine.get('description_narrative', '') + " " + wine.get('mots_cles', '')).lower()
        
        if any(word in query_lower for word in ['épicé', 'epice', 'épices', 'epices', 'spicy', 'épicée']):
            if any(word in wine_text_prefs for word in ['épicé', 'epice', 'épices', 'epices', 'épicée', 'poivre', 'poivré', 'épice']):
                final_score *= 1.2
                final_score = min(1.0, final_score)
            else:
                final_score *= 0.7
        
        if any(word in query_lower for word in ['fruité', 'fruite', 'fruit', 'fruits', 'fruity']):
            if any(word in wine_text_prefs for word in ['fruité', 'fruite', 'fruit', 'fruits', 'fruiteux', 'fruitée']):
                final_score *= 1.15
                final_score = min(1.0, final_score)
            else:
                final_score *= 0.8
        
        if any(word in query_lower for word in ['frais', 'fraiche', 'fraîche', 'fraîch', 'froid', 'froide', 
                                                 'désaltérant', 'desalterant']):
            if any(word in wine_text_prefs for word in ['frais', 'fraiche', 'fraîche', 'fraîch', 'froid', 'froide', 
                                                        'désaltérant', 'desalterant', 'léger', 'leger', 'soif', 
                                                        'citronné', 'citronne', 'vif', 'simple', 'efficace']):
                final_score *= 1.3
                final_score = min(1.0, final_score)
            else:
                if any(word in wine_text_prefs for word in ['corsé', 'corse', 'charpenté', 'charpente', 'puissant', 
                                                            'généreux', 'structuré', 'corps', 'mâche', 'mache', 'matière']):
                    final_score *= 0.4
        
        if any(word in query_lower for word in ['minéral', 'minerale', 'minéralité', 'mineralite', 'mineral']):
            if any(word in wine_text_prefs for word in ['minéral', 'minerale', 'minéralité', 'mineralite', 'mineral']):
                final_score *= 1.15
                final_score = min(1.0, final_score)
        
        if any(word in query_lower for word in ['corsé', 'corse', 'puissant', 'charpenté', 'charpente']):
            if any(word in wine_text_prefs for word in ['corsé', 'corse', 'puissant', 'charpenté', 'charpente', 'structuré']):
                final_score *= 1.15
                final_score = min(1.0, final_score)
            else:
                final_score *= 0.8
        
        return final_score
