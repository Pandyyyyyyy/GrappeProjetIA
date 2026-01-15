"""
Module d'analyse exploratoire et statistique des données de vins
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


class WineDataAnalysis:
    """Classe pour effectuer des analyses statistiques sur les données de vins"""
    
    def __init__(self, wines: List[Dict]):
        """
        Initialise l'analyseur avec les données de vins
        
        Args:
            wines: Liste de dictionnaires de vins
        """
        self.wines = wines
        self.df = pd.DataFrame(wines)
    
    def get_descriptive_statistics(self) -> Dict:
        """
        Calcule les statistiques descriptives sur les prix
        
        Returns:
            Dictionnaire avec les statistiques
        """
        prix = self.df['prix'].replace(0, np.nan).dropna()
        
        if len(prix) == 0:
            return {}
        
        stats_dict = {
            'count': len(prix),
            'mean': float(prix.mean()),
            'median': float(prix.median()),
            'std': float(prix.std()),
            'min': float(prix.min()),
            'max': float(prix.max()),
            'q25': float(prix.quantile(0.25)),
            'q75': float(prix.quantile(0.75)),
            'iqr': float(prix.quantile(0.75) - prix.quantile(0.25))
        }
        
        return stats_dict
    
    def get_price_by_type(self) -> Dict[str, Dict]:
        """
        Calcule les statistiques de prix par type de vin
        
        Returns:
            Dictionnaire avec stats par type
        """
        result = {}
        
        for wine_type in self.df['type'].unique():
            type_wines = self.df[self.df['type'] == wine_type]
            prix = type_wines['prix'].replace(0, np.nan).dropna()
            
            if len(prix) > 0:
                result[wine_type] = {
                    'count': len(type_wines),
                    'mean_price': float(prix.mean()),
                    'median_price': float(prix.median()),
                    'min_price': float(prix.min()),
                    'max_price': float(prix.max())
                }
        
        return result
    
    def get_price_by_region(self) -> Dict[str, Dict]:
        """
        Calcule les statistiques de prix par région
        
        Returns:
            Dictionnaire avec stats par région
        """
        result = {}
        
        for region in self.df['region'].unique():
            region_wines = self.df[self.df['region'] == region]
            prix = region_wines['prix'].replace(0, np.nan).dropna()
            
            if len(prix) > 0:
                result[region] = {
                    'count': len(region_wines),
                    'mean_price': float(prix.mean()),
                    'median_price': float(prix.median()),
                    'min_price': float(prix.min()),
                    'max_price': float(prix.max())
                }
        
        return result
    
    def get_type_distribution(self) -> Dict[str, int]:
        """Retourne la distribution des types de vins"""
        return dict(self.df['type'].value_counts())
    
    def get_region_distribution(self) -> Dict[str, int]:
        """Retourne la distribution des régions"""
        return dict(self.df['region'].value_counts())
    
    def get_cepage_frequency(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Analyse la fréquence des cépages
        
        Args:
            top_n: Nombre de cépages les plus fréquents à retourner
            
        Returns:
            Liste de tuples (cépage, fréquence)
        """
        all_cepages = []
        
        for cepages_str in self.df['cepages'].dropna():
            # Séparer les cépages (souvent séparés par des virgules)
            cepages = [c.strip() for c in str(cepages_str).split(',')]
            all_cepages.extend(cepages)
        
        cepage_counter = Counter(all_cepages)
        return cepage_counter.most_common(top_n)
    
    def get_keywords_frequency(self, top_n: int = 30) -> List[Tuple[str, int]]:
        """
        Analyse la fréquence des mots-clés
        
        Args:
            top_n: Nombre de mots-clés les plus fréquents
            
        Returns:
            Liste de tuples (mot-clé, fréquence)
        """
        all_keywords = []
        
        for keywords_str in self.df['mots_cles'].dropna():
            keywords = [k.strip() for k in str(keywords_str).split(',')]
            all_keywords.extend(keywords)
        
        keyword_counter = Counter(all_keywords)
        return keyword_counter.most_common(top_n)
    
    def analyze_correlations(self) -> Dict:
        """
        Analyse les corrélations entre variables numériques
        
        Returns:
            Dictionnaire avec les corrélations
        """
        # Pour l'instant, on a principalement le prix comme variable numérique
        # On pourrait créer des variables numériques à partir des types/régions
        correlations = {}
        
        # Créer des variables binaires pour les types
        type_dummies = pd.get_dummies(self.df['type'], prefix='type')
        prix = self.df['prix'].replace(0, np.nan)
        
        # Corrélation prix vs type
        for col in type_dummies.columns:
            if prix.notna().sum() > 0:
                corr = prix.corr(type_dummies[col])
                if not pd.isna(corr):
                    correlations[col] = float(corr)
        
        return correlations
    
    def get_insights_report(self) -> Dict:
        """
        Génère un rapport d'insights automatique
        
        Returns:
            Dictionnaire avec les insights
        """
        insights = {
            'summary': {},
            'top_categories': {},
            'price_analysis': {},
            'recommendations': []
        }
        
        # Résumé général
        insights['summary'] = {
            'total_wines': len(self.df),
            'unique_types': len(self.df['type'].unique()),
            'unique_regions': len(self.df['region'].unique()),
            'price_range': {
                'min': float(self.df['prix'].replace(0, np.nan).min()),
                'max': float(self.df['prix'].replace(0, np.nan).max()),
                'mean': float(self.df['prix'].replace(0, np.nan).mean())
            }
        }
        
        # Top catégories
        type_dist = self.get_type_distribution()
        region_dist = self.get_region_distribution()
        
        insights['top_categories'] = {
            'most_common_type': max(type_dist.items(), key=lambda x: x[1])[0] if type_dist else None,
            'most_common_region': max(region_dist.items(), key=lambda x: x[1])[0] if region_dist else None,
            'type_distribution': dict(sorted(type_dist.items(), key=lambda x: x[1], reverse=True)[:5]),
            'region_distribution': dict(sorted(region_dist.items(), key=lambda x: x[1], reverse=True)[:5])
        }
        
        # Analyse des prix
        price_by_type = self.get_price_by_type()
        price_by_region = self.get_price_by_region()
        
        insights['price_analysis'] = {
            'most_expensive_type': max(price_by_type.items(), key=lambda x: x[1]['mean_price'])[0] if price_by_type else None,
            'most_affordable_type': min(price_by_type.items(), key=lambda x: x[1]['mean_price'])[0] if price_by_type else None,
            'price_by_type': price_by_type,
            'price_by_region': dict(sorted(price_by_region.items(), key=lambda x: x[1]['mean_price'], reverse=True)[:5])
        }
        
        # Recommandations basées sur les statistiques
        if price_by_type:
            most_expensive = max(price_by_type.items(), key=lambda x: x[1]['mean_price'])
            most_affordable = min(price_by_type.items(), key=lambda x: x[1]['mean_price'])
            
            insights['recommendations'].append(
                f"Le type de vin le plus cher en moyenne est {most_expensive[0]} "
                f"({most_expensive[1]['mean_price']:.2f}€), "
                f"tandis que {most_affordable[0]} est le plus accessible "
                f"({most_affordable[1]['mean_price']:.2f}€)."
            )
        
        if region_dist:
            top_region = max(region_dist.items(), key=lambda x: x[1])
            insights['recommendations'].append(
                f"La région la plus représentée est {top_region[0]} avec {top_region[1]} vins."
            )
        
        return insights
