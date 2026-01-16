"""
Module d'évaluation des performances du système de recommandation
Métriques formelles pour C5.3-C1 : Métriques d'évaluation pertinentes
"""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import ndcg_score


def precision_at_k(recommended: List[Dict], relevant: List[str], k: int = 3) -> float:
    """
    Calcule la précision@K : proportion de vins pertinents dans les K premiers
    
    Args:
        recommended: Liste des vins recommandés (avec 'nom' ou 'id')
        relevant: Liste des IDs ou noms de vins pertinents
        k: Nombre de recommandations à considérer (par défaut 3)
    
    Returns:
        Précision@K (0-1)
    """
    if len(recommended) == 0:
        return 0.0
    
    top_k = recommended[:k]
    # Créer un set de noms pertinents normalisés (sans guillemets, accents, etc.)
    relevant_set = set()
    for r in relevant:
        r_clean = str(r).lower().strip().replace('"', '').replace("'", '').replace(' ', '')
        relevant_set.add(r_clean)
        # Ajouter aussi les mots-clés individuels (ex: "Sancerre" pour "Sancerre Blanc")
        for word in r_clean.split():
            if len(word) > 3:  # Ignorer les mots trop courts
                relevant_set.add(word)
    
    hits = 0
    for wine in top_k:
        wine_nom = str(wine.get('nom', wine.get('id', ''))).lower().strip()
        wine_nom_clean = wine_nom.replace('"', '').replace("'", '').replace(' ', '')
        
        # Correspondance exacte
        if wine_nom_clean in relevant_set or wine_nom in relevant_set:
            hits += 1
            continue
        
        # Correspondance partielle : vérifier si des mots-clés du vin sont dans relevant_set
        wine_words = set(wine_nom_clean.split())
        if wine_words & relevant_set:  # Intersection non vide
            hits += 1
            continue
        
        # Correspondance par région/type (ex: "Sancerre" matche "Sancerre Blanc")
        for relevant_name in relevant_set:
            if len(relevant_name) > 4 and relevant_name in wine_nom_clean:
                hits += 1
                break
    
    return hits / min(k, len(recommended))


def recall_at_k(recommended: List[Dict], relevant: List[str], k: int = 3) -> float:
    """
    Calcule le recall@K : proportion de vins pertinents retrouvés dans les K premiers
    
    Args:
        recommended: Liste des vins recommandés
        relevant: Liste des IDs ou noms de vins pertinents
        k: Nombre de recommandations à considérer
    
    Returns:
        Recall@K (0-1)
    """
    if len(relevant) == 0:
        return 0.0
    
    top_k = recommended[:k]
    # Créer un set de noms pertinents normalisés
    relevant_set = set()
    for r in relevant:
        r_clean = str(r).lower().strip().replace('"', '').replace("'", '').replace(' ', '')
        relevant_set.add(r_clean)
        # Ajouter aussi les mots-clés individuels
        for word in r_clean.split():
            if len(word) > 3:
                relevant_set.add(word)
    
    hits = 0
    for wine in top_k:
        wine_nom = str(wine.get('nom', wine.get('id', ''))).lower().strip()
        wine_nom_clean = wine_nom.replace('"', '').replace("'", '').replace(' ', '')
        
        # Correspondance exacte
        if wine_nom_clean in relevant_set or wine_nom in relevant_set:
            hits += 1
            continue
        
        # Correspondance partielle
        wine_words = set(wine_nom_clean.split())
        if wine_words & relevant_set:
            hits += 1
            continue
        
        # Correspondance par région/type
        for relevant_name in relevant_set:
            if len(relevant_name) > 4 and relevant_name in wine_nom_clean:
                hits += 1
                break
    
    return hits / len(relevant) if len(relevant) > 0 else 0.0


def ndcg_at_k(recommended: List[Dict], relevant: List[str], scores: List[float] = None, k: int = 3) -> float:
    """
    Calcule le NDCG@K (Normalized Discounted Cumulative Gain)
    Mesure la qualité du ranking en tenant compte de la position
    
    Args:
        recommended: Liste des vins recommandés
        relevant: Liste des IDs ou noms de vins pertinents
        scores: Scores de similarité (optionnel, utilisé pour le ranking)
        k: Nombre de recommandations à considérer
    
    Returns:
        NDCG@K (0-1)
    """
    if len(recommended) == 0 or len(relevant) == 0:
        return 0.0
    
    top_k = recommended[:k]
    relevant_set = set(str(r).lower() for r in relevant)
    
    # Créer le vecteur de relevances (1 si pertinent, 0 sinon)
    y_true = []
    y_score = []
    
    for i, wine in enumerate(top_k):
        wine_id = str(wine.get('nom', wine.get('id', ''))).lower()
        is_relevant = 1 if wine_id in relevant_set else 0
        y_true.append(is_relevant)
        
        # Utiliser le score fourni ou un score par défaut basé sur la position
        if scores and i < len(scores):
            y_score.append(scores[i])
        else:
            y_score.append(1.0 / (i + 1))  # Score décroissant avec la position
    
    # Calculer NDCG
    if sum(y_true) == 0:
        return 0.0
    
    # NDCG nécessite au moins 2 échantillons
    if len(y_true) < 2:
        return float(y_true[0]) if len(y_true) > 0 else 0.0
    
    try:
        ndcg = ndcg_score([y_true], [y_score], k=k)
        return float(ndcg)
    except:
        # Fallback : calcul manuel simplifié
        dcg = sum(y_true[i] / np.log2(i + 2) for i in range(len(y_true)))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(y_true), sum(y_true))))
        return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(recommended: List[Dict], relevant: List[str]) -> float:
    """
    Calcule le MRR (Mean Reciprocal Rank)
    Mesure la position du premier vin pertinent dans les recommandations
    
    Args:
        recommended: Liste des vins recommandés
        relevant: Liste des IDs ou noms de vins pertinents
    
    Returns:
        MRR (0-1)
    """
    if len(recommended) == 0 or len(relevant) == 0:
        return 0.0
    
    relevant_set = set(str(r).lower() for r in relevant)
    
    for i, wine in enumerate(recommended):
        wine_id = str(wine.get('nom', wine.get('id', ''))).lower()
        if wine_id in relevant_set:
            return 1.0 / (i + 1)
    
    return 0.0


def evaluate_recommendations(
    recommended: List[Dict],
    relevant: List[str],
    scores: List[float] = None,
    k_values: List[int] = [1, 3, 5]
) -> Dict[str, float]:
    """
    Évalue les recommandations avec plusieurs métriques
    
    Args:
        recommended: Liste des vins recommandés
        relevant: Liste des IDs ou noms de vins pertinents
        scores: Scores de similarité (optionnel)
        k_values: Liste des valeurs de K à évaluer
    
    Returns:
        Dictionnaire avec toutes les métriques
    """
    results = {}
    
    for k in k_values:
        results[f'precision@{k}'] = precision_at_k(recommended, relevant, k)
        results[f'recall@{k}'] = recall_at_k(recommended, relevant, k)
        results[f'ndcg@{k}'] = ndcg_at_k(recommended, relevant, scores, k)
    
    results['mrr'] = mean_reciprocal_rank(recommended, relevant)
    
    return results


def evaluate_justification_quality(justification: str) -> Dict[str, float]:
    """
    Évalue la qualité d'une justification générée par IA
    
    Args:
        justification: Texte de justification
    
    Returns:
        Dictionnaire avec métriques de qualité
    """
    if not justification:
        return {
            'length': 0,
            'coherence': 0.0,
            'has_explanation': 0.0,
            'has_practical_advice': 0.0
        }
    
    # Longueur (normalisée)
    length = len(justification.split())
    length_score = min(length / 100, 1.0)  # Normalisé sur 100 mots
    
    # Cohérence (présence de connecteurs logiques)
    coherence_words = ['car', 'caractérisé', 'notamment', 'ainsi', 'donc', 'parce que', 'grâce à']
    has_coherence = any(word in justification.lower() for word in coherence_words)
    coherence_score = 1.0 if has_coherence else 0.5
    
    # Présence d'explication (mots-clés explicatifs)
    explanation_words = ['pourquoi', 'caractérisé', 'profil', 'notes', 'arômes', 'saveurs']
    has_explanation = any(word in justification.lower() for word in explanation_words)
    explanation_score = 1.0 if has_explanation else 0.0
    
    # Présence de conseils pratiques
    practical_words = ['température', 'service', 'décantation', 'moment', 'accompagner']
    has_practical = any(word in justification.lower() for word in practical_words)
    practical_score = 1.0 if has_practical else 0.0
    
    return {
        'length': length,
        'length_score': length_score,
        'coherence': coherence_score,
        'has_explanation': explanation_score,
        'has_practical_advice': practical_score,
        'overall_quality': (length_score + coherence_score + explanation_score + practical_score) / 4
    }


def generate_evaluation_report(
    test_cases: List[Dict],
    recommended_wines: List[List[Dict]],
    k: int = 3
) -> Dict:
    """
    Génère un rapport d'évaluation complet
    
    Args:
        test_cases: Liste de cas de test avec requêtes et vins pertinents
        recommended_wines: Liste de listes de vins recommandés pour chaque cas
        k: Nombre de recommandations à considérer
    
    Returns:
        Rapport d'évaluation avec métriques moyennes
    """
    all_metrics = {
        'precision': [],
        'recall': [],
        'ndcg': [],
        'mrr': []
    }
    
    for i, test_case in enumerate(test_cases):
        if i >= len(recommended_wines):
            continue
        
        relevant = test_case.get('relevant_wines', [])
        recommended = recommended_wines[i]
        
        metrics = evaluate_recommendations(recommended, relevant, k_values=[k])
        
        all_metrics['precision'].append(metrics[f'precision@{k}'])
        all_metrics['recall'].append(metrics[f'recall@{k}'])
        all_metrics['ndcg'].append(metrics[f'ndcg@{k}'])
        all_metrics['mrr'].append(metrics['mrr'])
    
    # Calculer les moyennes
    report = {
        'num_test_cases': len(test_cases),
        'mean_precision': np.mean(all_metrics['precision']) if all_metrics['precision'] else 0.0,
        'mean_recall': np.mean(all_metrics['recall']) if all_metrics['recall'] else 0.0,
        'mean_ndcg': np.mean(all_metrics['ndcg']) if all_metrics['ndcg'] else 0.0,
        'mean_mrr': np.mean(all_metrics['mrr']) if all_metrics['mrr'] else 0.0,
        'std_precision': np.std(all_metrics['precision']) if all_metrics['precision'] else 0.0,
        'std_recall': np.std(all_metrics['recall']) if all_metrics['recall'] else 0.0,
    }
    
    return report
