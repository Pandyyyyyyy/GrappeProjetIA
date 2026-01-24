"""
Module d'analyse sémantique avec SBERT et similarité cosinus
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import pickle
import os


class SemanticWineSearch:
    """Système de recherche sémantique pour les vins"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialise le modèle SBERT
        """
        self.model_name = model_name
        self.model = None
        self.wine_embeddings = None
        self.wines = []
        self.embeddings_file = "wine_embeddings.pkl"
        
    def load_model(self):
        """Charge le modèle SBERT"""
        if self.model is None:
            print(f"Chargement du modèle SBERT: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("Modèle chargé avec succès!")
    
    def compute_embeddings(self, wines: List[Dict], force_recompute: bool = False):
        """
        Vectorise en SBERT la colonne fusionnée (description_fusionnee)
        Cette méthode vectorise les descriptions fusionnées des vins avec le modèle SBERT
        """
        self.wines = wines
        
        # Vérifier si les embeddings existent déjà
        if not force_recompute and os.path.exists(self.embeddings_file):
            print(f"Chargement des embeddings depuis {self.embeddings_file}...")
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.wine_embeddings = data['embeddings']
                self.wines = data['wines']
            print("Embeddings chargés avec succès!")
            return
        
        # Calculer les embeddings avec SBERT
        if self.model is None:
            self.load_model()
        
        print("Calcul des embeddings pour tous les vins...")
        # Vectoriser la colonne fusionnée (description_fusionnee) avec SBERT
        descriptions = [wine['description_fusionnee'] for wine in wines]
        self.wine_embeddings = self.model.encode(
            descriptions,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Sauvegarder pour réutilisation
        print(f"Sauvegarde des embeddings dans {self.embeddings_file}...")
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump({
                'embeddings': self.wine_embeddings,
                'wines': self.wines
            }, f)
        
        print(f"Embeddings calculés et sauvegardés pour {len(wines)} vins!")
    
    def search_similar(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Recherche les vins les plus similaires à la requête
        """
        return self.search_similar_in_wines(query, self.wines, top_k)
    
    def search_similar_in_wines(
        self, 
        query: str, 
        wines_subset: List[Dict], 
        top_k: int = 10
    ) -> List[Tuple[Dict, float]]:
        """
        Recherche les vins les plus similaires à la requête dans un sous-ensemble de vins
        """
        if self.model is None:
            self.load_model()
        
        if self.wine_embeddings is None or len(self.wines) == 0:
            raise ValueError("Les embeddings doivent être calculés avant la recherche!")
        
        if not wines_subset:
            return []
        
        # Vectoriser le texte utilisateur avec SBERT
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Créer un mapping des indices des vins filtrés vers les indices dans wine_embeddings
        # Les vins dans wines_subset doivent correspondre à ceux dans self.wines
        wine_ids = {wine.get('id'): idx for idx, wine in enumerate(self.wines)}
        filtered_indices = []
        wines_mapping = []  # Mapping pour retrouver le vin original
        
        for wine in wines_subset:
            wine_id = wine.get('id')
            if wine_id in wine_ids:
                idx_in_all = wine_ids[wine_id]
                filtered_indices.append(idx_in_all)
                wines_mapping.append(wine)  # Garder le vin original
        
        if not filtered_indices:
            return []
        
        # Calculer la similarité cosinus entre la requête vectorisée et les vins vectorisés
        filtered_embeddings = self.wine_embeddings[filtered_indices]
        similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]
        
        # Trier par similarité décroissante
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            # Utiliser le mapping pour retrouver le vin original
            wine_original = wines_mapping[idx]
            results.append((wine_original, float(similarities[idx])))
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[Dict, float]]]:
        """
        Recherche pour plusieurs requêtes en une fois
        """
        if self.model is None:
            self.load_model()
        
        if self.wine_embeddings is None or len(self.wines) == 0:
            raise ValueError("Les embeddings doivent être calculés avant la recherche!")
        
        # Encoder toutes les requêtes
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)
        
        # Calculer les similarités
        similarities = cosine_similarity(query_embeddings, self.wine_embeddings)
        
        all_results = []
        for sim_row in similarities:
            top_indices = np.argsort(sim_row)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append((self.wines[idx], float(sim_row[idx])))
            all_results.append(results)
        
        return all_results
