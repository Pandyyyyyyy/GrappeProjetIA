"""
Module de cache pour les réponses GenAI
Permet de réduire les appels API en réutilisant les réponses déjà obtenues
"""
import json
import hashlib
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import sqlite3


class GenAICache:
    """
    Système de cache pour les réponses GenAI
    Utilise SQLite pour stocker les réponses et les réutiliser automatiquement
    """
    
    def __init__(self, cache_file: str = "genai_cache.db"):
        """
        Initialise le cache
        """
        self.cache_file = cache_file
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialise la base de données SQLite pour le cache"""
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        # Créer la table si elle n'existe pas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS genai_cache (
                cache_key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                use_count INTEGER DEFAULT 1
            )
        """)
        
        # Créer un index pour accélérer les recherches
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_key ON genai_cache(cache_key)
        """)
        
        conn.commit()
        conn.close()
    
    def _generate_cache_key(self, prompt: str, model: str, function_name: str) -> str:
        """
        Génère une clé de cache unique à partir du prompt, modèle et fonction
        """
        # Combiner tous les éléments pour créer une clé unique
        key_string = f"{function_name}:{model}:{prompt}"
        # Utiliser MD5 pour créer un hash (rapide et suffisant pour ce cas)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def get(self, prompt: str, model: str, function_name: str) -> Optional[str]:
        """
        Récupère une réponse depuis le cache
        """
        cache_key = self._generate_cache_key(prompt, model, function_name)
        
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        # Chercher dans le cache
        cursor.execute("""
            SELECT response, last_used, use_count 
            FROM genai_cache 
            WHERE cache_key = ?
        """, (cache_key,))
        
        result = cursor.fetchone()
        
        if result:
            response, last_used, use_count = result
            # Mettre à jour le compteur d'utilisation et la date
            cursor.execute("""
                UPDATE genai_cache 
                SET last_used = CURRENT_TIMESTAMP, use_count = use_count + 1
                WHERE cache_key = ?
            """, (cache_key,))
            conn.commit()
            conn.close()
            return response
        
        conn.close()
        return None
    
    def set(self, prompt: str, model: str, function_name: str, response: str):
        """
        Stocke une réponse dans le cache
        """
        cache_key = self._generate_cache_key(prompt, model, function_name)
        
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        # Insérer ou mettre à jour le cache
        cursor.execute("""
            INSERT OR REPLACE INTO genai_cache (cache_key, response, created_at, last_used, use_count)
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
        """, (cache_key, response))
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le cache
        """
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        # Nombre total d'entrées
        cursor.execute("SELECT COUNT(*) FROM genai_cache")
        total_entries = cursor.fetchone()[0]
        
        # Nombre total d'utilisations
        cursor.execute("SELECT SUM(use_count) FROM genai_cache")
        total_uses = cursor.fetchone()[0] or 0
        
        # Entrées les plus utilisées
        cursor.execute("""
            SELECT cache_key, use_count, last_used 
            FROM genai_cache 
            ORDER BY use_count DESC 
            LIMIT 5
        """)
        top_entries = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_entries': total_entries,
            'total_uses': total_uses,
            'top_entries': top_entries
        }
    
    def clear_old_entries(self, days: int = 30):
        """
        Supprime les entrées du cache plus anciennes que X jours
        """
        conn = sqlite3.connect(self.cache_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM genai_cache 
            WHERE last_used < datetime('now', '-' || ? || ' days')
        """, (days,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted
