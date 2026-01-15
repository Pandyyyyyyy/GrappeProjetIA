"""
Module d'intégration GenAI pour enrichir les descriptions et générer des justifications
Supporte Google Gemini 2.5 Flash (gratuit) et OpenAI avec système de cache
"""
import os
import warnings
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Import conditionnel pour éviter erreurs si packages non installés
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Supprimer l'avertissement de dépréciation pour google.generativeai
# Utiliser catch_warnings pour capturer l'avertissement lors de l'import
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    try:
        import google.generativeai as genai
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False

from genai_cache import GenAICache

load_dotenv()


class WineGenAI:
    """
    Intégration GenAI pour enrichir et justifier les recommandations
    Supporte Google Gemini 2.5 Flash (gratuit) et OpenAI avec cache automatique
    """
    
    def __init__(self, 
                 provider: str = "gemini",  # "gemini" ou "openai"
                 api_key: Optional[str] = None, 
                 model: Optional[str] = None):
        """
        Initialise le client GenAI avec support de plusieurs providers
        
        Args:
            provider: "gemini" (gratuit, recommandé) ou "openai"
            api_key: Clé API (ou depuis variable d'environnement)
            model: Modèle à utiliser (auto-détecté selon provider)
        """
        self.provider = provider.lower()
        self.model = model
        self.client = None
        self.genai_client = None  # Pour Gemini
        
        # Initialiser le cache
        self.cache = GenAICache()
        
        if self.provider == "gemini":
            # Google Gemini 2.5 Flash (gratuit)
            if not GEMINI_AVAILABLE:
                print("[WARNING] google-generativeai non installe. Installez avec: pip install google-generativeai")
                return
            
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            self.model = model or "gemini-2.5-flash"  # Modèle gratuit avec meilleur quota
            
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.genai_client = genai.GenerativeModel(self.model)
                print(f"[OK] Google Gemini configure (modele: {self.model})")
            else:
                print("[WARNING] GOOGLE_API_KEY ou GEMINI_API_KEY non trouvee. Les fonctionnalites GenAI seront desactivees.")
        
        elif self.provider == "openai":
            # OpenAI (payant)
            if not OPENAI_AVAILABLE:
                print("[WARNING] openai non installe. Installez avec: pip install openai")
                return
            
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model or "gpt-4o-mini"  # Modèle économique
            
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                print(f"[OK] OpenAI configure (modele: {self.model})")
            else:
                print("[WARNING] OPENAI_API_KEY non trouvee. Les fonctionnalites GenAI seront desactivees.")
        else:
            print(f"[WARNING] Provider '{provider}' non supporte. Utilisez 'gemini' ou 'openai'.")
    
    def _call_api(self, prompt: str, system_prompt: str, function_name: str, max_tokens: int = 200, temperature: float = 0.7) -> Optional[str]:
        """
        Appelle l'API GenAI avec cache automatique
        
        Args:
            prompt: Le prompt utilisateur
            system_prompt: Le prompt système
            function_name: Nom de la fonction (pour le cache)
            max_tokens: Nombre maximum de tokens
            temperature: Température pour la génération
            
        Returns:
            La réponse de l'API ou None en cas d'erreur
        """
        # Vérifier le cache d'abord
        full_prompt = f"{system_prompt}\n\n{prompt}"
        cached_response = self.cache.get(full_prompt, self.model, function_name)
        
        if cached_response:
            print(f"[CACHE] Cache hit pour {function_name} (economie d'appel API)")
            return cached_response
        
        # Appel API si pas en cache
        try:
            if self.provider == "gemini" and self.genai_client:
                # Google Gemini
                response = self.genai_client.generate_content(
                    f"{system_prompt}\n\n{prompt}",
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
                # Récupérer le texte complet - vérifier plusieurs méthodes
                if hasattr(response, 'text') and response.text:
                    result = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    # Récupérer depuis les candidates si disponible
                    text_parts = []
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    text_parts.append(part.text)
                    result = ' '.join(text_parts).strip() if text_parts else None
                else:
                    result = None
                
            elif self.provider == "openai" and self.client:
                # OpenAI
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                result = response.choices[0].message.content.strip()
            else:
                return None
            
            # Mettre en cache la réponse
            if result:
                self.cache.set(full_prompt, self.model, function_name, result)
                print(f"[CACHE] Reponse mise en cache pour {function_name}")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            # Gérer spécifiquement les erreurs de quota
            if "429" in error_msg or "quota" in error_msg.lower() or "Quota exceeded" in error_msg:
                print(f"[ERROR] Quota API depasse pour {function_name}. Utilisation du texte de secours.")
            else:
                print(f"[ERROR] Erreur lors de l'appel API ({function_name}): {e}")
            return None
    
    def enrich_user_query(self, user_query: str) -> str:
        """
        Enrichit une requête utilisateur trop courte avec du contexte œnologique
        CONTRAINTE: Appel API seulement si requête < 5 mots (strictement limité)
        
        Args:
            user_query: Requête originale de l'utilisateur
            
        Returns:
            Requête enrichie
        """
        # CONTRAINTE: Enrichir seulement si requête très courte (< 5 mots)
        word_count = len(user_query.split())
        if word_count >= 5:
            # Requête déjà assez longue, ne pas appeler l'API
            return user_query
        
        # Vérifier qu'un client est disponible
        if not (self.client or self.genai_client):
            return user_query
        
        system_prompt = "Tu es un sommelier expert qui enrichit les requêtes de recherche de vins."
        prompt = f"""L'utilisateur a écrit cette requête pour trouver un vin:
"{user_query}"

Enrichis cette requête en ajoutant des termes œnologiques pertinents, des descripteurs sensoriels, et des contextes d'usage, tout en conservant l'intention originale. Réponds UNIQUEMENT avec la requête enrichie, sans explication supplémentaire.

Requête enrichie:"""
        
        enriched = self._call_api(prompt, system_prompt, "enrich_user_query", max_tokens=150, temperature=0.7)
        return enriched if enriched else user_query
    
    def generate_recommendation_justification(
        self,
        wine: Dict,
        user_query: str,
        semantic_score: float
    ) -> str:
        """
        Génère une justification personnalisée pour une recommandation
        Utilise le cache pour éviter les appels API répétés
        
        Args:
            wine: Dictionnaire du vin recommandé
            user_query: Requête originale de l'utilisateur
            semantic_score: Score de similarité sémantique
            
        Returns:
            Justification textuelle
        """
        if not (self.client or self.genai_client):
            return self._generate_fallback_justification(wine, semantic_score)
        
        system_prompt = "Tu es un sommelier expert qui rédige des notes de dégustation personnalisées et convaincantes."
        prompt = f"""Un utilisateur cherche un vin avec cette description:
"{user_query}"

Tu recommandes ce vin:
- Nom: {wine['nom']}
- Type: {wine['type']}
- Région: {wine['region']}
- Cépages: {wine['cepages']}
- Prix: {wine['prix_str']}
- Description: {wine['description_narrative']}
- Mots-clés: {wine['mots_cles']}
- Accords mets: {wine['accords_mets']}

Écris une note de dégustation personnalisée de 3-4 phrases complètes qui:
1. Explique pourquoi ce vin correspond parfaitement à sa recherche
2. Met en avant les caractéristiques qui matchent avec sa demande
3. Donne des conseils pratiques d'utilisation (température, moment de dégustation, etc.)
4. Utilise un ton chaleureux, expert et accessible

IMPORTANT: Réponds UNIQUEMENT avec la note de dégustation, sans introduction ni conclusion. Commence directement par expliquer pourquoi ce vin convient.

Note de dégustation:"""
        
        justification = self._call_api(prompt, system_prompt, "generate_recommendation_justification", max_tokens=300, temperature=0.7)
        return justification if justification else self._generate_fallback_justification(wine, semantic_score)
    
    def generate_food_pairing_analysis(
        self,
        wine: Dict,
        dish: Optional[str] = None
    ) -> str:
        """
        Génère une analyse pédagogique de l'accord mets-vins
        Utilise le cache pour éviter les appels API répétés
        
        Args:
            wine: Dictionnaire du vin
            dish: Plat spécifique mentionné par l'utilisateur (optionnel)
            
        Returns:
            Analyse textuelle de l'accord
        """
        if not (self.client or self.genai_client):
            return self._generate_fallback_pairing(wine, dish)
        
        dish_context = f" pour accompagner {dish}" if dish else ""
        system_prompt = "Tu es un sommelier expert qui explique les accords mets-vins de manière pédagogique et détaillée."
        prompt = f"""Explique de manière pédagogique et détaillée pourquoi ce vin s'accorde bien avec les plats suggérés.

Vin: {wine['nom']} ({wine['type']}, {wine['region']})
Accords suggérés: {wine['accords_mets']}
Caractéristiques: {wine['mots_cles']}
Description: {wine['description_narrative']}

Écris une analyse complète de 3-4 phrases qui:
1. Explique les principes de l'accord mets-vins pour ce vin spécifique
2. Détaille pourquoi ces accords fonctionnent (complémentarité, contraste, harmonie)
3. Donne des conseils pratiques sur la température de service et le moment idéal
4. Utilise un ton pédagogique, accessible et expert

IMPORTANT: Réponds UNIQUEMENT avec l'analyse, sans introduction. Commence directement par expliquer les accords.

Analyse de l'accord:"""
        
        analysis = self._call_api(prompt, system_prompt, "generate_food_pairing_analysis", max_tokens=300, temperature=0.7)
        return analysis if analysis else self._generate_fallback_pairing(wine, dish)
    
    def _generate_fallback_justification(self, wine: Dict, semantic_score: float) -> str:
        """Génère une justification basique sans GenAI"""
        score_percent = int(semantic_score * 100)
        return f"""Ce {wine['type'].lower()} de {wine['region']} correspond à votre recherche avec un score de {score_percent}%. 
{wine['description_narrative']} 
Idéal pour: {wine['accords_mets']}"""
    
    def _generate_fallback_pairing(self, wine: Dict, dish: Optional[str] = None) -> str:
        """Génère une analyse basique sans GenAI"""
        return f"""Les accords suggérés pour ce {wine['type'].lower()} ({wine['mots_cles']}) sont: {wine['accords_mets']}. 
Ces associations fonctionnent grâce aux caractéristiques du vin qui complètent ou contrastent harmonieusement avec les saveurs des plats."""
    
    def generate_profile_summary(
        self,
        user_profile: str,
        top_wines: List[Dict],
        average_coverage: float
    ) -> str:
        """
        Génère une synthèse de profil œnologique (EF4.3) - Bio professionnelle
        CONTRAINTE: UN SEUL APPEL API pour la bio finale
        
        Args:
            user_profile: Profil textuel de l'utilisateur (requête enrichie)
            top_wines: Liste des vins recommandés
            average_coverage: Score de couverture moyen (similarité moyenne)
            
        Returns:
            Synthèse de profil textuelle (biographie professionnelle courte)
        """
        if not (self.client or self.genai_client):
            return self._generate_fallback_summary(user_profile, top_wines)
        
        wines_summary = "\n".join([
            f"- {wine['nom']} ({wine['type']}, {wine['region']})"
            for wine in top_wines[:3]
        ])
        
        system_prompt = "Tu es un sommelier expert qui rédige des profils œnologiques personnalisés dans un style Executive Summary."
        prompt = f"""Profil de l'utilisateur: {user_profile}

Vins recommandés (score de similarité moyen: {average_coverage:.1%}):
{wines_summary}

Génère une biographie professionnelle œnologique complète de 3-4 phrases (style Executive Summary) qui:
1. Caractérise le profil gustatif de l'utilisateur de manière accrocheuse et détaillée
2. Met en avant ses préférences et affinités avec des exemples concrets
3. Suggère des pistes d'exploration pour enrichir son expérience
4. Utilise un ton professionnel, engageant et accessible

IMPORTANT: Réponds UNIQUEMENT avec la biographie complète, sans introduction. Commence directement par caractériser le profil. Assure-toi que la réponse soit complète et se termine par une phrase de conclusion.

Biographie professionnelle:"""
        
        # UN SEUL APPEL API pour la bio finale
        summary = self._call_api(prompt, system_prompt, "generate_profile_summary", max_tokens=400, temperature=0.7)
        return summary if summary else self._generate_fallback_summary(user_profile, top_wines)
    
    def _generate_fallback_summary(self, user_profile: str, top_wines: List[Dict]) -> str:
        """Génère une synthèse basique sans GenAI"""
        wine_types = list(set([wine['type'] for wine in top_wines]))
        return f"Votre profil œnologique montre une affinité pour les vins de type {', '.join(wine_types)}. Vos préférences incluent: {user_profile[:100]}..."
