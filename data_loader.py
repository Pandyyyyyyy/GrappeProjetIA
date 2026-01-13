"""
Module de chargement et traitement des données de vins
"""
import pandas as pd
import re
from typing import Dict, List, Optional


class WineDataLoader:
    """Charge et prépare les données de vins pour l'analyse"""
    
    def __init__(self, csv_path: str):
        """
        Initialise le chargeur de données
        
        Args:
            csv_path: Chemin vers le fichier CSV des vins
        """
        self.csv_path = csv_path
        self.df = None
        self.df_cleaned = None
        self.wines = []
        self.validation_errors = []
        self.quality_report = {}
        
    def load_data(self) -> pd.DataFrame:
        """Charge les données depuis le CSV"""
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            return self.df
        except Exception as e:
            raise Exception(f"Erreur lors du chargement du CSV: {e}")
    
    def normalize_text(self, text: str) -> str:
        """Normalise un texte (supprime accents, espaces multiples, etc.)"""
        if pd.isna(text) or text == '':
            return ''
        text = str(text).strip()
        # Normaliser les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def normalize_region(self, region: str) -> str:
        """Normalise le nom d'une région"""
        if pd.isna(region) or region == '':
            return ''
        region = str(region).strip()
        # Standardiser certaines variations
        region = region.replace('Sud-Ouest', 'Sud Ouest')
        region = region.replace('Rhône', 'Rhone')
        return region
    
    def validate_data(self) -> Dict:
        """
        Valide les données et retourne un rapport d'erreurs
        
        Returns:
            Dictionnaire avec les erreurs trouvées
        """
        if self.df is None:
            self.load_data()
        
        errors = {
            'duplicates': [],
            'missing_values': {},
            'invalid_prices': [],
            'invalid_ids': [],
            'empty_fields': {}
        }
        
        # Détecter les doublons
        if 'ID' in self.df.columns:
            duplicates = self.df[self.df.duplicated(subset=['ID'], keep=False)]
            if not duplicates.empty:
                errors['duplicates'] = duplicates[['ID', 'Nom_du_Vin']].to_dict('records')
        
        # Vérifier les valeurs manquantes
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                errors['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(self.df) * 100)
                }
        
        # Valider les prix
        if 'Prix' in self.df.columns:
            for idx, row in self.df.iterrows():
                prix_str = str(row.get('Prix', ''))
                prix_numeric = self._extract_price(prix_str)
                if prix_numeric <= 0 or prix_numeric > 1000:
                    errors['invalid_prices'].append({
                        'id': row.get('ID', 'N/A'),
                        'nom': row.get('Nom_du_Vin', 'N/A'),
                        'prix': prix_str
                    })
        
        # Valider les IDs
        if 'ID' in self.df.columns:
            invalid_ids = self.df[(self.df['ID'].isna()) | (self.df['ID'] <= 0)]
            if not invalid_ids.empty:
                errors['invalid_ids'] = invalid_ids[['ID', 'Nom_du_Vin']].to_dict('records')
        
        # Vérifier les champs vides critiques
        critical_fields = ['Nom_du_Vin', 'Type', 'Region']
        for field in critical_fields:
            if field in self.df.columns:
                empty = self.df[self.df[field].isna() | (self.df[field].astype(str).str.strip() == '')]
                if not empty.empty:
                    errors['empty_fields'][field] = len(empty)
        
        self.validation_errors = errors
        return errors
    
    def get_quality_report(self) -> Dict:
        """
        Génère un rapport de qualité des données
        
        Returns:
            Dictionnaire avec les métriques de qualité
        """
        if self.df is None:
            self.load_data()
        
        if not self.validation_errors:
            self.validate_data()
        
        total_rows = len(self.df)
        total_cols = len(self.df.columns)
        
        # Calculer la complétude
        completeness = {}
        for col in self.df.columns:
            non_null = self.df[col].notna().sum()
            completeness[col] = {
                'non_null': int(non_null),
                'null': int(total_rows - non_null),
                'completeness_pct': float(non_null / total_rows * 100) if total_rows > 0 else 0
            }
        
        # Statistiques sur les prix
        prix_stats = {}
        if 'Prix' in self.df.columns:
            prix_values = []
            for _, row in self.df.iterrows():
                prix_str = str(row.get('Prix', '€0,00'))
                prix_numeric = self._extract_price(prix_str)
                if prix_numeric > 0:
                    prix_values.append(prix_numeric)
            
            if prix_values:
                prix_stats = {
                    'min': float(min(prix_values)),
                    'max': float(max(prix_values)),
                    'mean': float(sum(prix_values) / len(prix_values)),
                    'count_valid': len(prix_values),
                    'count_invalid': total_rows - len(prix_values)
                }
        
        report = {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'completeness': completeness,
            'validation_errors': {
                'duplicates_count': len(self.validation_errors.get('duplicates', [])),
                'invalid_prices_count': len(self.validation_errors.get('invalid_prices', [])),
                'invalid_ids_count': len(self.validation_errors.get('invalid_ids', [])),
                'missing_values': self.validation_errors.get('missing_values', {}),
                'empty_fields': self.validation_errors.get('empty_fields', {})
            },
            'prix_statistics': prix_stats,
            'unique_values': {
                'types': len(self.df['Type'].unique()) if 'Type' in self.df.columns else 0,
                'regions': len(self.df['Region'].unique()) if 'Region' in self.df.columns else 0
            }
        }
        
        self.quality_report = report
        return report
    
    def clean_data(self) -> pd.DataFrame:
        """
        Nettoie et normalise les données
        
        Returns:
            DataFrame nettoyé
        """
        if self.df is None:
            self.load_data()
        
        df_cleaned = self.df.copy()
        
        # Normaliser les textes
        text_columns = ['Nom_du_Vin', 'Description_Narrative', 'Mots_Cles', 'Accords_Mets', 'Cepages']
        for col in text_columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].apply(self.normalize_text)
        
        # Normaliser les régions
        if 'Region' in df_cleaned.columns:
            df_cleaned['Region'] = df_cleaned['Region'].apply(self.normalize_region)
        
        # Normaliser les types
        if 'Type' in df_cleaned.columns:
            df_cleaned['Type'] = df_cleaned['Type'].apply(lambda x: self.normalize_text(x).capitalize() if pd.notna(x) else x)
        
        self.df_cleaned = df_cleaned
        return df_cleaned
    
    def preprocess_data(self) -> List[Dict]:
        """
        Prépare les données pour l'analyse sémantique
        ÉTAPE 2 : Fusionne UNIQUEMENT les 3 dernières colonnes de la BDD :
        - Description_Narrative
        - Mots_Cles
        - Accords_Mets
        """
        if self.df is None:
            self.load_data()
        
        # Nettoyer les données si ce n'est pas déjà fait
        if self.df_cleaned is None:
            self.clean_data()
        
        df_to_use = self.df_cleaned if self.df_cleaned is not None else self.df
        wines = []
        
        for _, row in df_to_use.iterrows():
            # FUSIONNER UNIQUEMENT LES 3 DERNIÈRES COLONNES DE LA BDD
            # IMPORTANT : Répéter Accords_Mets plusieurs fois pour lui donner plus de poids
            # dans l'embedding SBERT (priorité aux accords mets-vins)
            contexte_parts = []
            
            # 1. ACCORDS METS-VINS (RÉPÉTÉ 3 FOIS pour priorité absolue)
            # C'est l'information la plus importante pour la recherche par contexte/plat
            if pd.notna(row.get('Accords_Mets')):
                accords = str(row['Accords_Mets']).strip()
                if accords:
                    # Répéter 3 fois pour donner plus de poids dans l'embedding SBERT
                    contexte_parts.append(accords)
                    contexte_parts.append(accords)
                    contexte_parts.append(accords)
            
            # 2. MOTS-CLÉS (répété 2 fois pour donner du poids)
            if pd.notna(row.get('Mots_Cles')):
                mots_cles = str(row.get('Mots_Cles', '')).strip()
                if mots_cles:
                    contexte_parts.append(mots_cles)
                    contexte_parts.append(mots_cles)
            
            # 3. DESCRIPTION NARRATIVE (une seule fois, en dernier)
            # La description a été réduite dans la BDD, donc on l'utilise telle quelle
            if pd.notna(row.get('Description_Narrative')):
                desc_narrative = str(row.get('Description_Narrative', '')).strip()
                if desc_narrative:
                    contexte_parts.append(desc_narrative)
            
            # Créer la description fusionnée (uniquement les 3 dernières colonnes)
            full_description = " ".join(contexte_parts)
            
            # Extraire le prix numérique
            prix_str = str(row.get('Prix', '€0,00'))
            prix_numeric = self._extract_price(prix_str)
            
            wine = {
                'id': int(row.get('ID', 0)),
                'nom': str(row.get('Nom_du_Vin', '')),
                'type': str(row.get('Type', '')),
                'region': str(row.get('Region', '')),
                'cepages': str(row.get('Cepages', '')),
                'prix': prix_numeric,
                'prix_str': prix_str,
                'description_narrative': str(row.get('Description_Narrative', '')),
                'mots_cles': str(row.get('Mots_Cles', '')),
                'accords_mets': str(row.get('Accords_Mets', '')),
                'description_fusionnee': full_description  # Utilise le Super-Champ (Contexte_Complet)
            }
            
            wines.append(wine)
        
        self.wines = wines
        return wines
    
    def _extract_price(self, prix_str: str) -> float:
        """Extrait le prix numérique depuis une chaîne comme '€18,00'"""
        # Remplacer la virgule par un point et extraire les chiffres
        prix_clean = prix_str.replace('€', '').replace(',', '.').strip()
        try:
            return float(prix_clean)
        except:
            return 0.0
    
    def get_wine_by_id(self, wine_id: int) -> Optional[Dict]:
        """Récupère un vin par son ID"""
        for wine in self.wines:
            if wine['id'] == wine_id:
                return wine
        return None
    
    def get_wines_by_type(self, wine_type: str) -> List[Dict]:
        """Filtre les vins par type"""
        return [w for w in self.wines if w['type'].lower() == wine_type.lower()]
    
    def get_wines_by_region(self, region: str) -> List[Dict]:
        """Filtre les vins par région"""
        return [w for w in self.wines if region.lower() in w['region'].lower()]
    
    def get_all_regions(self) -> List[str]:
        """Retourne la liste de toutes les régions uniques"""
        if self.df is None:
            self.load_data()
        return sorted(self.df['Region'].dropna().unique().tolist())
    
    def get_all_types(self) -> List[str]:
        """Retourne la liste de tous les types uniques"""
        if self.df is None:
            self.load_data()
        return sorted(self.df['Type'].dropna().unique().tolist())
