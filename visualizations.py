"""
Module de visualisations graphiques pour les données de vins
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Optional
import numpy as np


class WineVisualizations:
    """Classe pour créer des visualisations graphiques des données de vins"""
    
    def __init__(self, wines: List[Dict]):
        """
        Initialise le visualiseur avec les données
        
        Args:
            wines: Liste de dictionnaires de vins
        """
        self.wines = wines
        self.df = pd.DataFrame(wines)
    
    def plot_price_distribution(self, fig_height: int = 400) -> go.Figure:
        """
        Crée un histogramme de la distribution des prix
        
        Args:
            fig_height: Hauteur de la figure
            
        Returns:
            Figure Plotly
        """
        prix = self.df['prix'].replace(0, np.nan).dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=prix,
            nbinsx=30,
            name='Distribution des prix',
            marker_color='#8B0000',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Distribution des Prix des Vins',
            xaxis_title='Prix (€)',
            yaxis_title='Nombre de vins',
            height=fig_height,
            template='plotly_white',
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def plot_price_by_type(self, fig_height: int = 400) -> go.Figure:
        """
        Crée un graphique en barres des prix moyens par type
        
        Args:
            fig_height: Hauteur de la figure
            
        Returns:
            Figure Plotly
        """
        type_stats = []
        for wine_type in self.df['type'].unique():
            type_wines = self.df[self.df['type'] == wine_type]
            prix = type_wines['prix'].replace(0, np.nan).dropna()
            if len(prix) > 0:
                type_stats.append({
                    'Type': wine_type,
                    'Prix moyen': prix.mean(),
                    'Nombre': len(type_wines)
                })
        
        df_stats = pd.DataFrame(type_stats).sort_values('Prix moyen', ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_stats['Prix moyen'],
            y=df_stats['Type'],
            orientation='h',
            marker_color='#722F37',
            text=[f"{p:.2f}€" for p in df_stats['Prix moyen']],
            textposition='outside',
            name='Prix moyen'
        ))
        
        fig.update_layout(
            title='Prix Moyen par Type de Vin',
            xaxis_title='Prix moyen (€)',
            yaxis_title='Type de vin',
            height=fig_height,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def plot_type_distribution(self, fig_height: int = 400) -> go.Figure:
        """
        Crée un graphique en camembert de la distribution des types
        
        Args:
            fig_height: Hauteur de la figure
            
        Returns:
            Figure Plotly
        """
        type_counts = self.df['type'].value_counts()
        
        colors = {
            'Rouge': '#8B0000',
            'Blanc': '#F5DEB3',
            'Rosé': '#FFB6C1',
            'Bulles': '#FFF8DC',
            'Liquoreux': '#D4AF37',
            'Orange': '#FFA500'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.4,
            marker_colors=[colors.get(t, '#CCCCCC') for t in type_counts.index],
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title='Distribution des Types de Vins',
            height=fig_height,
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def plot_region_distribution(self, top_n: int = 15, fig_height: int = 500) -> go.Figure:
        """
        Crée un graphique en barres des régions les plus représentées
        
        Args:
            top_n: Nombre de régions à afficher
            fig_height: Hauteur de la figure
            
        Returns:
            Figure Plotly
        """
        region_counts = self.df['region'].value_counts().head(top_n)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=region_counts.index,
            y=region_counts.values,
            marker_color='#722F37',
            text=region_counts.values,
            textposition='outside',
            name='Nombre de vins'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Régions les Plus Représentées',
            xaxis_title='Région',
            yaxis_title='Nombre de vins',
            height=fig_height,
            template='plotly_white',
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
    
    def plot_price_boxplot_by_type(self, fig_height: int = 500) -> go.Figure:
        """
        Crée un box plot des prix par type
        
        Args:
            fig_height: Hauteur de la figure
            
        Returns:
            Figure Plotly
        """
        data = []
        labels = []
        
        for wine_type in sorted(self.df['type'].unique()):
            type_wines = self.df[self.df['type'] == wine_type]
            prix = type_wines['prix'].replace(0, np.nan).dropna()
            if len(prix) > 0:
                data.append(prix.tolist())
                labels.append(wine_type)
        
        fig = go.Figure()
        
        for i, (label, prices) in enumerate(zip(labels, data)):
            fig.add_trace(go.Box(
                y=prices,
                name=label,
                boxpoints='outliers',
                marker_color='#8B0000' if i % 2 == 0 else '#722F37'
            ))
        
        fig.update_layout(
            title='Distribution des Prix par Type de Vin (Box Plot)',
            yaxis_title='Prix (€)',
            xaxis_title='Type de vin',
            height=fig_height,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def plot_price_vs_region(self, top_n_regions: int = 10, fig_height: int = 500) -> go.Figure:
        """
        Crée un graphique en barres groupées des prix moyens par région
        
        Args:
            top_n_regions: Nombre de régions à afficher
            fig_height: Hauteur de la figure
            
        Returns:
            Figure Plotly
        """
        # Sélectionner les top régions par nombre de vins
        top_regions = self.df['region'].value_counts().head(top_n_regions).index
        
        region_stats = []
        for region in top_regions:
            region_wines = self.df[self.df['region'] == region]
            prix = region_wines['prix'].replace(0, np.nan).dropna()
            if len(prix) > 0:
                region_stats.append({
                    'Région': region,
                    'Prix moyen': prix.mean(),
                    'Prix médian': prix.median(),
                    'Nombre': len(region_wines)
                })
        
        df_stats = pd.DataFrame(region_stats).sort_values('Prix moyen', ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Prix moyen',
            x=df_stats['Région'],
            y=df_stats['Prix moyen'],
            marker_color='#8B0000',
            text=[f"{p:.1f}€" for p in df_stats['Prix moyen']],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            name='Prix médian',
            x=df_stats['Région'],
            y=df_stats['Prix médian'],
            marker_color='#D4AF37',
            text=[f"{p:.1f}€" for p in df_stats['Prix médian']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'Prix Moyen et Médian par Région (Top {top_n_regions})',
            xaxis_title='Région',
            yaxis_title='Prix (€)',
            height=fig_height,
            template='plotly_white',
            barmode='group',
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def plot_keywords_cloud_data(self, top_n: int = 20) -> Dict:
        """
        Prépare les données pour un nuage de mots (retourne les fréquences)
        
        Args:
            top_n: Nombre de mots-clés à retourner
            
        Returns:
            Dictionnaire avec les fréquences
        """
        from collections import Counter
        
        all_keywords = []
        for keywords_str in self.df['mots_cles'].dropna():
            keywords = [k.strip() for k in str(keywords_str).split(',')]
            all_keywords.extend(keywords)
        
        keyword_counter = Counter(all_keywords)
        return dict(keyword_counter.most_common(top_n))
    
    def plot_cepage_frequency(self, top_n: int = 15, fig_height: int = 500) -> go.Figure:
        """
        Crée un graphique des cépages les plus fréquents
        
        Args:
            top_n: Nombre de cépages à afficher
            fig_height: Hauteur de la figure
            
        Returns:
            Figure Plotly
        """
        from collections import Counter
        
        all_cepages = []
        for cepages_str in self.df['cepages'].dropna():
            cepages = [c.strip() for c in str(cepages_str).split(',')]
            all_cepages.extend(cepages)
        
        cepage_counter = Counter(all_cepages)
        top_cepages = cepage_counter.most_common(top_n)
        
        cepages = [c[0] for c in top_cepages]
        counts = [c[1] for c in top_cepages]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cepages,
            y=counts,
            marker_color='#722F37',
            text=counts,
            textposition='outside',
            name='Fréquence'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Cépages les Plus Fréquents',
            xaxis_title='Cépage',
            yaxis_title='Nombre d\'occurrences',
            height=fig_height,
            template='plotly_white',
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
    
    def create_france_wine_map(self, wines: List[Dict], fig_height: int = 600) -> go.Figure:
        """
        Crée une carte interactive de la France avec les vins recommandés
        
        Args:
            wines: Liste des dictionnaires de vins recommandés
            fig_height: Hauteur de la figure
            
        Returns:
            Figure Plotly avec carte de la France
        """
        # Mapping des régions viticoles françaises vers leurs coordonnées approximatives
        region_coordinates = {
            'Bordeaux': {'lat': 44.8378, 'lon': -0.5792},
            'Bourgogne': {'lat': 47.0525, 'lon': 4.3833},
            'Champagne': {'lat': 49.2583, 'lon': 4.0317},
            'Alsace': {'lat': 48.5734, 'lon': 7.7521},
            'Loire': {'lat': 47.2184, 'lon': -0.5547},
            'Vallée de la Loire': {'lat': 47.2184, 'lon': -0.5547},
            'Rhône': {'lat': 45.7640, 'lon': 4.8357},
            'Rhone': {'lat': 45.7640, 'lon': 4.8357},
            'Provence': {'lat': 43.7102, 'lon': 7.2620},
            'Languedoc': {'lat': 43.6108, 'lon': 3.8767},
            'Languedoc-Roussillon': {'lat': 43.6108, 'lon': 3.8767},
            'Roussillon': {'lat': 42.6887, 'lon': 2.8947},
            'Beaujolais': {'lat': 46.0431, 'lon': 4.7234},
            'Jura': {'lat': 46.6756, 'lon': 5.5547},
            'Savoie': {'lat': 45.5646, 'lon': 5.9178},
            'Sud-Ouest': {'lat': 44.8378, 'lon': 1.1582},
            'Sud Ouest': {'lat': 44.8378, 'lon': 1.1582},
            'Corse': {'lat': 42.0396, 'lon': 9.0129},
            'Cognac': {'lat': 45.6956, 'lon': -0.3292},
            'Armagnac': {'lat': 43.6442, 'lon': 0.5866},
        }
        
        # Préparer les données pour la carte
        map_data = []
        for wine in wines:
            region = wine.get('region', '').strip()
            if not region:
                continue
            
            # Chercher les coordonnées de la région
            coords = None
            for reg_key, reg_coords in region_coordinates.items():
                if reg_key.lower() in region.lower() or region.lower() in reg_key.lower():
                    coords = reg_coords
                    break
            
            # Si pas trouvé, utiliser des coordonnées par défaut (centre de la France)
            if not coords:
                coords = {'lat': 46.6034, 'lon': 1.8883}  # Centre de la France
            
            map_data.append({
                'lat': coords['lat'],
                'lon': coords['lon'],
                'nom': wine['nom'],
                'type': wine['type'],
                'region': region,
                'cepages': wine.get('cepages', ''),
                'prix': wine.get('prix_str', ''),
            })
        
        if not map_data:
            # Carte vide avec message
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune région trouvée pour les vins recommandés",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="#722F37")
            )
            fig.update_layout(
                title="Carte des Vins Recommandés",
                height=fig_height,
                template='plotly_white'
            )
            return fig
        
        # Créer la carte
        df_map = pd.DataFrame(map_data)
        
        # Couleurs selon le type de vin
        color_map = {
            'Rouge': '#8B0000',
            'Blanc': '#F5DEB3',
            'Rosé': '#FFB6C1',
            'Bulles': '#FFF8DC',
            'Liquoreux': '#D4AF37',
            'Orange': '#FFA500'
        }
        
        fig = go.Figure()
        
        # Ajouter un marqueur pour chaque vin
        for wine_type in df_map['type'].unique():
            df_type = df_map[df_map['type'] == wine_type]
            color = color_map.get(wine_type, '#CCCCCC')
            
            fig.add_trace(go.Scattermapbox(
                lat=df_type['lat'],
                lon=df_type['lon'],
                mode='markers',
                marker=dict(
                    size=15,
                    color=color,
                    opacity=0.8
                ),
                text=df_type['nom'],
                customdata=df_type[['nom', 'type', 'region', 'cepages', 'prix']].values,
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    'Type: %{customdata[1]}<br>'
                    'Région: %{customdata[2]}<br>'
                    'Cépage: %{customdata[3]}<br>'
                    'Prix: %{customdata[4]}<br>'
                    '<extra></extra>'
                ),
                name=wine_type,
                showlegend=True
            ))
        
        # Configuration de la carte
        fig.update_layout(
            title=dict(
                text='🗺️ Carte de France - Vins Recommandés',
                x=0.5,
                xanchor='center',
                font=dict(size=20, family='Playfair Display', color='#2C1810')
            ),
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=46.6034, lon=2.2137),  # Centre de la France
                zoom=5.5
            ),
            height=fig_height,
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=80, b=0),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
