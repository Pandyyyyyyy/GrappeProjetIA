# 📊 Explication du Système de Scoring

## Questions Fréquentes sur le Calcul des Scores

### 1. Comment fonctionne le scoring global ?

Le système utilise un **score de similarité sémantique** (cosinus) comme base, puis applique des **pénalités** et des **bonus** selon différents critères.

**Formule générale :**
```
Score final = Score sémantique × Pénalités × Bonus
```

---

### 2. D'où vient le score sémantique initial ?

Le score sémantique (0-1) est calculé par **SBERT** (Sentence-BERT) qui compare la requête utilisateur avec la description complète du vin.

- **Score élevé (0.6-0.9)** : Le vin correspond très bien à la requête
- **Score moyen (0.4-0.6)** : Correspondance correcte
- **Score faible (< 0.3)** : Le vin est exclu automatiquement

**Exemple :**
- Requête : "vin pour apéro frais et fruité"
- Vin : "Sancerre Blanc - Frais, citronné, parfait pour l'apéro"
- Score sémantique : **0.72** (très bonne correspondance)

---

### 3. Quelles sont les pénalités appliquées ?

#### 3.1. Pénalité Budget
Si le vin dépasse le budget maximum :
```python
final_score *= 0.5  # Réduction de 50%
```

**Exemple :**
- Budget max : 20€
- Vin : 35€
- Score initial : 0.7
- Score après pénalité : **0.35**

---

#### 3.2. Pénalité Accords Incompatibles

**Pour viande rouge :**
- Si le vin mentionne SEULEMENT viande blanche/poisson → **EXCLUSION** (score = 0)
- Si le vin mentionne les deux (rouge ET blanc) → **Pénalité de 70%** (× 0.3)
- Si aucun accord mentionné → **Pénalité de 40%** (× 0.6)

**Exemple :**
- Requête : "vin pour entrecôte"
- Vin : "Chardonnay - Parfait avec poisson et fruits de mer"
- Résultat : **EXCLUSION** (score = 0)

---

#### 3.3. Pénalité Intensité Aromatique

Si l'utilisateur demande un vin **fort** mais le vin est **léger** :
```python
final_score *= 0.4  # Pénalité de 60%
```

**Exemple :**
- Requête : "vin fort et puissant"
- Vin : "Gamay - Léger et fruité"
- Score initial : 0.6
- Score après pénalité : **0.24**

---

#### 3.4. Pénalité Préférences Gustatives

Si l'utilisateur demande **épicé** mais le vin ne l'est pas :
```python
final_score *= 0.7  # Pénalité de 30%
```

Si l'utilisateur demande **frais** mais le vin est **corsé** :
```python
final_score *= 0.4  # Pénalité de 60%
```

---

### 4. Quels sont les bonus appliqués ?

#### 4.1. Bonus Accords Compatibles

**Pour viande rouge :**
- Si le vin mentionne SEULEMENT viande rouge → **Bonus de 15%** (× 1.15)

**Exemple :**
- Requête : "vin pour steak"
- Vin : "Bordeaux - Idéal avec bœuf et agneau"
- Score initial : 0.65
- Score après bonus : **0.75** (limité à 1.0)

---

#### 4.2. Bonus Apéro

**Priorités :**
1. Vin mentionne explicitement "apéro" → **Bonus de 50%** (× 1.5)
2. Vin mentionne accords d'apéro (fromage, charcuterie) → **Bonus de 30%** (× 1.3)
3. Vin frais ET fruité (si demandé) → **Bonus de 40%** (× 1.4)
4. Vin léger/frais/simple → **Bonus de 20%** (× 1.2)

**Exemple :**
- Requête : "vin pour apéro frais"
- Vin : "Muscadet - Frais, désaltérant, parfait pour l'apéro"
- Score initial : 0.6
- Score après bonus : **0.9** (0.6 × 1.5)

---

#### 4.3. Bonus Préférences Gustatives

**Si correspondance :**
- Épicé demandé + vin épicé → **Bonus de 20%** (× 1.2)
- Fruité demandé + vin fruité → **Bonus de 15%** (× 1.15)
- Frais demandé + vin frais → **Bonus de 30%** (× 1.3)
- Minéral demandé + vin minéral → **Bonus de 15%** (× 1.15)
- Corsé demandé + vin corsé → **Bonus de 15%** (× 1.15)

---

### 5. Comment sont combinés les bonus et pénalités ?

Les bonus et pénalités sont **multiplicatifs** et appliqués **séquentiellement**.

**Exemple complet :**
```
Score sémantique initial : 0.65

1. Budget OK → Pas de pénalité
2. Accords compatibles (viande rouge) → × 1.15 = 0.75
3. Intensité correspond (fort demandé + vin fort) → × 1.1 = 0.83
4. Préférence fruité correspond → × 1.15 = 0.95

Score final : 0.95 (limité à 1.0)
```

**Exemple avec pénalités :**
```
Score sémantique initial : 0.70

1. Budget dépassé → × 0.5 = 0.35
2. Accords incompatibles (mixte) → × 0.3 = 0.11
3. Intensité ne correspond pas → × 0.4 = 0.04

Score final : 0.04 < 0.2 → EXCLUSION
```

---

### 6. Pourquoi un seuil minimum de 0.2 ?

Le seuil de **0.2** permet d'exclure les vins qui ont accumulé trop de pénalités, même s'ils avaient un bon score sémantique initial.

**Logique :**
- Score < 0.2 → Vin **exclu** (trop inapproprié)
- Score ≥ 0.2 → Vin **conservé** (peut être proposé)

**Exemple :**
- Score sémantique : 0.75 (excellent)
- Mais budget dépassé (× 0.5) + accords incompatibles (× 0.3) = **0.11**
- Résultat : **EXCLUSION** (0.11 < 0.2)

---

### 7. Comment sont triés les résultats finaux ?

Les vins sont triés par **score final décroissant** (du meilleur au moins bon).

**Exemple de classement :**
1. Vin A : Score final = **0.92** → 1ère position
2. Vin B : Score final = **0.78** → 2ème position
3. Vin C : Score final = **0.65** → 3ème position

---

### 8. Pourquoi certains vins sont exclus même avec un bon score sémantique ?

Le système applique des **filtres stricts** avant même le calcul du score :

**Exclusions automatiques :**
- Viande rouge demandée → **Exclusion** de tous les blancs et rosés
- Poisson demandé → **Exclusion** de tous les rouges
- Apéro + préférence blanc → **Exclusion** de tous les rouges
- Vin dit explicitement "pas pour apéro" → **Exclusion**

**Exemple :**
- Requête : "vin pour entrecôte"
- Vin : "Sancerre Blanc" (score sémantique = 0.8)
- Résultat : **EXCLUSION** (blanc pour viande rouge)

---

### 9. Comment fonctionne la détection des préférences dans la description ?

Le système détecte automatiquement les préférences de type de vin dans la description utilisateur.

**Mots-clés détectés :**
- "je préfère les blancs" → Filtre **Blanc** appliqué
- "j'aime les rosés" → Filtre **Rosé** appliqué
- "blanc de préférence" → Filtre **Blanc** appliqué

**Exemple :**
- Requête : "vin pour apéro"
- Description : "je préfère les vins blanc"
- Résultat : **Tous les rouges exclus** avant le scoring

---

### 10. Comment sont gérées les négations dans les descriptions ?

Le système détecte les phrases négatives dans les descriptions de vins.

**Patterns détectés :**
- "ce n'est pas un vin d'apéro"
- "pas pour poisson"
- "ne convient pas à la viande blanche"

**Action :**
- Si négation détectée → **Pénalité de 95%** (× 0.05)

**Exemple :**
- Requête : "vin pour apéro"
- Vin : "Bordeaux - Ce n'est pas un vin d'apéro, à servir avec de la viande"
- Score initial : 0.7
- Score après pénalité négation : **0.035** (< 0.2) → **EXCLUSION**

---

## Résumé des Multiplicateurs

| Critère | Multiplicateur | Type |
|---------|---------------|------|
| Budget dépassé | × 0.5 | Pénalité |
| Accords incompatibles (mixte) | × 0.3 | Pénalité |
| Accords incompatibles (aucun) | × 0.6 | Pénalité |
| Intensité ne correspond pas | × 0.4 | Pénalité |
| Préférence ne correspond pas | × 0.7-0.8 | Pénalité |
| Négation détectée | × 0.05 | Pénalité forte |
| Accords compatibles | × 1.15 | Bonus |
| Apéro explicite | × 1.5 | Bonus fort |
| Accords d'apéro | × 1.3 | Bonus |
| Frais + fruité (apéro) | × 1.4 | Bonus |
| Préférence correspond | × 1.1-1.3 | Bonus |

---

## Exemple Complet de Calcul

**Requête :** "Je cherche un vin pour un apéro frais et fruité, je préfère les vins blanc"

**Vin analysé :** "Sancerre Blanc - Frais, citronné, fruité, parfait pour l'apéro avec fromage"

**Calcul étape par étape :**

1. **Score sémantique initial :** 0.68
   - Bonne correspondance avec "apéro frais fruité"

2. **Filtre préférence blanc :** ✅ Passé (c'est un blanc)

3. **Budget :** ✅ OK (18€ < 20€)

4. **Apéro explicite :** ✅ "parfait pour l'apéro"
   - Bonus : × 1.5 = **1.02** (limité à 1.0) → **1.0**

5. **Accords d'apéro :** ✅ "avec fromage"
   - Bonus : × 1.3 = **1.3** (limité à 1.0) → **1.0**

6. **Frais + fruité :** ✅ Les deux présents
   - Bonus : × 1.4 = **1.4** (limité à 1.0) → **1.0**

7. **Préférence frais :** ✅ Correspond
   - Bonus : × 1.3 = **1.3** (limité à 1.0) → **1.0**

**Score final : 1.0** → **Excellente recommandation** 🎯

---

## Points Clés à Retenir

1. **Le score sémantique est la base** : Il mesure la similarité textuelle
2. **Les filtres stricts excluent avant le scoring** : Type de vin, accords incompatibles
3. **Les pénalités réduisent le score** : Budget, incompatibilités, non-correspondances
4. **Les bonus augmentent le score** : Correspondances parfaites, accords explicites
5. **Le seuil de 0.2 élimine les mauvais résultats** : Même avec un bon score initial
6. **Le tri final classe par pertinence** : Du meilleur au moins bon
