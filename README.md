# Orotitan Nexus

Ce dépôt contient un screener GARP pour les actions du CAC 40 (et, via YAML,
pour toute sélection comme l'indice SBF120) basé sur des ratios fondamentaux
(PER, dette/capitaux propres, croissance EPS, PEG, capitalisation) et enrichi
d'un module de risque marché (volatilité annualisée, drawdown sur un an et
liquidité 3 mois) qui produit un `risk_score` complémentaire, un `safety_score`
et un `nexus_score`. Une surcouche "V1.1" optionnelle ajoute des filtres
liquidité/taille d'univers et un score binaire 0-5 (`score_v1_1`) pour classer
les titres SBF120 en catégories `ELITE_V1_1`, `WATCHLIST_V1_1`, etc.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python cac40_garp_screener.py --output cac40_screen_results.csv --max_rows 40
```

- La constante `CAC40_TICKERS` dans `cac40_garp_screener.py` contient déjà les
  40 tickers Yahoo Finance (Air Liquide, Airbus, LVMH, Sanofi, etc.) ; adaptez-la
  au besoin si la composition de l'indice évolue.
- Le script télécharge les fondamentaux via `yfinance`, applique les six filtres
  stricts décrits dans le code puis calcule un score GARP pondéré (valuation,
  croissance, qualité du bilan, taille).
- Les valeurs qui passent le filtre strict sont affichées en premier, suivies
  du classement global (top N configurable via `--max_rows`).
- L'ensemble du tableau trié est exporté dans le fichier CSV fourni via
  `--output` (défaut : `cac40_screen_results.csv`). Le fichier contient aussi
  les colonnes `vol_1y`, `mdd_1y`, `adv_3m`, `risk_score`, `safety_score`,
  `nexus_score`, les drapeaux `has_*` (`has_pe_ttm`, `has_eps_cagr`, etc.),
  `has_risk_data`, l'origine de la croissance (`eps_cagr_source`) ainsi que les
  nouveaux indicateurs de complétude `data_ready_nexus` et `data_ready_v1_1`
  utilisés par la couche Nexus/V1.1. Les drapeaux OroTitan restent présents
  (`universe_ok`, `data_complete_v1_1`, `score_v1_1`, `category_v1_1`).

## Configuration optionnelle

Tous les paramètres clés (univers de tickers, seuils des filtres GARP, poids des
scores GARP et Nexus) peuvent être surchargés via un fichier YAML passé avec
`--config` :

```yaml
universe:
  tickers:
    - "MC.PA"
    - "OR.PA"
    - "AI.PA"
  min_market_cap: 1000000000.0   # filtre taille de l'univers (SBF120 par défaut)
  min_adv_3m: 100000.0           # filtre liquidité univers (ADV 3 mois)

filters:
  max_pe_ttm: 25.0
  max_forward_pe: 15.0
  max_debt_to_equity: 0.35
  min_eps_cagr: 0.08
  min_eps_cagr_v1_1: 0.15        # seuil renforcé pour le score V1.1
  max_peg: 1.2
  min_market_cap: 5000000000

weights:
  garp:
    valuation: 0.30
    growth: 0.30
    balance_sheet: 0.25
    size: 0.15
  nexus:
    quality: 0.65
    safety: 0.35
```

```bash
python cac40_garp_screener.py --config my_cac40_config.yaml --max_rows 50
```

Si le fichier est absent ou invalide, le script retombe automatiquement sur les
valeurs par défaut codées en dur.

### Univers CAC 40 complet prêt à l'emploi

Un fichier `configs/cac40_base.yaml` contient déjà les 40 tickers Yahoo Finance
du CAC 40 ainsi que les seuils et poids par défaut. Il suffit alors d'appeler :

```bash
python cac40_garp_screener.py --config configs/cac40_base.yaml --profile balanced --max_rows 40 --output cac40_full.csv
python cac40_garp_screener.py --config configs/cac40_base.yaml --profile defensive --output cac40_defensive.csv
python cac40_garp_screener.py --config configs/cac40_base.yaml --profile offensive --output cac40_offensive.csv
```

Adaptez le YAML si la composition de l'indice change ou si vous souhaitez
tester une sélection personnalisée de valeurs.

### Profil SBF120 et filtre OroTitan V1.1

Un fichier `configs/sbf120.yaml` fournit un univers élargi (SBF120) assorti de
seuils d'univers (cap ≥ 1,5 Md€, ADV ≥ 200 k titres). Après exécution, une
section supplémentaire s'affiche dans la console :

```
=== OroTitan V1.1 – SBF120 GARP filter (0–5 points) ===
```

Elle liste les titres qui réunissent à la fois les critères d'univers et des
fondamentaux complets, puis classe les scores 3/5 et 5/5 (Watchlist / Elite)
avec leurs ratios clefs (`pe_ttm`, `eps_cagr`, `peg`, `adv_3m`, etc.). Exemple :

```bash
python cac40_garp_screener.py \
  --config configs/sbf120.yaml \
  --profile balanced \
  --max_rows 120 \
  --output sbf120_balanced.csv
```

Le CSV contiendra toutes les colonnes historiques + les nouveaux drapeaux
`universe_ok`, `data_complete_v1_1`, `score_v1_1`, `category_v1_1` pour vos
analyses downstream.

## Profils Nexus prêts à l'emploi

Pour changer rapidement de posture sans modifier le YAML, un flag `--profile`
permet d'appliquer des réglages prédéfinis après chargement de la configuration
principale :

- `defensive` : filtres resserrés (PER/D/E plus stricts, croissance minimale
  relevée) et pondération Nexus axée sur la sécurité (`nexus_safety` = 45 %).
- `balanced` : aucun ajustement supplémentaire (équivalent à l'absence de
  profil).
- `offensive` : filtres assouplis (PER/D/E plus tolérants, seuil de croissance
  abaissé) et pondération Nexus qui privilégie la qualité (`nexus_quality` =
  75 %).

Exemple d'appel :

```bash
python cac40_garp_screener.py --config my_cac40_config.yaml --profile defensive
```

## Mode diagnostics détaillés

Pour comprendre précisément pourquoi un titre obtient (ou non) un bon score,
le flag `--detail` accepte un ou plusieurs tickers et affiche après le screener
un rapport texte avec :

- les fondamentaux téléchargés (avec la source de croissance EPS et les drapeaux
  de complétude `has_*`),
- le statut de chaque filtre dur avec ses seuils,
- le détail des sous-scores GARP et des pondérations en vigueur,
- les métriques de risque (volatilité, drawdown, ADV) et les scores
  `risk/safety/nexus` (calculés uniquement quand `data_ready_nexus` vaut True),
- les drapeaux de la surcouche V1.1 (éligibilité univers, readiness
  `data_ready_v1_1`, score 0-5, catégorie).

```bash
python cac40_garp_screener.py --detail MC.PA OR.PA
```

Chaque ticker absent des résultats est signalé proprement pour éviter toute
ambiguïté.
