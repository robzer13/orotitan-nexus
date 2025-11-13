# Orotitan Nexus

Ce dépôt contient un screener GARP pour les actions du CAC 40 basé sur des
ratios fondamentaux (PER, dette/capitaux propres, croissance EPS, PEG et taille)
et enrichi d'un module de risque marché (volatilité annualisée, drawdown sur un
an et liquidité 3 mois) qui produit un `risk_score` complémentaire.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python cac40_garp_screener.py --output cac40_screen_results.csv --max_rows 40
```

- Compléter la constante `CAC40_TICKERS` dans `cac40_garp_screener.py` avec les
  tickers Yahoo Finance du CAC 40.
- Le script télécharge les fondamentaux via `yfinance`, applique les six filtres
  stricts décrits dans le code puis calcule un score GARP pondéré (valuation,
  croissance, qualité du bilan, taille).
- Les valeurs qui passent le filtre strict sont affichées en premier, suivies
  du classement global (top N configurable via `--max_rows`).
- L'ensemble du tableau trié est exporté dans le fichier CSV fourni via
  `--output` (défaut : `cac40_screen_results.csv`). Le fichier contient aussi
  les colonnes `vol_1y`, `mdd_1y`, `adv_3m`, `risk_score`, `safety_score` et
  `nexus_score` pour compléter la lecture fondamentale/risk.

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

filters:
  max_pe_ttm: 25.0
  max_forward_pe: 15.0
  max_debt_to_equity: 0.35
  min_eps_cagr: 0.08
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
