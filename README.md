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
  les colonnes `vol_1y`, `mdd_1y`, `adv_3m` et `risk_score` issues du module de
  risque pour compléter la lecture fondamentale.
