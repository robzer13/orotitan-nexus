# Orotitan Nexus

Ce dépôt contient un MVP de screener GARP pour les actions du CAC 40.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
python cac40_garp_screener.py --top_n 15 --min_score 60
```

- Les tickers du CAC 40 sont définis dans la constante `CAC40_TICKERS` à
  compléter dans `cac40_garp_screener.py`.
- Les résultats complets sont exportés dans le fichier CSV passé en argument
  (`--output`).
