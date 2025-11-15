# Orotitan Nexus

Ce dépôt contient un screener GARP pour les actions du CAC 40 (et, via YAML,
pour toute sélection comme l'indice SBF120) basé sur des ratios fondamentaux
(PER, dette/capitaux propres, croissance EPS, PEG, capitalisation) et enrichi
d'un module de risque marché (volatilité annualisée, drawdown sur un an et
liquidité 3 mois) qui produit un `risk_score` complémentaire, un `safety_score`
et un `nexus_score`. Une surcouche "V1.1" optionnelle ajoute des filtres
liquidité/taille d'univers et un score binaire 0-5 (`score_v1_1`) pour classer
les titres SBF120 en catégories `ELITE_V1_1`, `WATCHLIST_V1_1`, etc.

## Installation locale

```bash
pip install -e .
# pour inclure les dépendances de test :
pip install -e .[dev]
```

L'installation editable permet d'importer `orotitan_nexus` partout (tests, notebooks,
pipelines) sans ajuster `PYTHONPATH` manuellement.

## Utilisation

```bash
python cac40_garp_screener.py --output cac40_screen_results.csv --max_rows 40 --summary
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
  `--output` (défaut : `cac40_screen_results.csv`) ; le flag `--summary` ajoute
  un récapitulatif console (univers total, data-complete, catégories V1.1, top 5).
  Le fichier contient aussi
  les colonnes `vol_1y`, `mdd_1y`, `adv_3m`, `risk_score`, `safety_score`,
  `nexus_score`, les drapeaux `has_*` (`has_pe_ttm`, `has_eps_cagr`, etc.),
  `has_risk_data`, l'origine de la croissance (`eps_cagr_source`) ainsi que les
  nouveaux indicateurs de complétude `data_ready_nexus` et `data_ready_v1_1`
  utilisés par la couche Nexus/V1.1. Les drapeaux OroTitan restent présents
  (`universe_ok`, `data_complete_v1_1`, `score_v1_1`, `category_v1_1`).

### OroTitan CAC40 GARP Radar (v1.3)

Pour un focus "pur GARP" sur le CAC 40, une CLI dédiée applique strictement
les 5 règles suivantes (paramétrables via `profile.garp`) :

1. PER (ttm) < 25
2. PER anticipé < 15
3. Dette / Capitaux propres < 35 %
4. EPS CAGR > 15 %
5. PEG < 1,2

Elle produit deux CSV :

- `--output-full` (toutes les valeurs CAC40 + colonne booléenne `strict_pass_garp`)
- `--output-radar` (uniquement les titres qui respectent les 5 règles, triés par
  market cap)

```bash
python -m orotitan_nexus.cli_cac40_garp \
  --config configs/cac40.yaml \
  --profile balanced \
  --output-full /tmp/cac40_full_garp.csv \
  --output-radar /tmp/cac40_radar_garp.csv
```

La console affiche un résumé synthétique (univers total, valeurs data-complete,
compte par catégorie V1.1, top 5 des strict-pass GARP). Le module repose sur
`apply_garp_rules` et s'aligne automatiquement sur les seuils définis dans la
section `profile.garp` de vos YAML.

## Architecture v1.3

Le script monolithique historique a été refactoré en package Python `orotitan_nexus`
avec des modules spécialisés :

- `config.py` : chargement YAML, profils Nexus (`--profile`) et dataclasses
  `FilterSettings`, `WeightSettings`, `UniverseSettings`, **`GarpThresholds`**
  (section `profile.garp` dans les YAML).
- `universe.py` : gestion des listes de tickers (CAC 40 par défaut, SBF120 via
  YAML) et filtres d'univers (cap/ADV).
- `data_fetch.py` : téléchargement `yfinance` encapsulé (fondamentaux + prix).
- `normalization.py` : conversions agressives (`safe_float`, recalcul PER/PEG,
  reconstitution D/E, **CAGR EPS calculé depuis le Net Income des états de
  résultat `income_stmt`** avec la colonne `eps_cagr_source`, métriques de
  risque + drapeaux `has_*`, `data_ready_*`).
- `filters.py` : fonctions pures pour les filtres durs, l'éligibilité
  d'univers (avec `universe_exclusion_reason`) et les catégories V1.1.
- `scoring.py` : sous-scores GARP, score GARP (0-100), module de risque,
  `safety_score`, `nexus_score`, score binaire V1.1 (0-5) avec point bonus PEG.
- `garp_rules.py` : fonction pure `apply_garp_rules` qui applique les 5 règles
  GARP (PER ttm/fwd, D/E, EPS CAGR, PEG) et ajoute `strict_pass_garp` sans
  impacter les filtres historiques.
- `reporting.py` : affichages console (aperçu strict/global, overlay V1.1,
  diagnostics `--detail`, résumé `--summary`, écriture CSV testable).
- `cli.py` : parsing des flags (inchangés + `--summary`), orchestration complète,
  export CSV et diagnostics.
- `cli_cac40_garp.py` : CLI "radar" dédiée CAC40 qui applique les 5 règles
  GARP et génère un CSV complet ainsi qu'un CSV resserré sur les titres 5/5.

Le fichier racine `cac40_garp_screener.py` sert désormais de simple wrapper qui
appelle `orotitan_nexus.cli.main()` afin de conserver la compatibilité des
commandes existantes.

## Tests

Les tests unitaires s'exécutent depuis la racine du dépôt :

```bash
pytest
```

Les fixtures synthétisent des `income_stmt` multi-annuels pour valider le calcul
du CAGR (Net Income) ainsi que les drapeaux `has_eps_cagr` / `data_complete_v1_1`.
Des tests dédiés (`tests/test_garp_rules.py`, `tests/test_cli_cac40_garp.py`)
verrouillent les règles GARP (5/5) et la CLI radar afin d'éviter toute
régression lors de futures évolutions.

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

profile:
  name: balanced
  garp:
    pe_ttm_max: 25.0
    pe_fwd_max: 15.0
    debt_to_equity_max: 0.35
    eps_cagr_min: 0.15
    peg_max: 1.2
```

```bash
python cac40_garp_screener.py --config my_cac40_config.yaml --max_rows 50
```

Si le fichier est absent ou invalide, le script retombe automatiquement sur les
valeurs par défaut codées en dur.

### Univers CAC 40 complet prêt à l'emploi

Les fichiers `configs/cac40.yaml` (profil standard) et `configs/cac40_base.yaml`
contiennent déjà les 40 tickers Yahoo Finance du CAC 40 ainsi que les seuils et
poids par défaut. Il suffit alors d'appeler :

```bash
python cac40_garp_screener.py --config configs/cac40.yaml --profile balanced --max_rows 40 --output cac40_full.csv
python cac40_garp_screener.py --config configs/cac40.yaml --profile defensive --output cac40_defensive.csv
python cac40_garp_screener.py --config configs/cac40.yaml --profile offensive --output cac40_offensive.csv
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
`universe_ok`, `universe_exclusion_reason`, `data_complete_v1_1`, `score_v1_1`,
`category_v1_1` pour vos analyses downstream.

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
- les drapeaux de la surcouche V1.1 (éligibilité univers avec raison,
  readiness `data_ready_v1_1`, score 0-5, catégorie).

```bash
python cac40_garp_screener.py --detail MC.PA OR.PA
```

Chaque ticker absent des résultats est signalé proprement pour éviter toute
ambiguïté.

## Philosophie data-quality

- Les colonnes `has_*` et `data_ready_*` indiquent explicitement quelles métriques
  sont exploitables. Si `data_ready_nexus` est `False`, le score GARP, le module
  de risque, le `safety_score` et le `nexus_score` sont forcés à 0/NaN afin de
  ne pas classer de titres sur des données partielles.
- `data_complete_v1_1` nécessite désormais **les fondamentaux clés + la taille
  et les métriques de risque** : PER ttm/fwd, D/E, CAGR EPS issu des états
  financiers, capitalisation et volatilité/MDD/ADV exploitables. Le PEG reste un
  bonus pour obtenir le 5ᵉ point mais son absence ne bascule plus en
  `DATA_MISSING`.
- `category_v1_1` passe à `DATA_MISSING` dès que ces inputs requis manquent, ou
  à `EXCLUDED_UNIVERSE` si la capitalisation/ADV n'atteint pas les seuils (la
  colonne `universe_exclusion_reason` explicite alors le motif).

## Tests automatisés

Une suite `pytest` compacte garantit la reproductibilité :

```bash
pytest
```

Les tests couvrent les fonctions critiques (`normalization`, `scoring`) et un
scénario end-to-end simulé (fetch `yfinance` monkeypatché). Avant livraison,
vous pouvez également vérifier que le code se compile :

```bash
python -m compileall cac40_garp_screener.py orotitan_nexus
```
