# Orotitan Nexus

Ce dépôt contient un screener GARP pour les actions du CAC 40 (et, via YAML,
pour toute sélection comme l'indice SBF120) basé sur des ratios fondamentaux
(PER, dette/capitaux propres, croissance EPS, PEG, capitalisation) et enrichi
d'un module de risque marché (volatilité annualisée, drawdown sur un an et
liquidité 3 mois) qui produit un `risk_score` complémentaire, un `safety_score`
et un `nexus_score`. Une surcouche "V1.1" optionnelle ajoute des filtres
liquidité/taille d'univers et un score binaire 0-5 (`score_v1_1`) pour classer
les titres SBF120 en catégories `ELITE_V1_1`, `WATCHLIST_V1_1`, etc.

## OroTitan Nexus Screener v1.5 – Production Gate

La version 1.5 consolide l'API Python officielle (`orotitan_nexus.api`) et un
orchestrateur multi-univers. Outre les CLIs historiques, vous pouvez désormais
script-er le moteur comme suit :

```python
from orotitan_nexus.api import run_screen, run_cac40_garp, run_custom_garp

# Exécution V1.1 générique
df, summary = run_screen(config_path="configs/sbf120.yaml", profile_name="balanced")
print(summary["total"], "titres analysés")

# Radar CAC40 GARP strict 5/5
full_df, radar_df, garp_summary = run_cac40_garp(config_path="configs/cac40.yaml")
print("Strict passes:", garp_summary["strict_count"])

# Radar strict GARP sur un univers custom
full_custom, radar_custom, custom_summary = run_custom_garp(
    config_path="configs/sbf120.yaml",
    profile_name="balanced",
    tickers=["MC.PA", "OR.PA", "SAN.PA"],
)
print(custom_summary["strict_count"], "titres 5/5 sur l'univers custom")
```

Les CLIs s’appuient sur le même orchestrateur, garantissant un comportement
identique quel que soit le mode d’appel.

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

### OroTitan CAC40 GARP Radar (v1.6)

Pour un focus "pur GARP" sur le CAC 40, une CLI dédiée applique strictement
les 5 règles suivantes (paramétrables via `profile.garp`) :

1. PER (ttm) < 25
2. PER anticipé < 15
3. Dette / Capitaux propres < 35 %
4. EPS CAGR > 15 %
5. PEG < 1,2

Elle produit deux CSV :

- `--output-full` (toutes les valeurs CAC40 + colonnes `strict_pass_garp`,
  `strict_garp_score` et `strict_garp_bucket`).
- `--output-radar` (uniquement les titres strict 5/5, triés par
  `strict_garp_score`, `score_v1_1` puis market cap).

```bash
python -m orotitan_nexus.cli_cac40_garp \
  --config configs/cac40.yaml \
  --profile balanced \
  --output-full /tmp/cac40_full_garp.csv \
  --output-radar /tmp/cac40_radar_garp.csv

Avec un portefeuille PEA (CSV `ticker,quantity,cost_basis`) pour marquer les lignes déjà détenues :

python -m orotitan_nexus.cli_cac40_garp \
  --config configs/cac40.yaml \
  --profile balanced \
  --portfolio offline/portfolios/pea_2025-11-17.csv \
  --output-full data/cac40_full_with_pea.csv \
  --output-radar data/cac40_radar_with_pea.csv
```

La console (et le logger) affichent désormais un résumé enrichi : univers total,
valeurs data-complete, compte par catégorie V1.1, distribution des buckets
GARP (`ELITE`, `STRONG`, `BORDERLINE`, `REJECT`) et top strict-pass trié par
`strict_garp_score`. Le module repose sur `apply_garp_rules` +
`compute_garp_score` + `assign_garp_buckets` et s'aligne automatiquement sur les
seuils définis dans `profile.garp`.

### OroTitan Custom GARP Radar (v1.6)

Le radar strict 5/5 peut désormais être appliqué à **n'importe quel univers** en
fournissant vos propres tickers :

```bash
python -m orotitan_nexus.cli_custom_garp \
  --config configs/sbf120.yaml \
  --profile balanced \
  --tickers "MC.PA,OR.PA,SAN.PA" \
  --output-full /tmp/custom_full_garp.csv \
  --output-radar /tmp/custom_radar_garp.csv
```

ou via un fichier CSV listant les tickers :

```bash
python -m orotitan_nexus.cli_custom_garp \
  --config configs/sbf120.yaml \
  --profile balanced \
  --tickers-csv ./mes_tickers.csv \
  --tickers-column ticker \
  --output-full /tmp/custom_full_garp.csv \
  --output-radar /tmp/custom_radar_garp.csv
```

La CLI nettoie les doublons, applique `data_complete_v1_1 == True` avant
`apply_garp_rules`, calcule le score continu `strict_garp_score`, assigne un
`strict_garp_bucket`, exporte un CSV complet et un CSV "radar" (optionnel) puis
affiche un résumé : univers, profil, compte data-complete, strict-pass, buckets
et top 5 trié par `strict_garp_score` / `score_v1_1` / market cap. La même
logique est exposée via `run_custom_garp` côté API.

```python
from orotitan_nexus.api import run_custom_garp

full_df, radar_df, summary = run_custom_garp(
    config_path="configs/sbf120.yaml",
    profile_name="balanced",
    tickers=["MC.PA", "OR.PA", "SAN.PA"],
)

print(summary["bucket_counts"])
print(radar_df[["ticker", "strict_pass_garp", "strict_garp_score", "strict_garp_bucket"]])
```

### GARP History & Drift (v1.7)

La v1.7 ajoute un module d'historisation léger (`history.py`) capable :

1. d'appendre chaque run dans un CSV (`--history-path`, paramètres API
   `history_path`/`run_id`/`notes`) avec le total, `strict_garp_count` et la
   distribution des buckets `ELITE/STRONG/BORDERLINE/REJECT` ;
2. d'écrire un snapshot par run (`--snapshots-dir`) sous la forme
   `garp_<run_id>.csv`.

En combinant les snapshots avec `--compare-with-run-id`, la CLI génère un
rapport de drift : nouveaux strict-pass, sorties, upgrades/downgrades de bucket.

```bash
python -m orotitan_nexus.cli_cac40_garp \
  --config configs/cac40.yaml \
  --profile balanced \
  --history-path data/garp_history.csv \
  --snapshots-dir data/garp_snapshots \
  --run-id "2025-11-15-open" \
  --notes "Pré-market earnings"

python -m orotitan_nexus.cli_cac40_garp \
  --config configs/cac40.yaml \
  --profile balanced \
  --history-path data/garp_history.csv \
  --snapshots-dir data/garp_snapshots \
  --run-id "2025-11-18-open" \
  --compare-with-run-id "2025-11-15-open"
```

La même fonctionnalité est disponible pour la CLI custom ainsi que via l'API :

```python
from orotitan_nexus.api import run_custom_garp

full_df, radar_df, summary = run_custom_garp(
    config_path="configs/sbf120.yaml",
    profile_name="balanced",
    tickers=["MC.PA", "OR.PA", "SAN.PA"],
    history_path="data/garp_history.csv",
    run_id="2025-11-18-open",
    notes="Radar custom luxe",
)
print(summary["bucket_counts"])
```

## Architecture v1.4

Le script monolithique historique a été refactoré en package Python `orotitan_nexus`
avec des modules spécialisés :

- `api.py` : fonctions publiques `run_screen` et `run_cac40_garp` pour des
  intégrations programmatiques.
- `orchestrator.py` : coeur d'orchestration multi-univers (récupération des
  données, normalisation, filtres, scores, application optionnelle des règles
  GARP).
- `config.py` : chargement + validation YAML, profils Nexus (`--profile`) et dataclasses
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
- `garp_rules.py` : fonctions pures `apply_garp_rules`, `compute_garp_score` et
  `assign_garp_buckets` qui appliquent les 5 règles GARP (PER ttm/fwd, D/E, EPS
  CAGR, PEG), produisent `strict_pass_garp`, un score continu 0‑100
  (`strict_garp_score`) ainsi qu'un bucket lisible (`strict_garp_bucket`).
- `reporting.py` : affichages console (aperçu strict/global, overlay V1.1,
  diagnostics `--detail`, résumé `--summary`, écriture CSV testable) et
  fonctions `summarize_v1` / `summarize_garp` utilisées par les CLIs et l’API.
- `history.py` : dataclass `GarpRunRecord`, append CSV des runs,
  snapshots `garp_<run_id>.csv` et helper `compute_garp_diff` pour les rapports
  de drift.
- `cli.py` : parsing des flags (inchangés + `--summary`), orchestration complète,
  export CSV et diagnostics.
- `cli_cac40_garp.py` : CLI "radar" dédiée CAC40 qui applique les 5 règles
  GARP et génère un CSV complet ainsi qu'un CSV resserré sur les titres 5/5.

Le fichier racine `cac40_garp_screener.py` sert désormais de simple wrapper qui
appelle `orotitan_nexus.cli.main()` afin de conserver la compatibilité des
commandes existantes.

## Tests & QA

Les tests unitaires s'exécutent depuis la racine du dépôt :

```bash
python -m compileall cac40_garp_screener.py orotitan_nexus
pytest
python -m orotitan_nexus.cli_cac40_garp --config configs/cac40.yaml --profile balanced --output-full /tmp/cac40_full_garp.csv --output-radar /tmp/cac40_radar_garp.csv
python -m orotitan_nexus.cli_custom_garp --config configs/sbf120.yaml --profile balanced --tickers "MC.PA,OR.PA,SAN.PA" --output-full /tmp/custom_full_garp.csv --output-radar /tmp/custom_radar_garp.csv
```

Les fixtures synthétisent des `income_stmt` multi-annuels pour valider le calcul
du CAGR (Net Income) ainsi que les drapeaux `has_eps_cagr` / `data_complete_v1_1`.
Des tests dédiés (`tests/test_history.py`, `tests/test_garp_rules.py`,
`tests/test_cli_cac40_garp.py`, `tests/test_cli_custom_garp.py`,
`tests/test_api.py`, `tests/test_orchestrator.py`, etc.) verrouillent désormais
les 5 règles strictes **et** la nouvelle couche `strict_garp_score` /
`strict_garp_bucket`, l'historique (`append_history_record`, snapshots,
drift). Toute modification des seuils ou de la logique doit maintenir cette
suite au vert.

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

## Portfolio overlay & radar actionnable (v1.8)

Le moteur peut optionnellement croiser un portefeuille local (CSV avec au moins
`ticker`, `quantity`, `cost_basis`) avec le radar strict GARP pour distinguer :

- les titres déjà en portefeuille qui passent les 5 règles (candidats à
  renforcer ou à suivre),
- les titres hors portefeuille qui passent les 5 règles (nouvelles idées).

Des métriques simples d'évaluation (`owned`, `position_quantity`,
`position_cost_basis`, `position_value`, `unrealized_pl`) sont ajoutées dans le
CSV complet, et une option `--budget` propose une liste de nouvelles idées sous
une enveloppe donnée (1 action par ticker pour l'illustration).

Exemple CAC40 :

```bash
python -m orotitan_nexus.cli_cac40_garp \
  --config configs/cac40.yaml \
  --profile balanced \
  --portfolio data/portfolio_pea.csv \
  --budget 500.0 \
  --output-full /tmp/cac40_full_garp.csv \
  --output-radar /tmp/cac40_radar_garp.csv
```

## Offline GARP backtest (v1.9)

Vous pouvez comparer hors-ligne la performance d'un portefeuille strict GARP
équipondéré vs un benchmark équipondéré (toutes les valeurs `data_complete_v1_1`
) à partir :

- d'un snapshot CSV (export complet d'un run GARP) contenant au minimum
  `ticker`, `strict_pass_garp`, `data_complete_v1_1`,
- d'un CSV de prix longs avec `date`, `ticker`, `adj_close`.

API Python :

```python
from orotitan_nexus.api import run_garp_backtest_offline

df = run_garp_backtest_offline(
    snapshot_path="history/cac40_snapshot.csv",
    prices_path="data/cac40_prices.csv",
    start_date="2025-01-02",
    horizons=(21, 63, 252),
)
print(df)
```

CLI :

```bash
python -m orotitan_nexus.cli_garp_backtest \
  --snapshot history/cac40_snapshot.csv \
  --prices data/cac40_prices.csv \
  --start-date 2025-01-02 \
  --horizons 21,63,252
```

Tout est offline : les CSV de snapshot et de prix sont fournis par l'utilisateur
et aucun téléchargement n'est effectué. Le rapport console affiche, pour chaque
horizon, les rendements GARP, benchmark et l'excès de performance.

## GARP Rule Diagnostics & Calibration (v2.0)

Le module de diagnostics règle par règle permet d'évaluer, hors-ligne, la
performance future des titres qui passent ou échouent chaque critère GARP (et
le strict 5/5) sur plusieurs horizons :

- Comparaison pass vs fail pour chaque règle (PER ttm, PER fwd, D/E, EPS CAGR,
  PEG, strict 5/5) ;
- Calculs offline à partir d'un snapshot CSV (avec `garp_*_ok`,
  `strict_pass_garp`, `data_complete_v1_1`) et d'un CSV de prix longs (`date`,
  `ticker`, `adj_close`).

API Python :

```python
from orotitan_nexus.api import run_garp_diagnostics_offline

diag = run_garp_diagnostics_offline(
    snapshot_path="history/cac40_garp_snapshot_2025-01-02.csv",
    prices_path="data/cac40_prices_2024-2025.csv",
    start_date="2025-01-02",
    horizons=[21, 63, 252],
)
print(diag)
```

CLI :

```bash
python -m orotitan_nexus.cli_garp_diagnostics \
  --snapshot history/cac40_garp_snapshot_2025-01-02.csv \
  --prices data/cac40_prices_2024-2025.csv \
  --start-date 2025-01-02 \
  --horizons 21,63,252
```

Tout reste offline et déterministe. Ces diagnostics servent à valider ou
recalibrer les seuils GARP (25/15, 35 %, 15 %, PEG 1.2) et à suivre l'impact de
chaque règle ; la fonctionnalité est couverte par des tests `pytest` et doit le
rester.

## Nexus GARP v2 (multi-factor)

v2 ajoute, de manière optionnelle et entièrement pilotée par la config YAML, une couche multi-facteur sur le noyau strict GARP. Lorsque `profile.v2.enabled` est à `true`, le dataframe inclut :

- `garp_score_v2` (dérivé des surfaces strictes),
- `quality_score`, `momentum_score`, `risk_score_v2`, `macro_score`, `behavioral_score`,
- un score global `nexus_v2_score` (0–100) et `nexus_v2_bucket` (V2_ELITE / V2_STRONG / V2_NEUTRAL / V2_WEAK / V2_REJECT).

Les poids et seuils se règlent dans `profile.v2` (et les sous-sections `quality`, `momentum`, `risk`, `macro`, `behavioral`). Par défaut v2 est désactivé : le comportement v1.x reste identique tant que vous ne l’activez pas.

### API rapide

```python
from orotitan_nexus.api import run_screen

df, summary = run_screen(config_path="configs/cac40.yaml", profile_name="balanced")
# si profile.v2.enabled=True dans la YAML, df contient nexus_v2_score / nexus_v2_bucket
```

### Backtest offline par score (quintiles)

```bash
python -m orotitan_nexus.cli_garp_backtest \
  --snapshot history/cac40_snapshot.csv \
  --prices data/cac40_prices.csv \
  --start-date 2025-01-02 \
  --horizons 21,63,252 \
  --score-column nexus_v2_score
```

Les CLIs CAC40/custom continuent d’exposer les champs v1 ; si v2 est activé, les nouveaux champs apparaissent simplement dans les CSV complets/radar.

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

### Nexus GARP v2.2 – Robustesse et Explicabilité

La couche v2 reste 100 % optionnelle. Vous pouvez activer des contrôles supplémentaires via le YAML :

```yaml
profile:
  v2:
    enabled: true
  walkforward:
    enabled: true
    n_splits: 4
  sensitivity:
    enabled: true
    weight_perturbation_pct: 0.1
  regime:
    enabled: true
    benchmark_ticker: "^FCHI"
```

Outils disponibles :

- **CLI robustesse** : `python -m orotitan_nexus.cli_v2_robustness --config ... --snapshot ... --prices ...` produit un résumé walk-forward / sensibilité / régimes (option `--output-json`).
- **Explication v2** : `python -m orotitan_nexus.cli_custom_garp ... --explain MC.PA` affiche une décomposition des contributions par pilier pour le ticker demandé (si le profil v2 est activé).

Les nouvelles fonctionnalités sont purement additives et n’affectent pas les sorties v1.x. Toute modification future doit laisser la suite `pytest` au vert.

### Nexus Core v6.8-lite (v2.3)

Cette couche reste **désactivée par défaut** pour préserver les workflows v1/v2. Lorsqu’elle est activée (section `profile.nexus_core`), le screener ajoute :

- Piliers 0–100 : Quality (Q), Value (V), Momentum (M), Risk inverse (R), Behavior (B), Fit (F).
- Score global `nexus_core_score` (0–100) avec poids configurables (défaut : 30/20/20/15/10/5).
- Score `nexus_exceptionality` (0–10) combinant Q, Fit, Nexus global et indicateurs optionnels (Diamond/Gold/Red flag).
- Colonnes ajoutées : `nexus_core_q/v/m/r/b/f`, `nexus_core_score`, `nexus_exceptionality` (aucune colonne existante n’est modifiée).

Configuration type :

```yaml
profile:
  nexus_core:
    enabled: true
    weight_q: 0.30
    weight_v: 0.20
    weight_m: 0.20
    weight_r: 0.15
    weight_b: 0.10
    weight_f: 0.05
    fit_priority_sectors: ["Health", "Technology"]
    fit_pea_bonus: 30
    fit_liquidity_threshold: 1000000
```

Utilisation rapide (API) :

```python
from orotitan_nexus.api import run_custom_garp

df, radar, summary = run_custom_garp(
    config_path="configs/sbf120.yaml",
    profile_name="balanced",  # profil où nexus_core.enabled=True
    tickers=["MC.PA", "OR.PA", "SAN.PA"],
)
print(df[["ticker", "nexus_core_score", "nexus_exceptionality"]].head())
```

Les CLIs existantes continuent de fonctionner sans changement ; si la section `nexus_core` est activée, les CSV complets contiendront automatiquement les colonnes Nexus Core.

### Nexus Valuation & Execution (v6.8-lite adjunct)

- **Valuation (opt-in)** : prix cible composite P/FCF + EV/EBIT + PER anticipé, avec cap d’upside et repli vers un consensus si les fondamentaux manquent. Activez via `profile.valuation.enabled: true`.
- **Scénarios Bear/Base/Bull** : helper générique (`Scenario`) pour calculer un prix pondéré par scénario (métrique × multiple – net debt) / actions.
- **Exécution** : primitives d’entrée (breakout/pullback), stops ATR sectoriels, et sizing respectant budget/risque et un plafond 2 % ADV.
- **ETF Nexus score** : score coût/track/liquidité/diversification/fit pour les lignes ETF (si `profile.etf_scoring.enabled`).
- **Risque portefeuille** : métriques HHI et Top5 pour surveiller la concentration (helpers disponibles dans `portfolio_metrics`).