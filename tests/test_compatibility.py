import pandas as pd

from orotitan_nexus.compatibility import compute_line_compatibility_score
from orotitan_nexus.config import ProfileSettingsV2


def test_compatibility_rewards_diversification():
    profile = ProfileSettingsV2()
    base_portfolio = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "position_value": [100, 100],
            "primary_sector": ["Tech", "Tech"],
            "country_of_listing": ["FR", "FR"],
        }
    )
    divers_candidate = pd.Series(
        {"ticker": "CCC", "primary_sector": "Health", "country_of_listing": "US", "position_value": 50}
    )
    concentrated_candidate = pd.Series(
        {"ticker": "DDD", "primary_sector": "Tech", "country_of_listing": "FR", "position_value": 50}
    )
    divers_score = compute_line_compatibility_score(divers_candidate, base_portfolio, profile, line_budget_value=50)
    conc_score = compute_line_compatibility_score(concentrated_candidate, base_portfolio, profile, line_budget_value=50)
    assert divers_score > conc_score


def test_compatibility_pea_bonus():
    profile = ProfileSettingsV2()
    base_portfolio = pd.DataFrame({"ticker": ["AAA"], "position_value": [100], "primary_sector": ["Tech"], "country_of_listing": ["FR"]})
    row_no_pea = pd.Series({"ticker": "BBB", "primary_sector": "Tech", "country_of_listing": "US", "PEA_Eligible_bool": False})
    row_pea = pd.Series({"ticker": "CCC", "primary_sector": "Tech", "country_of_listing": "FR", "PEA_Eligible_bool": True})
    score_no = compute_line_compatibility_score(row_no_pea, base_portfolio, profile, line_budget_value=50)
    score_yes = compute_line_compatibility_score(row_pea, base_portfolio, profile, line_budget_value=50)
    assert score_yes > score_no
