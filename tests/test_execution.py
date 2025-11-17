from orotitan_nexus.config import EntrySettings, StopSettings, SizingSettings
from orotitan_nexus.execution import (
    compute_breakout_entry,
    compute_initial_stop,
    compute_position_size,
    compute_pullback_entry,
)


def test_breakout_and_pullback_entries():
    settings = EntrySettings()
    row = {"price": 100.0, "PH20": 102.0, "ATR14": 2.0, "Volume": 130.0, "ADV20_shares": 100.0}
    assert compute_breakout_entry(row, settings) == 100.5

    row_low_vol = {"price": 100.0, "PH20": 102.0, "ATR14": 2.0, "Volume": 50.0, "ADV20_shares": 100.0}
    assert compute_breakout_entry(row_low_vol, settings) is None

    pullback = compute_pullback_entry({"AVWAP_event_price": 110.0, "ATR14": 4.0}, settings)
    assert pullback == 109.0


def test_initial_stop_and_sizing():
    stop_settings = StopSettings()
    stop_price = compute_initial_stop({"price": 100.0, "ATR14": 2.0}, stop_settings, sector="Technology", is_etf=False)
    assert stop_price == 96.0

    etf_stop = compute_initial_stop({"price": 100.0, "ATR14": 2.0}, stop_settings, sector="", is_etf=True)
    assert 88.0 < etf_stop < 91.0

    sizing_settings = SizingSettings()
    qty = compute_position_size(
        entry=100.0,
        stop=95.0,
        budget_eur=1000.0,
        risk_eur=50.0,
        adv20_eur=10_000.0,
        sizing=sizing_settings,
    )
    assert qty == 2  # capped by ADV constraint

