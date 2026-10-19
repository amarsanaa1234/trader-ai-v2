"""Commodities market plug (added 2026-10-19): metals, energy and grains on OANDA.

Same OANDA connection, data and broker as FX (markets/fx), but its own symbols,
evidence, risk settings, database and bot. Interest-rate carry/changes are left
out (a barrel of oil has no central bank); the dollar, bond yields, big funds'
positioning (COT), stocks, trend and sessions stay.
"""
import pandas as pd

from markets.fx.adapter import FX

KEEP = ("crowd", "usd", "tsmom", "cot", "xasset", "session", "orders", "yields", "month_end")


class Commodities(FX):
    name = "commodities"
    label = "🥇 Түүхий эд"
    source_names = KEEP
    default_params = {k: v for k, v in FX.default_params.items()
                      if not k.startswith("w_") or k[2:] in KEEP or k == "w_news"}
    param_space = {k: v for k, v in FX.param_space.items() if not k.startswith("w_") or k[2:] in KEEP}
    rule_columns = ("long_pct", "usd_mom", "cot_net", "xasset", "session")
    rule_help = ("long_pct (% of OANDA clients long, 0-100; gold only), "
                 "usd_mom (1-day US-dollar basket log-move, + = dollar stronger, e.g. 0.003), "
                 "cot_net (speculators' net futures position, share of open interest, -0.5..0.5), "
                 "xasset (-1..1 pull from stocks), session (-1..1 London/New York range breakout)")

    def sources(self, df: pd.DataFrame, p: dict) -> dict:
        allp = super().sources(df, p)
        return {k: allp[k] for k in KEEP}


MARKET = Commodities()
