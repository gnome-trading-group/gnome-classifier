from classifier.adapters.hyperliquid import HyperliquidAdapter
from classifier.adapters.kalshi import KalshiAdapter
from classifier.adapters.polymarket_intl import PolymarketIntlAdapter

ADAPTERS = [
    PolymarketIntlAdapter(),
    KalshiAdapter(),
    HyperliquidAdapter(),
]
