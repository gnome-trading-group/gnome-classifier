from classifier.adapters.hyperliquid import HyperliquidAdapter
from classifier.adapters.kalshi import KalshiAdapter
from classifier.adapters.polymarket_intl import PolymarketIntlAdapter
from classifier.adapters.polymarket_us import PolymarketUsAdapter

ADAPTERS = [
    PolymarketIntlAdapter(),
    PolymarketUsAdapter(),
    KalshiAdapter(),
    HyperliquidAdapter(),
]
