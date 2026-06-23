from classifier.adapters.hyperliquid import HyperliquidAdapter
from classifier.adapters.kalshi import KalshiAdapter
from classifier.adapters.polymarket import PolymarketAdapter

ADAPTERS = [
    PolymarketAdapter(),
    KalshiAdapter(),
    HyperliquidAdapter(),
]
