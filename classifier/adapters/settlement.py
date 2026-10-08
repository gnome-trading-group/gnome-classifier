from decimal import Decimal, InvalidOperation
from typing import Iterator

# Settlement values are prices in the registry's fixed point: 1e9 = $1, the most an outcome can pay.
PRICE_SCALE = 1_000_000_000


def to_price(value) -> int | None:
    """A venue's decimal settlement value (dollars, 0 to 1) as an exact price, or None if it isn't one.

    Decimal rather than float, so "0.47" becomes exactly 470000000 instead of whatever the float rounds to.
    """
    try:
        scaled = Decimal(str(value)) * PRICE_SCALE
    except (InvalidOperation, ValueError):
        return None
    if not scaled.is_finite() or scaled != scaled.to_integral_value() or not 0 <= scaled <= PRICE_SCALE:
        return None
    return int(scaled)


def chunked(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
