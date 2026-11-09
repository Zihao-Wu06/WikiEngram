"""Prime search used to size the Engram hash buckets.

DeepSeek's reference code uses ``sympy.isprime``. Any correct primality test gives the same primes, so we
use a deterministic Miller-Rabin test and drop the sympy runtime dependency. ``tests/test_engram_parity.py`` checks
the result against sympy.
"""

from __future__ import annotations

_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin, exact for every n < 3.3e24 (far beyond any table size)."""
    if n < 2:
        return False
    for p in _WITNESSES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _WITNESSES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    """The smallest prime above ``start`` that has not been handed out yet (same rule as DeepSeek)."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate
