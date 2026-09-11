---
name: writing-tests
description: Write tests to encourage robustness and correctness of code. Use this when writing tests or evaluating whether a test should exist
---

# What is a Test Suite's Job?

Write tests that provide meaningful confidence in behavior without creating unnecessary maintenance burden. A good test suite is not the suite with the most tests. It is the suite that catches important mistakes, documents intended behavior, and remains easy to understand and change.

Every test should have a clear answer to the question "What meaningful bug would this test catch?"

If the answer is unclear, the test probably shouldn't exist. Prefer a small number of highly illuminating tests over a large number of mechanically generated or implementation-coupled tests.

Bad tests are noise that slow down development.

# What Should Tests Look Like?

A good test protects one or more of these:
- User-visible behavior
- Security or authorization boundaries
- Data integrity or memory safety
- Error handling
- Public APIs or contracts between components
- Complex logic that is difficult to reason about by inspection

Good tests fail when meaningful behavior breaks and stay passing when harmless implementation details change.

## What Does a Bad Test Look Like?

### ❌ Meaningless Tests

Do not test for facts that are effectively guaranteed by the language, framework, or trivial implementation.

Example:

```python
# production code
def add(a: float, b: float) -> float:
    return a + b

# test
def test_add_returns_three():
    assert add(1, 2) == 3

"""
This provides almost no useful confidence if `add` is literally one addition operation.
"""
```

### ❌ Testing Implementation Instead of Behavior

Tests should care about observable outcomes, not every internal behavior.

Example:

```python
# production code
class PriceService:
    def final_price(self, subtotal: float) -> float:
        discount = self._discount(subtotal)
        return subtotal - discount

    def _discount(self, subtotal: float) -> float:
        return 10 if subtotal >= 100 else 0

# test
from unittest.mock import patch

def test_final_price_calls_discount():
    service = PriceService()

    with patch.object(service, "_discount", return_value=10) as discount:
        result = service.final_price(100)

    discount.assert_called_once_with(100)
    assert result == 90

"""
This couples the test to the existence of _discount()

A better version might look like so:
"""

def test_discount_received():
    service = PriceService()
    assert service.final_price(100) == 90
```

## Guidelines for Good Tests

- Regression tests might be valuable when a real bug reveals an important case missing from the suite
- Do not commemorate every bug with a new, permanent test.
- Test along boundary cases, not random values
- Avoid testing private methods/helpers directly
