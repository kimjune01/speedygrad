## Round 6

1. **Math errors in Iter 7 table**
   - Location: `HYPOTHESIS_GRAPH.md` Iter 7 scorecard table.
   - The ratio `iter7/torch` for `sum_4096` is mathematically `26 / 32 = 0.8125x` (0.81x), but the table claims `0.83x`.
   - The ratio `iter7/torch` for `layernorm` is mathematically `30 / 42 = 0.714x` (0.71x), but the table claims `0.74x`.
   - These are miscalculations compared to the rest of the table which correctly calculates the ratio as `speedygrad / torch`.
