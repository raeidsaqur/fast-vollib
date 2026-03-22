# Compatibility

`fastiv` targets the public `py_vollib_vectorized` surface:

- pricing entrypoints
- implied volatility entrypoints
- greek entrypoints
- dataframe helper
- `py_vollib` monkeypatch support

Compatibility defaults:

- upstream argument order is preserved
- `return_as="dataframe"` and `"series"` behave as pandas outputs
- non-pandas returns default to NumPy-compatible arrays
- backend-native tensors/arrays require `return_native=True`
