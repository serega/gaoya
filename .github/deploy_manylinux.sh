#!/bin/bash

cd py-gaoya
rustup override set stable
export RUSTFLAGS='-C target-feature=+fxsr,+sse,+sse2,+sse3,+ssse3,+sse4.1,+popcnt'
maturin publish \
  -r https://test.pypi.org/legacy/ \
  --no-sdist \
  --username serega
