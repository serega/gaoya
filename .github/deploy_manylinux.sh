#!/bin/bash

cd py-gaoya
rustup override set stable
export RUSTFLAGS='-C target-feature=+fxsr,+sse,+sse2,+sse3,+ssse3,+sse4.1,+popcnt'
maturin publish \
  --no-sdist \
  --username serenky
