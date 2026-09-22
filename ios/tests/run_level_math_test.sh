#!/bin/sh
# Bullseye-level maths, checked on the Mac (no device, no simulator): LevelMath.swift is pure simd.
set -e
cd "$(dirname "$0")/.."
out=$(mktemp -d)/leveltest
swiftc -O ToolCutter/LevelMath.swift tests/level_math_test.swift -o "$out"
exec "$out"
