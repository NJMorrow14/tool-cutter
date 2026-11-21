#!/bin/sh
# On-device outlining maths (OutlineEngine.swift), checked on the Mac with swiftc: Foundation/CoreGraphics/simd only.
set -e
cd "$(dirname "$0")/.."
out=$(mktemp -d)/outlinetest
cp tests/outline_engine_test.swift "$(dirname "$out")/main.swift"
swiftc -O ToolCutter/OutlineEngine.swift "$(dirname "$out")/main.swift" -o "$out"
exec "$out"
