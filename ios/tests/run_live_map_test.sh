#!/bin/sh
# Live height-map geometry (unprojection, floor estimate, handedness), checked on the Mac with swiftc: LiveMap.swift is pure simd.
set -e
cd "$(dirname "$0")/.."
out=$(mktemp -d)/livemaptest
cp tests/live_map_test.swift "$(dirname "$out")/main.swift"
swiftc -O ToolCutter/LiveMap.swift "$(dirname "$out")/main.swift" -o "$out"
exec "$out"
