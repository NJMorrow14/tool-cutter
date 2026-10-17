#!/bin/sh
# The finger retrace (OutlineCorrection.swift) on the Mac with a fake SAM: Foundation/CoreGraphics (+ SAM2Engine.swift
# for its Prompt type; no models are loaded).
set -e
cd "$(dirname "$0")/.."
out=$(mktemp -d)/correctiontest
cp tests/outline_correction_test.swift "$(dirname "$out")/main.swift"
swiftc -O ToolCutter/OutlineEngine.swift ToolCutter/SAM2Engine.swift ToolCutter/OutlineCorrection.swift "$(dirname "$out")/main.swift" -o "$out"
exec "$out"
