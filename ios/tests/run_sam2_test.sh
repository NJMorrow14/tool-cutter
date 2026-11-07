#!/bin/sh
# SAM 2.1 Small through SAM2Engine.swift (generic MLModel API), on the Mac: compile the mlpackages once, encode a real
# rear-camera drawer photo, prompt a point, check the mask, write tests/out/sam2_overlay.png (LOOK at it).
set -e
cd "$(dirname "$0")/.."
models=tests/out/sam2
mkdir -p "$models"
for pkg in ToolCutter/Models/SAM2_1Small*FLOAT16.mlpackage; do
  name=$(basename "$pkg" .mlpackage)
  if [ ! -d "$models/$name.mlmodelc" ] || [ "$pkg" -nt "$models/$name.mlmodelc" ]; then
    echo "compiling $name"
    xcrun coremlcompiler compile "$pkg" "$models" >/dev/null
  fi
done
photo=${SAM2_TEST_PHOTO:-../backend/captures/a237eba87dba/frame_000.jpg}
if [ ! -f "$photo" ]; then echo "FAIL test photo missing: $photo (set SAM2_TEST_PHOTO)"; exit 1; fi
out=$(mktemp -d)/sam2test
cp tests/sam2_test.swift "$(dirname "$out")/main.swift"
swiftc -O ToolCutter/OutlineEngine.swift ToolCutter/SAM2Engine.swift "$(dirname "$out")/main.swift" -o "$out"
exec "$out" "$models" "$photo" tests/out/sam2_overlay.png
