#!/bin/sh
# Downloads Apple's Core ML build of SAM 2.1 Small (Apache-2.0) into ios/ToolCutter/Models (gitignored, ~90 MB).
# Needs the Hugging Face CLI: `pip install -U huggingface_hub` (gives `hf`).
set -e
cd "$(dirname "$0")/../ToolCutter/Models"
hf download apple/coreml-sam2.1-small --local-dir . --include "*.mlpackage/*"
ls -d *.mlpackage
