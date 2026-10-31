#!/bin/sh
# Builds the Unity plugin for macOS (without Core Motion) and checks it against golden.json.
# Regenerate the fixture after changing ml/src/threshold.py: python -m ml.src.threshold golden
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
xcrun clang++ -std=c++17 -fobjc-arc -Wall -Werror -DPOCKETSTEP_NO_COREMOTION=1 -framework Foundation \
  "$here/../Plugins/iOS/PocketStepCounter.mm" "$here/PluginTests.mm" -o "$out/plugin_tests"
"$out/plugin_tests" "$here/golden.json"
