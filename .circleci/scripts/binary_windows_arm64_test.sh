#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"

export VC_YEAR=2022

pushd "$PYTORCH_ROOT"

./ci/windows_arm64/smoke_test.bat

popd
