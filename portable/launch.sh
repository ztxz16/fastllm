#!/usr/bin/env bash

set -euo pipefail

FTLLM_BUNDLE_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "${FTLLM_BUNDLE_ROOT}/ftllm" launch "$@"
