#!/bin/sh
set -eu
BUNDLE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec "${BUNDLE_ROOT}/ftllm-launch-webui" "$@"
