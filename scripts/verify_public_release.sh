#!/usr/bin/env bash
cd "$(dirname "$0")/.."

WORKDIR="${PUBLIC_RELEASE_VERIFY_OUTDIR:-$PWD/release_verification_outputs/public_release_test}"

# Run the maintained public release verifier against the provided tuned checkpoint.
python reward_model_training/reward_model_framework/core_modules/scripts/verify_public_release.py --workdir "$WORKDIR" --tuned-pkl "$1"
