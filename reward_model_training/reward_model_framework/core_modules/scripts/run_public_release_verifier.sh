#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${FRAMEWORK_ROOT}/../.." && pwd)"

WORKING_REPO_CANDIDATE="${REPO_ROOT}/../eg3d_rlhf_code"
if [[ -d "${WORKING_REPO_CANDIDATE}" ]]; then
  WORKING_REPO_ROOT="$(cd "${WORKING_REPO_CANDIDATE}" && pwd)"
else
  WORKING_REPO_ROOT=""
fi

DEFAULT_WORKDIR="${REPO_ROOT}/release_verification_outputs/public_release_test"
DEFAULT_GENERATOR_PKL=""
DEFAULT_RWD_MODELS_DIR=""
if [[ -n "${WORKING_REPO_ROOT}" ]]; then
  DEFAULT_GENERATOR_PKL="${WORKING_REPO_ROOT}/pkl_pt/eg3d_1/ffhq512-128.pkl"
  DEFAULT_RWD_MODELS_DIR="${WORKING_REPO_ROOT}/reward_model_training/reward_model_framework/core_modules/RWD_MODELS_FOR_TUNING"
fi

CONDA_SH="${PUBLIC_RELEASE_VERIFY_CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${PUBLIC_RELEASE_VERIFY_CONDA_ENV:-hf_geom_eg3d_py39}"
WORKDIR="${2:-${PUBLIC_RELEASE_VERIFY_OUTDIR:-${DEFAULT_WORKDIR}}}"
GENERATOR_PKL="${PUBLIC_RELEASE_VERIFY_GENERATOR_PKL:-${EG3D_RLHF_ORIG_PKL:-${DEFAULT_GENERATOR_PKL}}}"
BASELINE_PKL="${PUBLIC_RELEASE_VERIFY_BASELINE_PKL:-${GENERATOR_PKL}}"
TUNED_PKL="${1:-${PUBLIC_RELEASE_VERIFY_TUNED_PKL:-}}"
REWARD_MODEL_ID="${PUBLIC_RELEASE_VERIFY_REWARD_MODEL_ID:-7wnzkgie}"
RWD_MODELS_DIR_VALUE="${PUBLIC_RELEASE_VERIFY_RWD_MODELS_DIR:-${RWD_MODELS_DIR:-${DEFAULT_RWD_MODELS_DIR}}}"
SEEDS="${PUBLIC_RELEASE_VERIFY_SEEDS:-28852,28853}"
TRUNCATION_PSI="${PUBLIC_RELEASE_VERIFY_TRUNCATION_PSI:-1.0}"
MESH_BANK_START_SEED="${PUBLIC_RELEASE_VERIFY_MESH_START_SEED:-9100000}"
MESH_BANK_NUM_SEEDS="${PUBLIC_RELEASE_VERIFY_MESH_NUM_SEEDS:-2}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Missing conda activation script: ${CONDA_SH}" >&2
  exit 1
fi
if [[ -z "${GENERATOR_PKL}" || ! -f "${GENERATOR_PKL}" ]]; then
  echo "Generator checkpoint not found. Set PUBLIC_RELEASE_VERIFY_GENERATOR_PKL or EG3D_RLHF_ORIG_PKL." >&2
  exit 1
fi
if [[ -z "${BASELINE_PKL}" || ! -f "${BASELINE_PKL}" ]]; then
  echo "Baseline checkpoint not found. Set PUBLIC_RELEASE_VERIFY_BASELINE_PKL." >&2
  exit 1
fi
if [[ -z "${TUNED_PKL}" || ! -f "${TUNED_PKL}" ]]; then
  echo "Tuned checkpoint not found. Pass it as the first argument or set PUBLIC_RELEASE_VERIFY_TUNED_PKL." >&2
  exit 1
fi
if [[ -z "${RWD_MODELS_DIR_VALUE}" || ! -d "${RWD_MODELS_DIR_VALUE}" ]]; then
  echo "Released reward-model bundle not found. Set PUBLIC_RELEASE_VERIFY_RWD_MODELS_DIR or RWD_MODELS_DIR." >&2
  exit 1
fi

mkdir -p "${WORKDIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${WORKDIR}/pycache}"
export RWD_MODELS_DIR="${RWD_MODELS_DIR_VALUE}"

echo "Running public release verifier"
echo "  workdir: ${WORKDIR}"
echo "  generator: ${GENERATOR_PKL}"
echo "  baseline: ${BASELINE_PKL}"
echo "  tuned: ${TUNED_PKL}"
echo "  reward models: ${RWD_MODELS_DIR}"

python "${FRAMEWORK_ROOT}/core_modules/scripts/verify_public_release.py"   --workdir "${WORKDIR}"   --seeds "${SEEDS}"   --truncation-psi "${TRUNCATION_PSI}"   --generator-pkl "${GENERATOR_PKL}"   --reward-model-id "${REWARD_MODEL_ID}"   --baseline-pkl "${BASELINE_PKL}"   --tuned-pkl "${TUNED_PKL}"   --mesh-bank-start-seed "${MESH_BANK_START_SEED}"   --mesh-bank-num-seeds "${MESH_BANK_NUM_SEEDS}"

echo
echo "Verification outputs written to: ${WORKDIR}"
echo "Summary JSON: ${WORKDIR}/verification_summary.json"
