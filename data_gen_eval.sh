#!/bin/bash
# Evaluation script for mcmlnet gpumcml data generation

set -e  # Exit on any *unexpected* error
set -o pipefail

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

print_status()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

check_command() { command -v "$1" >/dev/null 2>&1 || { print_error "$1 is not installed or not in PATH"; exit 1; }; }

# Run a command with timeout and handle exit codes (0, 124 timeout, other = error)
run_with_timeout() {
  local seconds="$1"; shift
  local what="$1"; shift
  if timeout "$seconds" "$@"; then
    print_success "$what completed"
    return 0
  else
    local code=$?
    if [ "$code" -eq 124 ]; then
      print_warning "$what stopped after ${seconds}s (timeout)"
      return 0   # treat timeout as handled so script continues
    else
      print_error "$what failed with exit code $code"
      return "$code"
    fi
  fi
}

echo "=========================================="
echo "mcmlnet Data Generation Evaluation Script"
echo "=========================================="

print_status "Checking prerequisites..."
check_command python
check_command uv pip

[ -f ".env" ] || print_warning ".env file not found. Please ensure environment variables are set."

if [ -z "${CFG_DIR:-}" ]; then
  print_warning "CFG_DIR not set. Using default config path."
  export CFG_DIR="mcmlnet/training/configs/"
fi

print_status "Starting data generation evaluation process..."

# Step 1: Test gpumcml data generation
print_status "Step 1: Testing gpumcml functionality (Reflectance Data Generation) (~2 minutes)"

run_with_timeout 20 "Physiological data generation" \
  python mcmlnet/data_gen/physiological_data_script.py --run_id=0 --n_runs=20 --batch_size=10

run_with_timeout 20 "Physical generalization data generation" \
  python mcmlnet/data_gen/physical_generalization_data_script.py --run_id=0 --n_runs=20 --batch_size=10

run_with_timeout 40 "Empirical 1000M photons validation data generation" \
  python mcmlnet/data_gen/empirical_validation_1000M_photons_script.py --run_id=0 --n_runs=20 --n_batches=100000

# Step 2: Related work data reproducibility (~1 minute)
print_status "Step 2: Related work data reproducibility (~1 minute)"

input_nb="mcmlnet/data_gen/related_work_resimulation.ipynb"
out_dir="$(dirname "$input_nb")"
out_base="$(basename "$input_nb" .ipynb).executed"

run_with_timeout 600 "Related work resimulation notebook" \
  jupyter nbconvert \
    --to notebook \
    --execute "$input_nb" \
    --output "$out_base" \
    --output-dir "$out_dir" \
    --ExecutePreprocessor.kernel_name=python3 \
    --ExecutePreprocessor.timeout=600


print_status "All operations finished successfully. Please have a look at the intermediate files, simulated data batches and saved notebook."
