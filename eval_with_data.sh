#!/bin/bash
# Evaluation script for mcmlnet
# To capture the logging, run as: ./eval.sh | tee eval_$(date +%Y%m%d_%H%M%S).log

set -e
set -o pipefail

echo "=========================================="
echo "mcmlnet Evaluation Script"
echo "=========================================="

# Start timer
SCRIPT_START_TIME=$(date +%s)
STEP_START_TIME=$SCRIPT_START_TIME

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

print_status()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to format seconds into readable time
format_time() {
  local seconds=$1
  local hours=$((seconds / 3600))
  local mins=$(((seconds % 3600) / 60))
  local secs=$((seconds % 60))

  if [ $hours -gt 0 ]; then
    printf "%dh %dm %ds" $hours $mins $secs
  elif [ $mins -gt 0 ]; then
    printf "%dm %ds" $mins $secs
  else
    printf "%ds" $secs
  fi
}

# Function to print elapsed time for current step and total
print_elapsed() {
  local step_name="$1"
  local current_time=$(date +%s)
  local step_elapsed=$((current_time - STEP_START_TIME))
  local total_elapsed=$((current_time - SCRIPT_START_TIME))

  echo -e "${GREEN}[TIME]${NC} $step_name completed in $(format_time $step_elapsed) | Total elapsed: $(format_time $total_elapsed)"
  STEP_START_TIME=$current_time
}

# Function to start a new step
start_step() {
  local step_name="$1"
  STEP_START_TIME=$(date +%s)
  echo ""
  echo "==========================================  "
  print_status "Starting: $step_name"
  echo "=========================================="
}

check_command() { command -v "$1" >/dev/null 2>&1 || { print_error "$1 is not installed or not in PATH"; exit 1; }; }

# Serial timeout runner: treats 124 (timeout) as handled warning so the script continues.
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
      return 0
    else
      print_error "$what failed with exit code $code"
      return $code
    fi
  fi
}

print_status "Checking prerequisites..."
check_command python
check_command uv pip
check_command jupyter

# .env notice
[ -f ".env" ] || print_warning ".env file not found. Please ensure environment variables are set."

print_status "Starting evaluation process..."
print_elapsed "Prerequisites check"

#############################################
# Step 1: Surrogate model and KAN training  #
#############################################
start_step "Step 1: Surrogate model and KAN training (~5 minutes)"

print_status "Running MLP training test..."
run_with_timeout 300 "MLP training test" \
  python mcmlnet/training/optimization/optimizer.py train_data_ratio=0.1 trainer.max_epochs=10 trainer.check_val_every_n_epoch=5

print_status "Running KAN training test..."
run_with_timeout 600 "KAN training test" \
  python mcmlnet/training/optimization/optimizer.py model=forward_surrogate_kan batch_size=128 train_data_ratio=0.01 trainer.max_epochs=10 trainer.check_val_every_n_epoch=5

print_elapsed "Step 1: Surrogate model and KAN training"

#############################################################
# Step 2: Analytical baselines and neural network training  #
#############################################################
start_step "Step 2: Analytical baselines and neural network training (~3 hours)"

print_status "Running analytical baselines notebook..."

input_nb="mcmlnet/training/optimization/fit_analytical_baselines.ipynb"
out_dir="$(dirname "$input_nb")"
out_base="$(basename "$input_nb" .ipynb).executed"

jupyter nbconvert \
  --to notebook \
  --execute "$input_nb" \
  --output "$out_base" \
  --output-dir "$out_dir" \
  --ExecutePreprocessor.kernel_name=python3 \

print_status "Running related work neural network training notebook..."

input_nb="mcmlnet/training/optimization/related_work_neural_network_training.ipynb"
out_dir="$(dirname "$input_nb")"
out_base="$(basename "$input_nb" .ipynb).executed"

jupyter nbconvert \
  --to notebook \
  --execute "$input_nb" \
  --output "$out_base" \
  --output-dir "$out_dir" \
  --ExecutePreprocessor.kernel_name=python3 \

print_elapsed "Step 2: Analytical baselines and neural network training"

##########################################
# Step 3: Surrogate model inference (NB) #
##########################################
start_step "Step 3: Surrogate model inference"
print_status "Running surrogate model inference notebook (10-15 minutes)..."

input_nb="mcmlnet/data_gen/surrogate_model_inference.ipynb"
out_dir="$(dirname "$input_nb")"
out_base="$(basename "$input_nb" .ipynb).executed"

jupyter nbconvert \
  --to notebook \
  --execute "$input_nb" \
  --output "$out_base" \
  --output-dir "$out_dir" \
  --ExecutePreprocessor.kernel_name=python3 \

print_elapsed "Step 3: Surrogate model inference"

#########################################
# Step 4: Evaluation scripts (Python)   #
#########################################
start_step "Step 4: Running evaluation scripts (~2 hours)"

print_status "Final model evaluation (no timeout specified)..."
python mcmlnet/experiments/evaluate_final_model.py

print_status "PCA analysis..."
export SURROGATE_MODEL=True
python mcmlnet/experiments/spectral_pca_analysis.py
export SURROGATE_MODEL=False
python mcmlnet/experiments/spectral_pca_analysis.py

print_status "Spectral recall analysis..."
export SURROGATE_MODEL=True
python mcmlnet/experiments/spectral_recall_analysis.py
export SURROGATE_MODEL=False
python mcmlnet/experiments/spectral_recall_analysis.py

print_status "Neural scaling analysis..."
python mcmlnet/experiments/discover_models.py
python mcmlnet/experiments/evaluate_scaling_models.py

print_status "Related work and other scaling analysis..."
python mcmlnet/experiments/evaluate_other_scaling_models.py
python mcmlnet/experiments/evaluate_related_scaling_models.py

print_status "Generating scaling plots..."
python mcmlnet/experiments/plot_scaling_results.py
python mcmlnet/experiments/plot_scaling_results_appendix.py

print_status "Explainability analysis..."
python mcmlnet/experiments/explainability.py

print_elapsed "Step 4: Evaluation scripts"

#############################################
# Step 5: Performance timing (notebook)     #
#############################################
start_step "Step 5: Performance timing analysis (<5 minutes)"
print_status "Timing analysis notebook..."

input_nb="mcmlnet/experiments/timings.ipynb"
out_dir="$(dirname "$input_nb")"
out_base="$(basename "$input_nb" .ipynb).executed"

jupyter nbconvert \
  --to notebook \
  --execute "$input_nb" \
  --output "$out_base" \
  --output-dir "$out_dir" \
  --ExecutePreprocessor.kernel_name=python3 \

print_elapsed "Step 5: Performance timing analysis"

# Final summary
echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
TOTAL_ELAPSED=$(($(date +%s) - SCRIPT_START_TIME))
print_success "All evaluation steps reached their intended stopping points."
echo -e "${GREEN}[TIME]${NC} Total script execution time: $(format_time $TOTAL_ELAPSED)"
print_status "Check results and artifacts in:"
print_status "- lightning_logs/ (training logs)"
print_status "- cache/ (cached computation results)"
print_status "- data/models/ (trained model checkpoints)"
print_status "- mcmlnet/**/.executed.ipynb (executed notebooks)"
print_status "- logs/ (stdout/stderr for any parallel tasks)"

print_status "To view training progress:"
print_status "tensorboard --logdir lightning_logs/"
echo ""
print_success "Evaluation script completed."
