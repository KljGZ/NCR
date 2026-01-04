set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

DEFENSES=(avg flame scope fld alignins fltrust snowball multikrum fld)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BASE_ARGS=(
  --model cnn
  --dataset fashion_mnist
  --attack NCR
  --lr 0.1
)

mkdir -p save/NCR-fashion logs

for defence in "${DEFENSES[@]}"; do
  run_tag="NCR_fashion_${defence}_${TIMESTAMP}"
  log_path="logs/${run_tag}.log"
  echo "[INFO] Starting defence=${defence}, log=${log_path}"
  python main_NCR.py "${BASE_ARGS[@]}" --defence "${defence}" --save "save/NCR-fashion/${run_tag}" \
    | tee "${log_path}"
  if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "[WARN] Run with defence=${defence} exited with error, continuing to next task."
  fi
done

echo "[INFO] All defence runs dispatched."
