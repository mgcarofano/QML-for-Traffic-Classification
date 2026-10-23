#!/usr/bin/env bash

#   ####################################################################    #
#   Opzioni di BASH

# Interrompe lo script quando un comando restituisce un codice di uscita diverso da zero
set -e

# Considera un errore l’utilizzo di una variabile non definita.
set -u

# Fa risultare fallita una pipeline se almeno uno dei suoi comandi fallisce.
set -o pipefail

#   ####################################################################    #
#   Funzioni di utilità

help() {
    cat <<'HELP'
Uso:
    ./Run.sh [--dry-run] PRESET [argomenti CLI aggiuntivi]

Preset:
    mirage-test         Esegue il preset locale con il modello hybrid.
    flowpic-test        Esegue il preset locale con il modello resnet e il dataset FlowPic.
    paper2025           Esegue il preset con iperparametri dal paper QML2025 (quantum).
    baseline-paper2026 
    quantum-paper2026   Esegue il preset con iperparametri dal paper QML2026.

--dry-run stampa il comando senza eseguirlo.
Gli argomenti CLI aggiuntivi sovrascrivono i valori del preset.
HELP
}

#   ####################################################################    #
#   Definizione dei preset.

# PRESET="${1:-baseline}"
# if (( $# )); then shift; fi

[[ -n "${1:-}" ]] || { echo "Nome del preset mancante. Usare --help." >&2; exit 2; }

PRESET="$1"
shift

LOCAL=(
    --kfold.sub_sample_ratio 0.01 --kfold.test_folds 1 --kfold.test_ratio 0.2
    --training.epochs 3 --training.batch_size 50 --no-training.early_stopping
)

PAPER_2025=(
    --requested_device cpu --model.n_shots None
    --model.name ampe_fixed_probs
    --kfold.test_folds 1 --kfold.test_ratio 0.2 --kfold.train_ratio 0.9
    --kfold.val_ratio 0.1 --training.epochs 100 --training.batch_size 32
    --training.early_stopping
    --optimizer.name "Adam" --optimizer.learning_rate 0.01
    --optimizer.beta1 0.9 --optimizer.beta2 0.999 --optimizer.epsilon 0.0000001
    --scheduler.name "NoSched"
)

PAPER_2026=(
    --kfold.test_folds 5 --kfold.train_ratio 0.8 --kfold.val_ratio 0.2
    --training.epochs 100 --training.batch_size 50
    --training.early_stopping --training.patience 10
    --optimizer.name "Adam" --optimizer.learning_rate 0.01
    --optimizer.beta1 0.9 --optimizer.beta2 0.999 --optimizer.epsilon 0.0000001
    --loss.name "CrossEntropy"
)

case "$PRESET" in
    mirage-test) ARGS=("${LOCAL[@]}" --model.name baseline) ;;
    flowpic-test) ARGS=("${LOCAL[@]}" --model.name resnet --dataset.is_flowpic) ;;
    paper2025) ARGS=("${QML_PAPER_2025[@]}") ;;
    baseline-paper2026) ARGS=("${PAPER_2026[@]}" --model.name baseline) ;;
    quantum-paper2026) ARGS=("${PAPER_2026[@]}" --requested_device cpu --model.name ampe_fixed_probs --model.n_shots None --model.n_qubits 5) ;;
    -h|--help|help|--list) help; exit 0 ;;
    *) echo "Preset sconosciuto: $PRESET. Usare --help." >&2; exit 2 ;;
esac

if [[ "${1:-}" == "--" ]]; then shift; fi

#   ####################################################################    #
#   Impostazione delle variabili ed esecuzione del comando.

PYTHON="${PYTHON:-python}"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    shift
fi

COMMAND=("$PYTHON" -u "Main.py" "${ARGS[@]}" "$@")

if "$DRY_RUN"; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${COMMAND[@]}"
