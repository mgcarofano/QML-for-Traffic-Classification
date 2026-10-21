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
    mirage-test    Esegue il preset locale con il modello hybrid.
    flowpic-test   Esegue il preset locale con il modello resnet e il dataset FlowPic.
    baseline       Esegue il preset baseline con il dataset completo e 5 fold.

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

BASELINE=(
    --kfold.sub_sample_ratio 1.0 --kfold.test_folds 5
    --training.epochs 20 --training.batch_size 50
)

case "$PRESET" in
    mirage-test) ARGS=("${LOCAL[@]}" --model.name hybrid) ;;
    flowpic-test) ARGS=("${LOCAL[@]}" --model.name resnet --dataset.is_flowpic) ;;
    baseline) ARGS=("${BASELINE[@]}") ;;
    -h|--help|help|--list) help; exit 0 ;;
    *) echo "Preset sconosciuto: $PRESET. Usare --help." >&2; exit 2 ;;
esac

if [[ "${1:-}" == "--" ]]; then shift; fi
# for ARG in "$@"; do
#     case "$ARG" in
#         --source_session_id|--source_session_id=*|--exec_mode_train|--exec_mode_train=*|--no-exec_mode_train)
#             printf 'Modalità e sessione sorgente sono definite dal preset.\n' >&2; exit 2 ;;
#     esac
# done

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
