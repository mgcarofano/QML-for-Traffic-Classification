#!/usr/bin/env bash

set -euo pipefail

# Parametri posizionali con valore di default per results_dir
SOURCE_SESSION_ID="${1:-}"
RESULTS_DIR="${2:-Results}"

if [[ -z "$SOURCE_SESSION_ID" ]]; then
    echo "Errore: 'source_session_id' non specificato." >&2
    echo "Uso: $0 <source_session_id> [results_dir]" >&2
    exit 2
fi

# Verifica l'esistenza della directory dei risultati
if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Configurazione della sessione '$SOURCE_SESSION_ID' non trovata in '$RESULTS_DIR'." >&2
    exit 1
fi

# Risolve il percorso assoluto della cartella dei risultati
ABS_RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd -P)"

# Cerca i file 'config.json' contenuti in cartelle con nome pari a SOURCE_SESSION_ID
# Salva i percorsi assoluti trovati in un array
MATCHES=()
while IFS= read -r -d '' config_file; do
    # Verifica che il genitore immediato sia esattamente la sessione cercata
    parent_dir="$(dirname "$config_file")"
    if [[ "$(basename "$parent_dir")" == "$SOURCE_SESSION_ID" ]]; then
        MATCHES+=("$parent_dir")
    fi
done < <(find "$ABS_RESULTS_DIR" -type f -name "config.json" -print0)

# Gestione casi: 0 occorrenze, > 1 occorrenze, 1 occorrenza
COUNT="${#MATCHES[@]}"

if (( COUNT == 0 )); then
    echo "Configurazione della sessione '$SOURCE_SESSION_ID' non trovata in '$RESULTS_DIR'." >&2
    exit 1
elif (( COUNT > 1 )); then
    echo "Sono state trovate più configurazioni per la sessione '$SOURCE_SESSION_ID':" >&2
    for match in "${MATCHES[@]}"; do
        echo "  - '$match'" >&2
    done
    exit 2
fi

# Restituisce il percorso assoluto della cartella della sessione
echo "'${MATCHES[0]}'"
