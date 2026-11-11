"""
    ## cross_validation_report.py \n
    *di Mario Gabriele Carofano*

    ---

    Modulo per la gestione dei report degli esperimenti di cross-validation e holdout.
"""

#   ####################################################################    #
#   Librerie e costanti

from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Source.training_functions import TestResult
from Source.training_functions import \
    compute_test_metrics

from Source.visualization_functions import draw_confusion_matrix
from sklearn.metrics import \
    classification_report, \
    confusion_matrix

from Source.utils import \
    atomic_dump_json, \
    atomic_dump_text, \
    atomic_write, \
    check_positive_int

METRIC_NAMES = ("test_loss", "test_accuracy", "test_macrof1", "test_weightedf1")

#   ####################################################################    #
#   Funzioni di utilità

def print_metrics_table(table):
    """Stampa le metriche di test formattando il numero dei campioni in valori interi e i valori numerici in formato decimale con quattro cifre dopo la virgola."""

    print(table.to_string(
        index=False,
        float_format=lambda value: f"{value:g}",
        justify="right",
    ))

    # end

#   ####################################################################    #
#   Definizione della classe CrossValidationReport

class CrossValidationReport:
    """Gestisce il tracciamento e l'aggregazione statistica dei risultati degli esperimenti di K-fold cross-validation stratificata / holdout singolo.

    ## Note
    - I dati devono essere già validati e preprocessati dal chiamante. La classe verifica specifici vincoli del protocollo e dello stato dei fold, ma non esegue una validazione completa degli input.
    - Con un solo fold si identifica il protocollo holdout, per il quale non è richiesta la copertura dell'intero dataset. Con più fold, invece, i test pianificati devono coprire ogni campione esattamente una volta.
    - La finalizzazione distingue tra report completo, parziale e vuoto. Se nessun fold è stato valutato, non viene esportato alcun artefatto.

    ## Funzionalità
    - Registrazione dello stato di un fold nel suo ciclo di vita (avvio, esecuzione, interruzione o skip).
    - Registrazione delle metriche di test, delle informazioni di addestramento e dei tempi di esecuzione.
    - Raccolta e allineamento delle predizioni individuali con i metadati di dataset.
    - Calcolo delle statistiche descrittive per fold (medie e deviazioni standard campionarie).
    - Aggregazione complessiva delle predizioni (pooled / out-of-fold) e calcolo delle metriche globali.
    - Stampa a terminale del consuntivo di sessione con tabelle e dettagli di copertura.
    - Esportazione degli artefatti su disco:
        - cv_report.json (riepilogo strutturato);
        - fold_metrics.csv (tabella analitica di metriche e durate);
        - predictions_{oof|holdout}.csv (predizioni aggregate complete);
        - classification_report_{oof|holdout}.txt (report dettagliato di classificazione per classe);
        - confusion_matrix_{oof|holdout}.png (matrice di confusione normalizzata).

    Args:
        y_true (np.ndarray): Etichette reali codificate del dataset utilizzato nell'esperimento, dopo eventuali operazioni di sottocampionamento.
        metadata (pd.DataFrame): Metadati dei campioni, allineati per posizione a y_true. Deve contenere le colonne "FlowID" e "DatasetID".
        folds (list[tuple]): Split esterni pianificati. Ogni tupla contiene gli indici dei campioni non destinati al test e gli indici dei campioni di test, riferiti alle posizioni in y_true e metadata.
        class_names (list[str]): Nomi delle classi, ordinati secondo la codifica numerica delle etichette.
        activated (bool, optional): Indica se la raccolta dei risultati e la generazione del report sono abilitate. Defaults to True.
    
    Raises:
        ValueError: Durante l'inizializzazione, se sono previsti più fold e i test pianificati non coprono ogni campione esattamente una volta.
    """

    #   ################################################################    #
    #   Inizializzazione e funzioni ausiliarie.

    def __init__(self, y_true, metadata, folds, class_names, activated: bool = True):

        #	############################################################	#
        #   Inizializzazione degli attributi interni.
        #   I dati devono essere già validati e preprocessati dal chiamante.

        # Se non è richiesta l'attivazione del report, l'inizializzazione viene interrotta.
        self.activated = activated
        if not activated:
            return

        self.y_true = y_true

        self.metadata = metadata[["FlowID", "DatasetID"]].reset_index(drop=True).copy()

        self.folds = folds
        self.protocol = "holdout" if len(folds) == 1 else "stratified_kfold"

        self.class_names = class_names
        self.num_classes = len(class_names)

        self.expected_tests = []
        self.rows = []
        self.predictions = []

        self.test_counts = np.zeros(self.y_true.size, dtype=np.int64)
        planned_counts = np.zeros_like(self.test_counts)

        for fold_index, (non_test, test) in enumerate(folds):

            self.expected_tests.append(test)
            planned_counts[test] += 1

            self.rows.append({
                "fold_index": fold_index,
                "status": "not_evaluated",
                "reason": None,
                "n_train": None, "n_val": None, "n_test": int(test.size),
                "best_epoch": None, "monitored_metric": None,
                "best_validation_score": None, "training_interrupted": None,
                **dict.fromkeys(METRIC_NAMES),
                "training_time_seconds": None,
                "fold_time_seconds": None,
            })

            # end for fold_index, (non_test, test)

        if self.protocol != "holdout" and not np.all(planned_counts == 1):
            raise ValueError(
                "Errore nell'inizializzazione del report: \n"
                "I test pianificati non coprono ogni campione esattamente una volta."
            )

        self.planned_samples = int(np.count_nonzero(planned_counts))

        # end

    def _row(self, fold_index: int) -> dict:
        """Restituisce il report del fold specificato dopo aver validato l'indice.

        Args:
            fold_index (int): Indice del fold da restituire.

        Returns:
            dict: Report del fold indicato.
        """

        fold_index = check_positive_int(fold_index, "fold_index", allow_zero=True, maximum=len(self.rows) - 1)
        return self.rows[int(fold_index)]

        # end

    def _print_progress(self, report_status):
        """Stampa lo stato dei fold, i contatori e le coperture del dataset allo stato corrente dell'elaborazione."""

        def print_fold_details(status, label):
            indices = [row["fold_index"] for row in self.rows if row["status"] == status]
            details = f" {indices}" if indices else ""
            print(f"{label}: {len(indices)} / {len(self.rows)}{details}")

            # end

        print(f"\nRiepilogo {self.protocol}: {report_status}")
        print_fold_details("evaluated", "Fold valutati")
        print_fold_details("skipped", "Fold saltati")
        print_fold_details("not_evaluated", "Fold non valutati")

        covered = int(np.count_nonzero(self.test_counts))
        print(
            f"Copertura dataset: {covered} / {self.y_true.size} "
            f"({100.0 * covered / self.y_true.size:.2f}%)\n"
            f"Copertura test pianificati: {100.0 * covered / self.planned_samples:.2f}%"
        )

        # end

    #   ####################################################################    #
    #   Funzioni principali.

    def start_fold(
            self,
            fold_index: int,
            train_indices, val_indices, test_indices,
            activated: bool = True) -> None:
        """Aggiorna il report della sessione per il fold specificato registrando le lunghezze dei dati di train e validation. I dati devono essere già validati e preprocessati dal chiamante.

        Args:
            fold_index (int): Indice del fold da aggiornare.
            train_indices (np.ndarray): Indici dei campioni di training.
            val_indices (np.ndarray): Indici dei campioni di validation.
            test_indices (np.ndarray): Indici dei campioni di test.
            activated (bool, optional): Indica se il report è attivato. Defaults to True.

        Raises:
            ValueError: Se il fold è già avviato o finalizzato.
            ValueError: Se gli indici dei campioni di test non corrispondono a quelli pianificati.
        """

        if activated != self.activated:
            raise ValueError(
                f"Errore nell'aggiornamento del report per il 'fold{fold_index}': \n"
                "Lo stato di attivazione del report non corrisponde a quello previsto."
            )

        # Se il report non è stato attivato, termina l'esecuzione della funzione.
        if not activated:
            return

        row = self._row(fold_index)
        if (row["n_train"] is not None or row["n_val"] is not None) or row["status"] != "not_evaluated":
            raise ValueError(
                f"Errore nell'aggiornamento del report per il 'fold{fold_index}': \n"
                "Fold già avviato o finalizzato."
            )

        if not np.array_equal(test_indices, self.expected_tests[fold_index]):
            raise ValueError(
                f"Errore nell'aggiornamento del report per il 'fold{fold_index}': \n"
                "Indici dei campioni di test diversi da quelli pianificati."
            )

        # Il numero di campioni di test viene già fornito in __init__().
        row.update(
            status= "started",
            n_train=int(train_indices.size),
            n_val=int(val_indices.size)
        )

        # end

    def skip_fold(
            self, fold_index: int, reason: str,
            training_time_seconds=None, fold_time_seconds=None,
            training_interrupted=None,
            activated: bool = True,
        ) -> None:
        """Aggiorna il report della sessione registrando il fold come saltato. I dati devono essere già validati e preprocessati dal chiamante.

        Args:
            fold_index (int): Indice del fold da aggiornare.
            reason (str): Motivo per cui il fold viene saltato.
            training_time_seconds (float, optional): Tempo di training trascorso prima di saltare il fold. Defaults to None.
            fold_time_seconds (float, optional): Tempo di elaborazione del fold prima di saltarlo. Defaults to None.
            training_interrupted (bool, optional): Indica se il training è stato interrotto. Defaults to None.
            activated (bool, optional): Indica se il report è attivato. Defaults to True.

        Raises:
            ValueError: Se il fold è già avviato o finalizzato.
        """

        if activated != self.activated:
            raise ValueError(
                f"Errore nell'aggiornamento del report per il 'fold{fold_index}': \n"
                "Lo stato di attivazione del report non corrisponde a quello previsto."
            )

        # Se il report non è stato attivato, termina l'esecuzione della funzione.
        if not activated:
            return

        row = self._row(fold_index)
        if row["status"] == "evaluated":
            raise ValueError(
                f"Errore nell'aggiornamento del report per il 'fold{fold_index}': \n"
                "Atteso: fold inizializzato. Ricevuto: 'evaluated'"
            )

        row.update(
            status= "skipped",
            reason= str(reason),
            training_interrupted= None if training_interrupted is None else bool(training_interrupted),
            training_time_seconds= training_time_seconds,
            fold_time_seconds= fold_time_seconds,
        )

        # end

    def finalize_fold(
            self, fold_index: int, result: TestResult,
            best_epoch=None, monitored_metric=None, best_validation_score=None,
            training_interrupted=None,
            training_time_seconds=None, fold_time_seconds=None,
            activated: bool = True,
        ) -> None:
        """Aggiorna il report della sessione registrando i risultati del fold. I dati devono essere già validati e preprocessati dal chiamante. Inoltre, accumula una copia delle predizioni associandole agli indici di test pianificati e ai relativi metadati, in modo da calcolare le vere metriche aggregate (pooled/out-of-fold metrics).

        Args:
            fold_index (int): Indice del fold da aggiornare.
            result (TestResult): Risultati del test del fold.
            best_epoch (int, optional): Epoca migliore durante l'addestramento. Defaults to None.
            monitored_metric (str, optional): Nome della metrica monitorata. Defaults to None.
            best_validation_score (float, optional): Miglior punteggio di validazione ottenuto. Defaults to None.
            training_interrupted (bool, optional): Indica se l'addestramento è stato interrotto. Defaults to None.
            training_time_seconds (float, optional): Tempo di addestramento in secondi. Defaults to None.
            fold_time_seconds (float, optional): Tempo totale del fold in secondi. Defaults to None.
            activated (bool, optional): Indica se il report è attivato. Defaults to True.

        Raises:
            ValueError: Se il fold è già stato valutato o se le predizioni non sono allineate con gli indici di test.
            TypeError: Se il risultato del test non è un'istanza di TestResult.
            ValueError: Se le predizioni del fold sono incomplete o non allineate con gli indici di test.
        """

        #   ############################################################    #
        #   Verifica degli input.

        if activated != self.activated:
            raise ValueError(
                f"Errore nell'aggiornamento del report per il 'fold{fold_index}': \n"
                "Lo stato di attivazione del report non corrisponde a quello previsto."
            )

        if result is not None and not isinstance(result, TestResult):
            raise TypeError(
                f"Errore nell'aggiornamento del report per il 'fold{fold_index}': \n"
                "Ricevuti risultati del test non strutturati."
            )

        #   ############################################################    #
        #   CASO 1: Report non attivo
        #   Stampa secondo la formattazione del report le metriche di test ottenute e poi ritorna.

        if not activated:

            if result is None:
                return

            table = pd.DataFrame([{
                "n_test": len(result.predictions),
                **{name: result.metrics.get(name) for name in METRIC_NAMES},
            }], columns=["n_test", *METRIC_NAMES])

            if not table.empty:
                print("\nMetriche del Test set:")
                print_metrics_table(table)

            return

        #   ############################################################    #
        #   CASO 2: Report attivo

        # Recupera e verifica la riga corrispondente al fold corrente.
        row = self._row(fold_index)
        if row["status"] != "started":
            raise ValueError(
                f"Errore nell'aggiornamento del report per il 'fold{fold_index}': \n"
                "Fold non avviato o già finalizzato."
            )

        # Estrae e verifica gli indici dei test e le predizioni corrispondenti per il fold corrente.
        test_indices = self.expected_tests[fold_index]
        test_predictions = result.predictions
        if len(test_predictions) != test_indices.size:
            raise ValueError(
                f"Errore nell'aggiornamento del report per il 'fold{fold_index}': \n"
                "Predizioni del fold incomplete o non allineate."
            )

        # Prepara le predizioni del test con le informazioni aggiuntive: Fold, SampleIndex e DatasetID.
        pooled = test_predictions.reset_index(drop=True).copy()
        pooled.insert(0, "DatasetID", self.metadata["DatasetID"].iloc[test_indices].to_numpy())
        pooled.insert(0, "Fold", int(fold_index))
        pooled.insert(0, "SampleIndex", test_indices)
        self.predictions.append(pooled)
        self.test_counts[test_indices] += 1

        # Aggiornamento dello stato del fold corrente con i risultati del test.
        row.update(
            status="evaluated",
            best_epoch= int(best_epoch), monitored_metric= str(monitored_metric),
            best_validation_score= best_validation_score,
            training_interrupted= None if training_interrupted is None else bool(training_interrupted),
            **result.metrics,
            training_time_seconds= training_time_seconds,
            fold_time_seconds= fold_time_seconds,
        )

        #   ############################################################    #
        #   Stampa il progresso parziale.
        #   Stampa la tabella riepilogativa di tutti i fold finora valutati.

        self._print_progress("partial")
        table = pd.DataFrame([{
                "fold_index": row["fold_index"],
                "status": row["status"],
                "n_test": row["n_test"],
                **{name: row.get(name) for name in METRIC_NAMES},
            }
            for row in self.rows
            if row["status"] == "evaluated"
        ], columns=["fold_index", "status", "n_test", *METRIC_NAMES])

        if not table.empty:
            print("\nTabella riepilogativa dei fold: ")
            print_metrics_table(table)

        for name in METRIC_NAMES:
            table[name] = pd.to_numeric(table[name], errors="coerce")

        # end

    def finalize_report(self, report_dir, activated: bool = True):
        """Finalizza il report della cross-validation. Salva i file di report nella directory specificata. Controlla la coerenza delle predizioni aggregate e dei fold completati.

        Args:
            report_dir: Directory in cui salvare i file di report della cross-validation.
            activated (bool, optional): Indica se il report è attivato. Defaults to True.

        Raises:
            ValueError: Se ci sono predizioni duplicate per lo stesso indice del dataset.
            ValueError: Se il numero di predizioni aggregate non corrisponde alla copertura pianificata.
        """

        if activated != self.activated:
            raise ValueError(
                f"Errore nella finalizzazione del report: \n"
                "Lo stato di attivazione del report non corrisponde a quello previsto."
            )

        # Se il report non è stato attivato, termina l'esecuzione della funzione.
        if not activated:
            return

        #   ############################################################    #
        #   Preparazione dei percorsi per i file di report.

        report_dir = Path(report_dir)
        suffix = "holdout" if self.protocol == "holdout" else "oof"
        plot_prefix = 'Holdout' if suffix == 'holdout' else 'Out-of-fold'

        cv_report_path = report_dir / "cv_report.json"
        fold_metrics_path = report_dir / "fold_metrics.csv"

        pooled_predictions_path = report_dir / f"predictions_{suffix}.csv"
        pooled_classification_path = report_dir / f"classification_report_{suffix}.txt"
        pooled_confusion_matrix_path = report_dir / f"confusion_matrix_{suffix}.png"

        #   ############################################################    #
        #   Seleziona i fold completati.
        #   Concatena le predizioni out-of-fold / holdout ordinate per indice del campione.
        #   Verifica la validità delle predizioni aggregate.

        summary_columns = ["fold_index", "status", "n_test", *METRIC_NAMES]
        integer_columns = ["fold_index", "n_train", "n_val", "n_test", "best_epoch"]
        rows_df = pd.DataFrame(self.rows).astype({col: "Int64" for col in integer_columns})
        summary_df = rows_df.loc[:, summary_columns].copy()

        evaluated = rows_df[rows_df["status"] == "evaluated"].to_dict(orient="records")
        count = len(evaluated)

        pooled = (
            pd.concat(self.predictions, ignore_index=True)
            .sort_values("SampleIndex")
            .reset_index(drop=True)
            if self.predictions
            else pd.DataFrame(columns=["SampleIndex", "Fold", "DatasetID", "FlowID", "Actual", "Predicted"])
        )

        covered = int(np.count_nonzero(self.test_counts))
        if np.any(self.test_counts > 1):
            raise ValueError(
                "Errore nella finalizzazione del report: \n"
                "Ci sono predizioni ripetute per lo stesso indice del dataset."
            )
        if len(pooled) != int(self.test_counts.sum()):
            raise ValueError(
                "Errore nella finalizzazione del report: \n"
                "Il numero di predizioni aggregate non corrisponde al numero di campioni di test attesi."
            )

        report_status = (
            "complete"
            if (count == len(self.rows)) and (covered == self.planned_samples)
            else "empty" if count == 0 else "partial"
        )

        #   ############################################################    #
        #   Calcolo delle statistiche sui fold e delle metriche aggregate (pooled).

        fold_metrics = {}
        for metric in METRIC_NAMES:
            values = np.array([row[metric] for row in evaluated], dtype=float)
            fold_metrics[metric] = {
                "mean": float(values.mean()) if count > 0 else None,
                "std": float(values.std(ddof=1)) if count > 1 else None,
            }

        pooled_metrics = compute_test_metrics(
            pooled["Actual"].to_numpy(),
            pooled["Predicted"].to_numpy(),
            self.num_classes
        ) if count > 0 else None

        summary = {
            "status": report_status,
            "folds": {
                "evaluated_folds": count,
                "not_evaluated_folds": [row["fold_index"] for row in self.rows if row["status"] == "not_evaluated"],
                "skipped_folds": [row["fold_index"] for row in self.rows if row["status"] == "skipped"],
                "statistics": fold_metrics,
            },
            "coverage": {
                "n_samples": int(self.y_true.size),
                "planned_test_samples": self.planned_samples,
                "evaluated_test_samples": int(covered),
                "dataset_coverage": covered / int(self.y_true.size),
                "planned_test_coverage": covered / self.planned_samples,
            },
            "pooled_metrics": pooled_metrics,
        }

        #   ############################################################    #
        #   Salvataggio su disco del riepilogo JSON e delle metriche per fold.

        def normalize_report(value, path="summary"):
            """Converte scalari NumPy e NaN in valori compatibili con JSON."""

            # Normalizza ricorsivamente i valori di un dizionario.
            if isinstance(value, dict):
                return {
                    key: normalize_report(item, f"{path}.{key}")
                    for key, item in value.items()
                }

            # Normalizza ricorsivamente i valori di una lista o tupla.
            if isinstance(value, (list, tuple)):
                return [
                    normalize_report(item, f"{path}[{index}]")
                    for index, item in enumerate(value)
                ]

            # Normalizza gli scalari NumPy nei corrispondenti tipi Python.
            if isinstance(value, np.generic):
                return normalize_report(value.item(), path)

            # Normalizza i valori float, gestendo NaN e infiniti.
            if isinstance(value, float):
                if math.isnan(value):
                    print(f"\nValore NaN nel report: {path} -> convertito in None")
                    return None

                if not math.isfinite(value):
                    print(f"\nValore INF nel report: {path} -> convertito in None")
                    return None

            return value

            # end

        if report_status != "empty":
            atomic_dump_json(cv_report_path, normalize_report(summary))
            atomic_dump_text(
                fold_metrics_path,
                rows_df.to_csv(index=False, lineterminator="\n")
            )

            # end if

        #   ############################################################    #
        #   Generazione degli artefatti aggregati (predizioni, report e matrici).

        if not pooled.empty:

            labels = np.arange(self.num_classes)

            # Salvataggio delle predizioni aggregate (inclusi i soft-output).
            atomic_dump_text(pooled_predictions_path, pooled.to_csv(index=False, lineterminator="\n"))

            # Salvataggio del report di classificazione per classe.
            report = classification_report(
                pooled["Actual"],
                pooled["Predicted"],
                labels=labels,
                target_names=self.class_names,
                digits=4,
                zero_division=0,
            )

            atomic_dump_text(pooled_classification_path, report)

            # Calcolo e salvataggio in CSV della matrice di confusione grezza e normalizzata.
            # raw_cm = confusion_matrix(pooled["Actual"], pooled["Predicted"], labels=labels)
            norm_cm = confusion_matrix(pooled["Actual"], pooled["Predicted"], labels=labels, normalize="true")
            # for name, matrix in (("", raw_cm), ("_normalized", norm_cm)):
            #     table = pd.DataFrame(matrix, index=labels, columns=labels)
            #     table.index.name = "Actual/Predicted"
            #     atomic_dump_text(
            #         output_dir / f"confusion_matrix_{suffix}{name}.csv",
            #         table.to_csv(lineterminator="\n"),
            #     )

            # Disegno e salvataggio grafico della matrice di confusione normalizzata.
            figure = plt.figure(figsize=(10, 8))
            try:
                plt.tight_layout()
                plt.title(f"{plot_prefix} — {summary['status']}")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")

                draw_confusion_matrix(norm_cm, self.class_names)
                atomic_write(
                    pooled_confusion_matrix_path,
                    lambda stream: figure.savefig(stream, format="png", bbox_inches="tight"),
                )
            finally:
                plt.close(figure)

            # end if pooled.empty

        #   ############################################################    #
        #   Stampa a terminale del riepilogo della sessione.

        cov = summary["coverage"]
        self._print_progress(summary["status"])

        #   ############################################################    #
        #   Stampa a terminale una tabella riepilogativa dei fold.

        # Consente di aggiungere etichette testuali alla colonna degli indici.
        summary_df["fold_index"] = summary_df["fold_index"].astype(object)

        # Inizializzazione delle righe di riepilogo (media, std)
        # Inserisce tutte celle vuote, che verranno poi popolate con le statistiche dei fold.
        mean_row = {column: None for column in summary_columns}
        mean_row["fold_index"] = "Media"

        std_row = {column: None for column in summary_columns}
        std_row["fold_index"] = "STD campionaria"

        for name in METRIC_NAMES:
            stats = fold_metrics.get(name, {})
            mean_row[name] = stats.get("mean")
            std_row[name] = stats.get("std")
        
        # Inizializzazione delle righe di riepilogo (statistiche aggregate)
        pooled_row = {column: None for column in summary_columns}
        pooled_row["fold_index"] = "Aggregate"
        pooled_row["n_test"] = int(cov['evaluated_test_samples'])

        pooled_metrics = summary.get("pooled_metrics")
        if pooled_metrics is not None:
            for name in METRIC_NAMES:
                pooled_row[name] = pooled_metrics.get(name)
        else:
            pooled_row["status"] = "N/D"

        # Aggiunge le tre righe riepilogative al DataFrame dei fold.
        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    [mean_row, std_row, pooled_row],
                    columns=summary_columns,
                ).fillna(""),
            ],
            ignore_index=True,
        )

        # Stampa a terminale la tabella riepilogativa dei fold con le righe di riepilogo.
        if report_status != "empty" and not summary_df.empty:
            print("\nTabella riepilogativa dei fold: ")
            print_metrics_table(summary_df)

        # end
