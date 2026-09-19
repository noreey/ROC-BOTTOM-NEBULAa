"""
Door predictor — TEMPLATE, not yet implemented.

Task: temporal segment detection + binary classification on a continuous stream of door-motor
telemetry (Motor current, Voltage, back-EMF, Door leaf position, switch/command flags — 17
columns, 50 Hz). Find each open/close cycle's start_time/end_time, classify each as
"Normal" or "Abnormal resistance".

Whoever implements this: keep the function signature and output schema below unchanged — the app
only needs `predict(uploaded_files)` to exist and return the right columns, everything inside it
is yours to write however you like.

Input file: a single continuous stream file (e.g. Test.csv), NOT one row per event — you must find
the segment boundaries yourself.

Output schema is different from the other three subsystems: no file_id column, one row per
PREDICTED SEGMENT (there can be many segments per uploaded file), start_time/end_time in the
same "Y-M-D-H-Mi-S-ms" format as the input Datetime column, prediction in
{"Normal", "Abnormal resistance"} exactly.
"""
import pandas as pd

SUBSYSTEM_NAME = "Door"
OUTPUT_FILENAME = "door_predictions.csv"
OUTPUT_COLUMNS = ["start_time", "end_time", "prediction"]
ACCEPTED_FILE_TYPES = ["csv"]


def predict(uploaded_files) -> pd.DataFrame:
    """uploaded_files: list of Streamlit UploadedFile objects, each a continuous Door telemetry
    stream (17 columns, see Door_Subsystem_Info_Kit.md for the schema).

    TODO (Door owner): replace this stub with real segmentation + classification.
    Must return a DataFrame with columns exactly: start_time, end_time, prediction
    — one row per predicted segment, across all uploaded files combined.
    """
    raise NotImplementedError(
        "Door predictor not implemented yet. See Door_Subsystem_Info_Kit.md for the task, "
        "and predict.py's expected columns above."
    )
