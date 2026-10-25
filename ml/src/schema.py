"""Names and allowed values of the exported session files (schema version 1.0).

These must match the iOS app's export; see docs/data_format.md.
"""

SUPPORTED_SCHEMA_VERSIONS = ("1.0",)

SENSOR_FILE = "sensor_data.csv"
LABELS_FILE = "step_labels.csv"
METADATA_FILE = "session_metadata.json"

TIME = "timestamp_seconds"
ELAPSED = "elapsed_seconds"
AXES = ("accel_x", "accel_y", "accel_z")
SENSOR_COLUMNS = (TIME, ELAPSED, *AXES)

LABEL_COLUMNS = ("step_id", TIME, "foot_label", "label_source", "is_valid", "notes")
# Written by the PocketStep app; optional so hand-made label files still load.
OPTIONAL_LABEL_COLUMNS = ("reviewed",)

FOOT_LABELS = ("left", "right", "unknown")
LABEL_SOURCES = ("manual", "assisted")
SIDES = ("left", "right")
WALKING_CONDITIONS = ("slow", "comfortable", "brisk")

METADATA_REQUIRED = (
    "schema_version",
    "participant_id",
    "session_id",
    "phone_model",
    "ios_version",
    "pocket_side",
    "starting_foot",
    "target_steps",
    "walking_condition",
    "sampling_interval_seconds",
    "session_start",
    "session_end",
    "notes",
)
