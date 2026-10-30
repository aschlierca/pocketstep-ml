# PocketStep dataset

Each folder here is one recording session published with `python -m ml.src.ingest`:

```
<session_id>/
├── sensor_data.csv        raw accelerometer samples (g, seconds since session start)
├── step_labels.csv        annotated footfalls (left / right / unknown, validity, review state)
└── session_metadata.json  anonymous participant ID, pocket side, protocol and recording details
```

The format is documented in [docs/data_format.md](../../../docs/data_format.md).

- Participants are identified only by anonymous codes. Sessions are published only after the
  researcher confirms that the participant consented to public release.
- Do not edit these files by hand. To correct labels, fix them in the app, export again and re-run
  `python -m ml.src.ingest`; it replaces the labels and keeps the recording.
- Every push that changes this folder retrains the published model (see `ml/published/README.md`).
