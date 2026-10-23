# Migrating the 2024 demo to 0.2

This is a breaking development revision, not a released compatibility patch.

| Old path | New path |
|---|---|
| Monolithic DINOStream class | SearchManager + configured sources |
| Raw TCP port 15555 commands | Versioned HTTP searches on loopback 8765 |
| Mandatory Ollama rewrite | Literal targets; optional structured suggestions |
| OpenCV display windows | Standalone browser previews |
| Automatic minute-by-minute JPEG history | Removed; explicit memory-limited search results |
| Interleaved multi-camera recording | Removed; no automatic recording |
| Model metadata labelled SAM2 | Independent grounded-sam2-stream package |
| Hardcoded webcam list | JSON source configuration |

The old DINOStream constructor and raw TCP protocol are not retained. dino_stream.py and multicam_demo.py now delegate to the documented CLI. video_demo.py runs a short search on the included video. Install the package before invoking them.

Old code remains recoverable in Git history (pre-migration commit 373a564); no Git history was rewritten. Existing licenses and assets are preserved. The llm/ollama.py wrapper keeps list_models/model_exists/send convenience methods while using validated local structured responses. It does not create or pull models.

To migrate a client, configure stable source IDs, POST targets/source_ids, wait for a terminal snapshot and use normalized boxes. A source finishing its file or failing no longer starves later sources. Source IDs no longer derive from underscore-delimited image filenames.

There is no historical-search replacement yet. SAM2 segmentation, speech transcription and tracking were not implemented by the old active streaming code and are not claimed by this revision.
