# Data Ingestion Module

## Overview

The Data Ingestion module is responsible for handling the download, quality check, compression, and upload of audio datasets to Hugging Face. This module is structured to support multiple data sources and ingestion processes, making it scalable and easy to extend.

## Module Structure

```
data/
└── ingestion/
    └── codecfake/
        ├── __init__.py
        ├── download.py
        ├── extract.py
        ├── codecfake_data_pipeline.py
        └── upload_to_huggingface.py
```