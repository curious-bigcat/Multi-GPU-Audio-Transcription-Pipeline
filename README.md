# Distributed Multi-Node, Multi-GPU Audio Transcription Pipeline

This project demonstrates a scalable, distributed audio transcription pipeline using Snowflake, Ray, and the OpenAI Whisper model. It is designed for large-scale, multi-node, multi-GPU inferencing, and includes Hindi-to-English translation using GPT-4.1. The workflow is implemented in a Jupyter notebook and is supported by a Snowflake SQL setup script.

---

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Snowflake Setup](#snowflake-setup)
- [Running the Notebook](#running-the-notebook)
- [Pipeline Steps](#pipeline-steps)
- [Translation Step](#translation-step)
- [Notes & Troubleshooting](#notes--troubleshooting)

---

## Overview

- **Goal:** Transcribe large volumes of Hindi audio (from video files) using distributed GPU resources, and translate the transcriptions to English.
- **Technologies:**
  - Snowflake (stages, compute pools, external access integration)
  - Ray (distributed data processing)
  - OpenAI Whisper (speech-to-text)
  - Transformers (HuggingFace)
  - Streamlit (optional UI)
  - Python (Jupyter notebook)

---

## Prerequisites

- **Snowflake Account** with access to:
  - Compute pools (GPU)
  - External access integration
  - Sufficient privileges to create databases, schemas, warehouses, and integrations
- **Python Environment** with:
  - Jupyter Notebook
  - `snowflake-snowpark-python`, `snowflake-ml-python`, `transformers`, `torch`, `ray`, `ffmpeg`, `pandas`, `numpy`, `streamlit`
- **FFmpeg** installed and available in PATH
- **HuggingFace Transformers** and OpenAI Whisper model access

---

## Snowflake Setup

1. Edit and run `setup.sql` in your Snowflake worksheet or via the Snowflake CLI. This will:
   - Create the database, schema, and warehouse
   - Set up a GPU compute pool
   - Configure network rules and external access integration
   - Grant necessary privileges

2. Upload your MP4 audio/video files to the created stage (`AUDIO_FILES_STAGE`).

---

## Running the Notebook

1. Open `Audio Processing - Distributed Inferencing v2.ipynb` in Jupyter.
2. Ensure your Python environment has all required packages and is authenticated to Snowflake.
3. Step through the notebook cells in order:
   - **Imports & Session:** Import packages and initialize Snowflake session.
   - **Stage Setup:** Create and use the Snowflake stage for audio files.
   - **File Handling:**
     - List MP4 files in the stage
     - Download locally
     - Convert to MP3 using ffmpeg
     - Upload MP3s back to the stage
   - **Cluster Management:**
     - Initialize Ray
     - Scale up to 5 nodes (or as needed)
     - Configure logging
   - **Data Loading:**
     - Use `SFStageBinaryFileDataSource` to load MP3s as a Ray dataset
   - **Model Setup:**
     - Load Whisper model (Hindi, large-v3)
     - Set device and batch size
   - **Distributed Inference:**
     - Define a class for batch inference using Whisper
     - Map batches over the Ray dataset
   - **Results Storage:**
     - Write transcriptions to a Snowflake table (`WHISPER_DEMO_OUTPUT`)
     - Query and view results

---

## Pipeline Steps

1. **Upload MP4 Files:**
   - Place your MP4 files in the Snowflake stage (`AUDIO_FILES_STAGE`).
2. **Convert to MP3:**
   - The notebook downloads MP4s, converts them to MP3 using ffmpeg, and uploads them back to the stage.
3. **Distributed Processing:**
   - Ray is initialized and scaled to the desired number of nodes/GPUs.
   - MP3 files are loaded as a distributed dataset.
4. **Transcription:**
   - The Whisper model is loaded on each node/GPU.
   - Audio files are transcribed in parallel batches.
   - Results are written to a Snowflake table.

---

## Translation Step

- The final SQL cell in the notebook creates a new table with English translations:
  - Uses `SNOWFLAKE.CORTEX.COMPLETE` with GPT-4.1 to translate Hindi transcriptions to English.
  - Stores both original and translated text in `WHISPER_DEMO_TRANSLATION`.

---

## Notes & Troubleshooting

- **Compute Pool:** Ensure your compute pool is active and has sufficient GPU nodes.
- **External Access:** The external access integration must allow outbound internet for model downloads.
- **FFmpeg:** Must be installed and accessible in the environment running the notebook.
- **Model Download:** The first run may take time as the Whisper model is downloaded.
- **Snowflake Privileges:** You may need ACCOUNTADMIN or SYSADMIN roles for setup.
- **Error Handling:** Check Ray and Snowflake logs for distributed errors.

---

## References
- [Snowflake ML Documentation](https://docs.snowflake.com/en/developer-guide/snowpark-ml)
- [Ray Documentation](https://docs.ray.io/en/latest/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
