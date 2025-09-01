<!--
  Project Technologies: Snowflake, Ray, HuggingFace, OpenAI
-->

<p align="center"><b>Snowflake | Ray | HuggingFace | OpenAI</b></p>

<p align="center" style="font-size: 0.9em;">
  <a href="https://commons.wikimedia.org/wiki/File:Snowflake_Logo.svg#/media/File:Snowflake_Logo.svg">Snowflake Logo</a> |
  <a href="https://upload.wikimedia.org/wikipedia/commons/4/4d/OpenAI_Logo.svg">OpenAI Logo</a>
</p>

<h1 align="center">Distributed Multi-Node, Multi-GPU Media Transcription Pipeline</h1>

<p align="center">
  <b>Scalable, distributed audio & video transcription using <a href="https://www.snowflake.com/">Snowflake</a>, <a href="https://www.ray.io/">Ray</a>, <a href="https://huggingface.co/openai/whisper">OpenAI Whisper</a>, and <a href="https://huggingface.co/">HuggingFace</a>.</b>
</p>

---

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Implementation Steps](#implementation-steps)
- [Snowflake Setup](#snowflake-setup)
- [Running the Notebook](#running-the-notebook)
- [Pipeline Steps](#pipeline-steps)
- [Translation Step](#translation-step)
- [Notes & Troubleshooting](#notes--troubleshooting)
- [References](#references)

---

## Overview

- **Goal:** Transcribe large volumes of Hindi audio and video using distributed GPU resources, and translate the transcriptions to English.
- **Technologies:**
  - <a href="https://www.snowflake.com/">Snowflake</a> (stages, compute pools, external access integration)
  - <a href="https://www.ray.io/">Ray</a> (distributed data processing)
  - <a href="https://huggingface.co/openai/whisper">OpenAI Whisper</a> (speech-to-text)
  - <a href="https://huggingface.co/">HuggingFace Transformers</a>
  - <a href="https://www.python.org/">Python</a> (Jupyter notebook)

---

## Prerequisites

- **Snowflake Account** with:
  - Compute pools (GPU)
  - External access integration
  - Privileges to create databases, schemas, warehouses, and integrations
- **Python Environment** with:
  - Jupyter Notebook
  - `snowflake-snowpark-python`, `snowflake-ml-python`, `transformers`, `torch`, `ray`, `ffmpeg`, `pandas`, `numpy`, `streamlit`
- **FFmpeg** installed and available in PATH
- **HuggingFace Transformers** and OpenAI Whisper model access

---

## Implementation Steps

1. **Snowflake Setup**
   - Open [`snowflake_setup.sql`](./snowflake_setup.sql) in your Snowflake worksheet and run all statements to create the required database, schema, warehouse, compute pool, and network integration.

2. **Import Notebook & Configure Compute**
   - Import [`distributed_media_transcription_pipeline.ipynb`](./distributed_media_transcription_pipeline.ipynb) into your Snowflake ML runtime or compatible Jupyter environment.
   - Ensure the runtime/container is configured to use the GPU compute pool created in step 1.

3. **Enable Network External Access**
   - Make sure the external access integration (created by the SQL script) is enabled for your notebook/container session. This allows model downloads and outbound internet access.

4. **Initialize Notebook**
   - Run the first 2 cells to import packages and set up the Snowflake session and stage.

5. **Upload Media Files**
   - Upload your MP4 audio/video files to the Snowflake stage (`AUDIO_FILES_STAGE`) created by the setup script. This can be done via the Snowflake UI, CLI, or programmatically.

6. **Run the Pipeline**
   - Continue executing the remaining cells in order to process, transcribe, and store results. The pipeline will handle conversion, distributed inference, and result storage.

---

## Snowflake Setup

1. Edit and run [`snowflake_setup.sql`](./snowflake_setup.sql) in your Snowflake worksheet or via the Snowflake CLI. This will:
   - Create the database, schema, and warehouse
   - Set up a GPU compute pool
   - Configure network rules and external access integration
   - Grant necessary privileges

2. Upload your MP4 audio/video files to the created stage (`AUDIO_FILES_STAGE`).

---

## Running the Notebook

1. Open [`distributed_media_transcription_pipeline.ipynb`](./distributed_media_transcription_pipeline.ipynb) in Jupyter.
2. Ensure your Python environment has all required packages and is authenticated to Snowflake.
3. Step through the notebook cells in order:
   - **Imports & Session:** Import packages and initialize Snowflake session.
   - **Stage Setup:** Create and use the Snowflake stage for audio and video files.
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
   - Place your MP4 audio or video files in the Snowflake stage (`AUDIO_FILES_STAGE`).
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
