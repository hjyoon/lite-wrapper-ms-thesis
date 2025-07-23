# lite-wrapper-ms-thesis

This repository contains a lightweight implementation of a simplified version of the core ideas from my master’s thesis.

> **Thesis Title**:  
> _Korean Speech-to-Text Summarization using Word Graph_  
> ([RISS Link](https://www.riss.kr/link?id=T15514634))

The pipeline integrates transcription, translation, and summarization tasks, optimized for easy local execution using recent lightweight Hugging Face models.

The pipeline supports `.flac`, `.wav`, and `.mp3` audio file formats.

## Setup

> **Tested with Python 3.12.11**  
> This project has been tested and verified to work with Python 3.12.11.  
> (Other Python 3.12.x versions may work, but 3.12.11 is recommended for full compatibility.)

### Step 1: Initialize environment

```bash
./init.sh
```

### Step 2: Download required models

Make sure you have [Git LFS](https://git-lfs.com/) installed.

You can download models manually using:

```bash
./models.sh
```

Or allow the pipeline to automatically download models if they are not already present locally.

## Usage

```bash
./run.sh <audio_path> [summary_out_path]
```

- `audio_path`: Path to the audio file to process (`.flac`, `.wav`, `.mp3`).
- `summary_out_path`: Optional path to output the summary text.

## Pipeline Workflow

The pipeline performs the following tasks:

1. Transcription (Korean): Converts Korean audio into text using Whisper.

2. Translation (Early Translation): Translates the full transcript into English.

3. Summarization:
   - Generates an English summary from the translated transcript.

   - Generates a Korean summary directly from the original transcript.

4. Translation (Late Translation): Translates the Korean summary into English.

5. Final Summary: Merges early and late English summaries into a single coherent summary.

## License

This repository is provided under the MIT License.
