#!/bin/sh

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Error: git-lfs is not installed."
    echo "Please install git-lfs before running this script."
    exit 1
fi

if [ ! -d "./models/whisper-small" ]; then
    git clone https://huggingface.co/openai/whisper-small ./models/whisper-small
fi

if [ ! -d "./models/opus-mt-ko-en" ]; then
    git clone https://huggingface.co/Helsinki-NLP/opus-mt-ko-en ./models/opus-mt-ko-en
fi

if [ ! -d "./models/distilbart-cnn-12-6" ]; then
    git clone https://huggingface.co/sshleifer/distilbart-cnn-12-6 ./models/distilbart-cnn-12-6
fi

if [ ! -d "./models/kobart-summary-v3" ]; then
    git clone https://huggingface.co/EbanLee/kobart-summary-v3 ./models/kobart-summary-v3
fi
