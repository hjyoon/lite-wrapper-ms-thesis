#!/bin/sh

cd "$(dirname "$0")" || exit

if [ -d "venv" ]; then
    . ./venv/bin/activate
    python run.py "$@"
fi
