#!/bin/bash
set -e
cd "$(dirname "$0")"
pip install pyinstaller
pyinstaller --onefile --name livescribe-server main.py
cp dist/livescribe-server .
