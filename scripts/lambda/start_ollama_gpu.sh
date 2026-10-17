#!/usr/bin/env bash
export OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=8 OLLAMA_KEEP_ALIVE=-1 OLLAMA_MODELS=$HOME/.ollama-gpu
exec ollama serve
