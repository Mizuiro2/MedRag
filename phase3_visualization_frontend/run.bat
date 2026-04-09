@echo off
cd /d "%~dp0"
G:\Anaconda\envs\rag_fyp\python.exe -m pip install -q flask flask-cors
G:\Anaconda\envs\rag_fyp\python.exe app.py
