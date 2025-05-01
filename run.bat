@echo off

@echo off
REM Activate the virtual environment
call .venv\Scripts\activate

cd .\src
REM Run the API server with the specified parameters
python -m main

REM Keep the window open after execution
pause