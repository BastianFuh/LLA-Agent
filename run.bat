@echo off
echo Find conda installation...
REM --- Dynamically find the conda executable ---
for /f "delims=" %%i in ('where conda') do set CONDA_PATH=%%i

echo Found conda at %CONDA_PATH%
REM --- Strip the trailing "\Scripts\conda.exe" to get the root folder ---
REM Remove \Scripts\conda.exe or \Library\bin\conda.bat
set CONDA_ROOT=%CONDA_CMD%
set CONDA_ROOT=%CONDA_ROOT:\Scripts\conda.exe=%
set CONDA_ROOT=%CONDA_ROOT:\Library\bin\conda.bat=%


echo Found Conda at %CONDA_ROOT%
echo Activating lla-agent environment...
REM --- Call activate.bat inside the found Conda folder ---
call "%CONDA_ROOT%\Scripts\activate.bat" lla-agent

cd .\src
echo Starting LLA-Agent...
python main.py

REM Keep the window open after execution
pause