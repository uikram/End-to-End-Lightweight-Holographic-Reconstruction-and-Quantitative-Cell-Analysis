@echo off
REM ===========================================================================
REM  HoloQPI - clean up the project folder on this PC.
REM
REM  Run it by double-clicking, or from cmd:   CLEANUP_PC.cmd
REM
REM  WHAT IT DOES
REM    The project root currently holds an old, incomplete copy of the code and
REM    results, while the good complete copy sits in Server_Code. This throws
REM    away the old one and moves the good one up into the root.
REM
REM  WHAT IT KEEPS
REM    data\            the dataset - NEVER touched, it is the input
REM    weights\         the pretrained encoder - never touched
REM    Claude outputs\  your own folder - never touched
REM    runs\ figures\   the results, taken from Server_Code
REM    the code         holoqpi, scripts, config, docs, main.py, the .sh drivers
REM    logs_run\        every .out run transcript, gathered in one place
REM
REM  WHAT IT DELETES, AND YOU SHOULD KNOW BEFORE RUNNING
REM    runs_before_audit_20260916_181054\   the pre-audit run's results
REM    figures_before_audit_20260916_181054\
REM       Superseded. Nothing in the report uses them, and they are not
REM       comparable with the current run because the objective changed.
REM    logs\            the per-script .log folders, ~20 dated directories.
REM       The .out transcripts kept in logs_run\ carry the same summaries.
REM    .run_state\ .finish_state\
REM       Step markers. These MUST go: if they reach a fresh server, the
REM       drivers see them and skip every phase, silently doing nothing.
REM    holo_qpi_fix\ and holoqpi_code_audit_2026-09-16.zip
REM       The zips I sent, already applied.
REM
REM  If you want to keep the two *_before_audit folders, comment out their two
REM  lines below (put REM in front) before running.
REM ===========================================================================

setlocal
cd /d "%~dp0"

REM --- Safety gate ----------------------------------------------------------
if not exist "Server_Code\runs\RESULTS.md" (
  echo.
  echo STOP: Server_Code\runs\RESULTS.md was not found.
  echo.
  echo This script must sit in the project root, beside the Server_Code folder.
  echo Nothing has been changed.
  echo.
  pause
  exit /b 1
)
if not exist "data\phase" (
  echo.
  echo STOP: data\phase was not found, so this is not the project root.
  echo Nothing has been changed.
  echo.
  pause
  exit /b 1
)

echo.
echo ===========================================================================
echo  This will delete the old code and results from the project root and
echo  replace them with the complete copy in Server_Code.
echo.
echo  data\ and weights\ are NOT touched.
echo ===========================================================================
echo.
choice /c YN /m "Continue"
if errorlevel 2 (
  echo.
  echo Cancelled. Nothing has been changed.
  pause
  exit /b 0
)

echo.
echo [1/5] Removing the old, incomplete copy from the project root...

for %%D in (holoqpi scripts config docs runs figures logs .run_state .finish_state holo_qpi_fix) do (
  if exist "%%D\" rd /s /q "%%D"
)

REM The two pre-audit archives. Comment these two lines out to keep them.
if exist "runs_before_audit_20260916_181054\"    rd /s /q "runs_before_audit_20260916_181054"
if exist "figures_before_audit_20260916_181054\" rd /s /q "figures_before_audit_20260916_181054"

for %%F in (main.py run_v2.sh run_study.sh run_all.sh study.sh RUN_ON_SERVER.sh FINISH_ON_SERVER.sh RUN_THIS.md README.md requirements.txt environment.yml cleanup.sh cleanup.cmd holoqpi_code_audit_2026-09-16.zip) do (
  if exist "%%F" del /q "%%F"
)
if exist "*.out" del /q "*.out"

echo [2/5] Dropping the duplicate dataset and weights inside Server_Code...

REM One small file the root copy is missing: it records which parameters
REM produced the masks, and without it a later run cannot prove they are current.
if exist "Server_Code\data\mask\_provenance.json" (
  if exist "data\mask\" copy /y "Server_Code\data\mask\_provenance.json" "data\mask\" >nul
)
if exist "Server_Code\data\amplitude_reference\_provenance.json" (
  if exist "data\amplitude_reference\" copy /y "Server_Code\data\amplitude_reference\_provenance.json" "data\amplitude_reference\" >nul
)

for %%D in (data weights logs .run_state .finish_state runs_before_audit_20260916_181054 figures_before_audit_20260916_181054) do (
  if exist "Server_Code\%%D\" rd /s /q "Server_Code\%%D"
)
if exist "Server_Code\RUN_ON_SERVER copy.sh" del /q "Server_Code\RUN_ON_SERVER copy.sh"
if exist "Server_Code\holoqpi_code_audit_2026-09-16.zip" del /q "Server_Code\holoqpi_code_audit_2026-09-16.zip"

echo [3/5] Moving the clean copy into the project root...

for %%D in (holoqpi scripts config docs runs figures) do (
  if exist "Server_Code\%%D\" move "Server_Code\%%D" "%%D" >nul
)
for %%F in (main.py run_v2.sh run_study.sh run_all.sh study.sh RUN_ON_SERVER.sh FINISH_ON_SERVER.sh RUN_THIS.md README.md requirements.txt environment.yml cleanup.sh cleanup.cmd .gitignore .gitattributes) do (
  if exist "Server_Code\%%F" move "Server_Code\%%F" "%%F" >nul
)

echo [4/5] Gathering the run transcripts into logs_run\...

if not exist "logs_run\" mkdir "logs_run"
move "Server_Code\*.out" "logs_run\" >nul 2>&1

if exist "Server_Code\" rd /s /q "Server_Code"

echo [5/5] Checking the result...
echo.

set FAIL=0
for %%D in (holoqpi scripts config data weights runs figures) do (
  if exist "%%D\" (echo   OK       %%D\) else (echo   MISSING  %%D\ & set FAIL=1)
)
for %%F in (main.py run_v2.sh RUN_ON_SERVER.sh FINISH_ON_SERVER.sh runs\RESULTS.md) do (
  if exist "%%F" (echo   OK       %%F) else (echo   MISSING  %%F & set FAIL=1)
)

echo.
echo   holoqpi .py files found:
dir /s /b holoqpi\*.py 2>nul | find /c ".py"
echo   ^(this must read 39^)
echo.
echo   figures\ PNG files found:
dir /b figures\*.png 2>nul | find /c ".png"
echo   ^(this must read 18^)
echo.

if "%FAIL%"=="1" (
  echo ===========================================================================
  echo  SOMETHING IS MISSING - do not copy this to the server. Send me the list.
  echo ===========================================================================
) else (
  echo ===========================================================================
  echo  Clean. See the note about the server before you copy anything across.
  echo ===========================================================================
)
echo.
pause
