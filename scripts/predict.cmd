@echo off
setlocal

REM Ensure we run from repo root
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%.."

REM Pass-through to Python prediction script
python -m src.converter.predict %*

popd
endlocal
