@echo off
setlocal

REM Ensure we run from repo root
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%.."

python -m src.converter.extract_squares %*

popd
endlocal
