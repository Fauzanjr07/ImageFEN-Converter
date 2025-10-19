@echo off
setlocal

REM Ensure we run from repo root
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%.."

REM Pass-through to Python training script
python -m src.converter.train %*

popd
endlocal
