@echo off
setlocal
pushd "%~dp0.."
python -m knowledge_estimation.simulator %*
if errorlevel 1 pause
popd
endlocal
