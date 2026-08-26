@echo off
setlocal EnableExtensions
set "PATH=C:\SiftBuild\tools\uv;%PATH%"
set "POLARS_SKIP_CPU_CHECK=1"
set "UV_CACHE_DIR=C:\SiftBuild\uv-cache"
set "UV_PYTHON_INSTALL_DIR=C:\SiftBuild\python"
cd /d C:\SiftBuild\src
if exist C:\SiftBuild\service-finalize.exit del /q C:\SiftBuild\service-finalize.exit
if exist C:\SiftBuild\service-finalize.log del /q C:\SiftBuild\service-finalize.log
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File C:\SiftBuild\finalize-service-release.ps1 > C:\SiftBuild\service-finalize.log 2>&1
set "rc=%ERRORLEVEL%"
> C:\SiftBuild\service-finalize.exit echo %rc%
exit /b %rc%
