@echo off
setlocal EnableExtensions
set "EXE=C:\SiftBuild\src\dist\Sift\Sift.exe"
set "LOG=C:\Users\Public\sift-service-compatible-checks.log"
set "RESULT=C:\Users\Public\sift-service-compatible-checks.exit"
if exist "%LOG%" del /q "%LOG%"
if exist "%RESULT%" del /q "%RESULT%"
call :run --platform-check || goto fail
call :run --integration-check || goto fail
call :run --analysis-check || goto fail
call :run --credential-store-check || goto fail
call :run --help || goto fail
> "%RESULT%" echo 0
exit /b 0

:run
>> "%LOG%" echo === %1 ===
"%EXE%" %1 >> "%LOG%" 2>&1
exit /b %ERRORLEVEL%

:fail
> "%RESULT%" echo 1
exit /b 1
