@echo off

set SCRIPT=run_ode.py

for /l %%i in (1, 1, 24) do (
  start cmd /c "python %SCRIPT% %%i"
)

echo "All instances started."