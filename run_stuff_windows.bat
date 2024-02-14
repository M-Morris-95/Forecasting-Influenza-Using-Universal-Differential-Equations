@echo off

set SCRIPT=testing_no_interpolation.py

for /l %%i in (1, 1, 10) do (
  start cmd /c "python %SCRIPT% %%i"
)

echo "All instances started."