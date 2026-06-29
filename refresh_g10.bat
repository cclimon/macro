@echo off
cd /d "%~dp0"

echo ============================================
echo  G10 FX Signal Dashboard -- Morning Refresh
echo ============================================
echo.

echo [1/3] Fetching data from Bloomberg...
python macro\main.py
if errorlevel 1 (
    echo.
    echo ERROR: Data fetch failed. Is Bloomberg Terminal open and logged in?
    pause
    exit /b 1
)

echo.
echo [2/3] Committing updated cache...
git add macro/data/cache/
git commit -m "Refresh G10 FX signals %date%"

echo.
echo [3/3] Pushing to GitHub...
git push origin main

echo.
echo ============================================
echo  Done! Streamlit Cloud will update shortly.
echo ============================================
echo.
pause
