@echo off
cd /d "%~dp0"

echo ============================================
echo  G10 FX Signal Dashboard -- Morning Refresh
echo ============================================
echo.

:: Save current branch so we can return after the refresh
for /f "tokens=*" %%b in ('git branch --show-current') do set CURRENT_BRANCH=%%b
echo Current branch: %CURRENT_BRANCH%

echo.
echo [1/4] Switching to main and pulling latest...
git checkout main
if errorlevel 1 (
    echo ERROR: Could not switch to main. Commit or stash your changes first.
    pause
    exit /b 1
)
git pull origin main
if errorlevel 1 (
    echo ERROR: git pull failed.
    pause
    exit /b 1
)

echo.
echo [2/4] Fetching data from Bloomberg...
"C:\Users\CarlosCliment\AppData\Local\Programs\Python\Python312\python.exe" macro\main.py
if errorlevel 1 (
    echo.
    echo ERROR: Data fetch failed. Is Bloomberg Terminal open and logged in?
    git checkout %CURRENT_BRANCH%
    pause
    exit /b 1
)

echo.
echo [3/4] Committing updated cache to main...
git add macro/data/cache/
git commit -m "Refresh G10 FX signals %date%"

echo.
echo [4/4] Pushing to GitHub...
git push origin main
if errorlevel 1 (
    echo ERROR: Push failed.
    git checkout %CURRENT_BRANCH%
    pause
    exit /b 1
)

echo.
echo Returning to %CURRENT_BRANCH%...
git checkout %CURRENT_BRANCH%

echo.
echo ============================================
echo  Done! Streamlit Cloud will update shortly.
echo ============================================
echo.
pause
