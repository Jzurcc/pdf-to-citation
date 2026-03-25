@echo off
title Auto-APA Citation Generator
setlocal enabledelayedexpansion

echo ===================================================
echo       Auto-APA Citation Generator (ArXiv)
echo ===================================================
echo.

:: 1. CHECK PYTHON DEPENDENCIES
echo [System] Checking Python dependencies...
python -c "import fitz, requests, citeproc, dotenv, tqdm" >nul 2>&1
if %errorlevel% neq 0 (
    echo [Setup] Missing required Python libraries! Installing them now...
    pip install PyMuPDF requests citeproc-py python-dotenv tqdm
    echo [Setup] Python libraries installed successfully!
    echo.
) else (
    echo [System] Python dependencies: OK
)

:: 2. CHECK TOGETHER AI API KEY
echo [System] Checking for Together AI API Key...
set TOGETHER_API_KEY=
if exist .env (
    for /f "tokens=1,2 delims==" %%A in (.env) do (
        if "%%A"=="TOGETHER_API_KEY" set TOGETHER_API_KEY=%%B
    )
)
if not "!TOGETHER_API_KEY!"=="" (
    echo [System] Found Together API Key in .env! Using Together AI.
    set SKIP_OLLAMA=1
    goto skip_model_check
) else (
    echo [Setup] Together AI API key not found in .env.
    echo Together AI is recommended as the default fast cloud alternative to local Ollama.
    set /p NEW_TOGETHER_KEY="Enter your Together AI API key (leave blank to use local Ollama instead): "
    if not "!NEW_TOGETHER_KEY!"=="" (
        echo TOGETHER_API_KEY=!NEW_TOGETHER_KEY!>>.env
        echo [System] Saved Together AI Key to .env! Using Together AI.
        set SKIP_OLLAMA=1
        goto skip_model_check
    ) else (
        echo [System] No key entered. Falling back to local Ollama app over API.
        echo.
    )
)

:: 3. CHECK OLLAMA APP INSTALLATION
echo [System] Checking for Ollama...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [WARNING]: Ollama is NOT installed!
    echo The AI fallback for tricky PDFs requires the free Ollama app.
    echo.
    set /p INSTALL_OLLAMA="Would you like to install Ollama now? (Y/N): "
    if /i "!INSTALL_OLLAMA!"=="Y" (
        echo Opening https://ollama.com in your default browser...
        start https://ollama.com
        echo.
        echo Please install Ollama, restart your computer or terminal, and run this file again.
        pause
        exit /b
    ) else (
        echo [System] Skipping Ollama setup. AI fallback will be disabled.
        set SKIP_OLLAMA=1
    )
) else (
    echo [System] Ollama App: OK
    set SKIP_OLLAMA=0
)

:: Skip the model pulling if Ollama isn't installed
if "!SKIP_OLLAMA!"=="1" goto skip_model_check

:: 4. WAKE OLLAMA SERVER & CHECK FOR Llama 3.2:3B MODEL
echo [System] Waking up local Ollama server...
start /b ollama serve >nul 2>&1
timeout /t 2 /nobreak >nul

echo [System] Checking for Llama 3.2:3B model...
ollama list | findstr /i "llama3.2:3b" >nul
if %errorlevel% neq 0 (
    echo.
    echo [Setup] Llama 3.2:3B model not found! Downloading it now...
    echo Note: This is a ~2GB download.
    ollama pull llama3.2:3b
    echo [Setup] Llama 3.2:3B downloaded successfully!
    echo.
) else (
    echo [System] Llama 3.2:3B Model: OK
)
echo.

:skip_model_check

:: 5. CHECK SAVED EMAIL CONFIGURATION
set USER_EMAIL=
if exist .env (
    for /f "tokens=1,2 delims==" %%A in (.env) do (
        if "%%A"=="EMAIL" set USER_EMAIL=%%B
    )
)
if not "!USER_EMAIL!"=="" (
    echo [System] Found saved email: !USER_EMAIL!
) else (
    echo [Setup] Email setup
    echo Crossref and Semantic Scholar APIs require an email address
    echo to use their free "Polite Pool" servers.
    echo Don't worry, your credentials are safe, you won't be hacked.
    echo.
    set /p USER_EMAIL="Please enter your email address: "
    echo EMAIL=!USER_EMAIL!>>.env
    echo.
    echo [System] Email saved to .env! You won't be asked again.
)
echo.

:: 6. EXECUTE THE PIPELINE
echo [System] Starting PDF processing pipeline...
echo.

python main.py --email "!USER_EMAIL!" --dir ./papers --output bibliography.html -v

echo.
echo ===================================================
echo Task Complete! You can now open bibliography.html
echo ===================================================
pause