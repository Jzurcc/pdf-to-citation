@echo off
title Auto-APA Citation Generator
setlocal enabledelayedexpansion

echo ===================================================
echo       Auto-APA Citation Generator (ArXiv)
echo ===================================================
echo.

:: 1. CHECK PYTHON DEPENDENCIES
echo [System] Checking Python dependencies...
python -c "import fitz, requests, citeproc" >nul 2>&1
if %errorlevel% neq 0 (
    echo [Setup] Missing required Python libraries! Installing them now...
    pip install PyMuPDF requests citeproc-py
    echo [Setup] Python libraries installed successfully!
    echo.
) else (
    echo [System] Python dependencies: OK
)

:: 2. CHECK OLLAMA APP INSTALLATION
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

:: 3. WAKE OLLAMA SERVER & CHECK FOR Llama 3.2:3B MODEL
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

:: 4. CHECK SAVED EMAIL CONFIGURATION
if exist email_config.txt (
    set /p USER_EMAIL=<email_config.txt
    echo [System] Found saved email: !USER_EMAIL!
) else (
    echo [Setup] First time setup!
    echo Crossref and Semantic Scholar APIs require an email address 
    echo to use their free "Polite Pool" servers.
    echo Don't worry, your credentials are safe, you won't be hacked.
    echo.
    set /p USER_EMAIL="Please enter your email address: "
    echo !USER_EMAIL!>email_config.txt
    echo.
    echo [System] Email saved to email_config.txt! You won't be asked again.
)
echo.

:: 5. EXECUTE THE PIPELINE
echo [System] Starting PDF processing pipeline...
echo.

python main.py --email "!USER_EMAIL!" --dir ./papers --output bibliography.html -v

echo.
echo ===================================================
echo Task Complete! You can now open bibliography.html
echo ===================================================
pause