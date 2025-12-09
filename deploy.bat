@echo off
echo Starting servers...

echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated.

start "Backend" cmd /k "python run_server.py"
timeout /t 2 /nobreak > nul
start "Frontend" cmd /k "cd dfrontend && python -m http.server 3000"

echo Backend: http://localhost:5002
echo Frontend: http://localhost:3000
pause