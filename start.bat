@echo off
cd /d "C:\Users\Aditi\OneDrive\Desktop\semantic-search-newsgroups"
set PYTHONPATH=.;app
echo Starting Trademarkia Semantic Search API...
echo Open your browser: http://localhost:8000/docs
echo.
echo Server will start in 5 seconds...
timeout /t 3 /nobreak >nul
py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause