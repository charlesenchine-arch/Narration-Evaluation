@echo off
rem Autostart: rating server (8770) + cloudflared tunnel. Idempotent.
setlocal
set PROJ=C:\Users\LiuSh\Desktop\Development\Time-for-narration\narrative-evaluator
set LOGS=%PROJ%\logs
cd /d "%PROJ%"
if not exist "%LOGS%" mkdir "%LOGS%"

rem ---- 1. rating server (8770) ----
netstat -ano | findstr ":8770" | findstr "LISTENING" >nul
if errorlevel 1 (
  powershell -NoProfile -Command "Start-Process -FilePath 'C:\Users\LiuSh\miniconda3\python.exe' -ArgumentList 'scripts/rating_server.py','--data','data/eval_dataset/rating_set.jsonl','--per-rater','15','--host','0.0.0.0','--password','narrate2026','--port','8770','--label-mode','none' -WorkingDirectory '%PROJ%' -WindowStyle Hidden -RedirectStandardOutput '%LOGS%\server_8770.out.log' -RedirectStandardError '%LOGS%\server_8770.err.log'"
)

rem ---- 2. cloudflared tunnel ----
tasklist /FI "IMAGENAME eq cloudflared.exe" 2>nul | findstr /i "cloudflared.exe" >nul
if errorlevel 1 (
  powershell -NoProfile -Command "Start-Process -FilePath 'C:\Users\LiuSh\AppData\Roaming\npm\cloudflared.cmd' -ArgumentList 'tunnel','--url','http://localhost:8770' -WorkingDirectory '%PROJ%' -WindowStyle Hidden -RedirectStandardOutput '%LOGS%\cf8770.out.log' -RedirectStandardError '%LOGS%\cf8770.err.log'"
)

rem ---- 3. wait for tunnel, write new URL to CURRENT_URL.txt ----
set /a tries=0
:waiturl
set /a tries+=1
findstr /i "trycloudflare.com" "%LOGS%\cf8770.err.log" >nul 2>nul
if not errorlevel 1 goto goturl
if %tries% geq 8 goto done
ping -n 4 127.0.0.1 >nul
goto waiturl
:goturl
powershell -NoProfile -Command "$c=Get-Content '%LOGS%\cf8770.err.log' -Raw; $m=[regex]::Match($c,'https://[a-zA-Z0-9.\-]+\.trycloudflare\.com'); if($m.Success){$m.Value|Out-File '%LOGS%\CURRENT_URL.txt' -Encoding utf8}"
:done
endlocal
exit /b 0
