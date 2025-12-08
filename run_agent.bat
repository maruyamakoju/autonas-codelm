@echo off
REM Claude Coding Autopilot Agent - 起動スクリプト

echo ========================================
echo Claude Coding Autopilot Agent
echo ========================================
echo.

REM 仮想環境が存在するかチェック
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] 仮想環境が見つかりません。
    echo セットアップを実行してください:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM 仮想環境をアクティベート
echo 仮想環境をアクティベート中...
call .venv\Scripts\activate.bat

REM エージェントを起動
echo.
echo エージェントを起動中...
echo Ctrl+C または マウスを画面左上(0,0)に移動で停止できます。
echo.
python claude_coding_agent.py

REM エラーチェック
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] エージェントがエラーで終了しました。
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo エージェントを正常に終了しました。
pause
