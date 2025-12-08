@echo off
REM Claude Coding Autopilot Agent - セットアップスクリプト

echo ========================================
echo Claude Coding Autopilot Agent - Setup
echo ========================================
echo.

REM Python がインストールされているかチェック
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python が見つかりません。
    echo Python 3.10 以降をインストールしてください。
    pause
    exit /b 1
)

echo Python が見つかりました:
python --version
echo.

REM 仮想環境を作成
if exist ".venv" (
    echo 仮想環境は既に存在します。
) else (
    echo 仮想環境を作成中...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] 仮想環境の作成に失敗しました。
        pause
        exit /b 1
    )
    echo 仮想環境を作成しました。
)
echo.

REM 仮想環境をアクティベート
echo 仮想環境をアクティベート中...
call .venv\Scripts\activate.bat

REM 依存パッケージをインストール
echo.
echo 依存パッケージをインストール中...
echo これには数分かかる場合があります。
echo.
pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] パッケージのインストールに失敗しました。
    pause
    exit /b 1
)

echo.
echo ========================================
echo セットアップが完了しました！
echo ========================================
echo.
echo 次のコマンドでエージェントを起動できます:
echo   run_agent.bat
echo.
echo または:
echo   .venv\Scripts\activate
echo   python claude_coding_agent.py
echo.
pause
