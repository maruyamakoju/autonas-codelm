@echo off
REM OpenCUA推奨バージョンへの依存パッケージ再インストール

echo ========================================
echo OpenCUA Dependencies Fix
echo ========================================
echo.

REM 仮想環境が存在するかチェック
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] 仮想環境が見つかりません。
    echo setup.bat を先に実行してください。
    pause
    exit /b 1
)

REM 仮想環境をアクティベート
echo 仮想環境をアクティベート中...
call .venv\Scripts\activate.bat

echo.
echo OpenCUA公式推奨バージョンをインストール中...
echo これには数分かかる場合があります。
echo.

REM OpenCUA公式推奨バージョンをインストール
pip install ^
  "transformers==4.53.0" ^
  "torch==2.8.0" ^
  "pillow==11.3.0" ^
  "tiktoken==0.11.0" ^
  "blobfile==3.0.0" ^
  "accelerate==1.10.0"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] パッケージのインストールに失敗しました。
    pause
    exit /b 1
)

echo.
echo その他の依存パッケージをインストール中...
pip install mss pyautogui pyperclip pyyaml

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] パッケージのインストールに失敗しました。
    pause
    exit /b 1
)

echo.
echo ========================================
echo インストール完了！
echo ========================================
echo.
echo インストールされたバージョンを確認:
python -m pip show transformers torch pillow tiktoken accelerate
echo.
echo 次のステップ:
echo   1. config.yaml でdry_run: true になっているか確認
echo   2. python claude_coding_agent.py で起動
echo.
pause
