"""
環境チェックスクリプト
Claude Coding Autopilot Agent の実行環境を確認
"""

import os
import sys
import platform


def check_python_version():
    """Python バージョンチェック"""
    print("=" * 60)
    print("Python バージョンチェック")
    print("=" * 60)
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 10:
        print("[OK] Python 3.10以降")
        return True
    else:
        print("[WARNING] Python 3.10以降を推奨します")
        return False


def check_cuda():
    """CUDA / GPU チェック"""
    print("\n" + "=" * 60)
    print("CUDA / GPU チェック")
    print("=" * 60)

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            print(f"Device capability: {torch.cuda.get_device_capability(0)}")
            print("[OK] GPU利用可能")
            return True
        else:
            print("[WARNING] CUDAが利用できません（CPUモードで動作）")
            return False
    except ImportError:
        print("[ERROR] PyTorchがインストールされていません")
        print("pip install torch をしてください")
        return False


def check_dependencies():
    """依存パッケージチェック"""
    print("\n" + "=" * 60)
    print("依存パッケージチェック")
    print("=" * 60)

    required_packages = [
        ("transformers", "transformers"),
        ("PIL", "pillow"),
        ("mss", "mss"),
        ("pyautogui", "pyautogui"),
        ("pyperclip", "pyperclip"),
        ("yaml", "pyyaml")
    ]

    all_ok = True
    for import_name, package_name in required_packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {package_name}: {version}")
        except ImportError:
            print(f"[ERROR] {package_name} がインストールされていません")
            print(f"  → pip install {package_name}")
            all_ok = False

    return all_ok


def check_display():
    """ディスプレイ設定チェック"""
    print("\n" + "=" * 60)
    print("ディスプレイ設定チェック")
    print("=" * 60)

    try:
        import mss
        with mss.mss() as sct:
            monitors = sct.monitors
            print(f"モニタ数: {len(monitors) - 1}")  # monitors[0]は全画面

            primary = monitors[1]
            print(f"プライマリモニタ: {primary['width']}x{primary['height']}")
            print(f"  位置: ({primary['left']}, {primary['top']})")

            if primary['width'] == 1920 and primary['height'] == 1080:
                print("[OK] フルHD (1920x1080)")
            else:
                print(f"[INFO] フルHD以外の解像度です")

        print("\n[推奨設定]")
        print("- Windowsスケーリング: 100%")
        print("- マルチモニタ: Claude Codeをプライマリモニタに配置")

        return True
    except Exception as e:
        print(f"[ERROR] ディスプレイチェックに失敗: {e}")
        return False


def check_config_file():
    """config.yaml チェック"""
    print("\n" + "=" * 60)
    print("設定ファイルチェック")
    print("=" * 60)

    if not os.path.exists("config.yaml"):
        print("[WARNING] config.yaml が見つかりません")
        print("デフォルト設定で動作します")
        return False

    try:
        from config_loader import load_config, validate_config
        config = load_config("config.yaml")
        if validate_config(config):
            print("[OK] config.yaml は正常です")
            print(f"  - ループ検知閾値: {config['loop_protection']['max_same_state_repetitions']}")
            print(f"  - クロップ使用: {config['capture']['use_crop']}")
            print(f"  - JSONログ: {config['logging']['enable_json_log']}")
            print(f"  - Dry-runモード: {config['safety']['dry_run']}")
            return True
        else:
            print("[ERROR] config.yamlの検証に失敗しました")
            return False
    except Exception as e:
        print(f"[ERROR] config.yamlの読み込みに失敗: {e}")
        return False


def check_test_project():
    """テストプロジェクトチェック"""
    print("\n" + "=" * 60)
    print("テストプロジェクトチェック")
    print("=" * 60)

    if not os.path.exists("test_project"):
        print("[INFO] test_project が見つかりません")
        print("テスト用プロジェクトがない場合は作成を推奨します")
        return False

    required_files = [
        "test_project/calculator.py",
        "test_project/test_calculator.py",
        "test_project/README.md"
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path}")
        else:
            print(f"[MISSING] {file_path}")
            all_exist = False

    if all_exist:
        print("\n[OK] テストプロジェクトは完全です")
    else:
        print("\n[INFO] 一部のファイルが不足しています")

    return all_exist


def main():
    """環境チェック実行"""
    print("\n" + "=" * 60)
    print("Claude Coding Autopilot Agent - 環境チェック")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"プロセッサ: {platform.processor()}")
    print()

    results = []
    results.append(("Python バージョン", check_python_version()))
    results.append(("CUDA / GPU", check_cuda()))
    results.append(("依存パッケージ", check_dependencies()))
    results.append(("ディスプレイ設定", check_display()))
    results.append(("config.yaml", check_config_file()))
    results.append(("テストプロジェクト", check_test_project()))

    # サマリー
    print("\n" + "=" * 60)
    print("チェック結果サマリー")
    print("=" * 60)

    for name, result in results:
        status = "[OK]" if result else "[WARNING/ERROR]"
        print(f"{status} {name}")

    all_ok = all(result for _, result in results)
    print()

    if all_ok:
        print("全てのチェックに合格しました！")
        print("エージェントを実行する準備ができています。")
        print("\n実行方法:")
        print("  python claude_coding_agent.py")
        print("または:")
        print("  run_agent.bat")
    else:
        print("一部のチェックで警告/エラーがありました。")
        print("上記の警告を確認して、必要に応じて対処してください。")
        print("\nエージェントは動作する可能性がありますが、")
        print("最適なパフォーマンスのために推奨設定を確認してください。")


if __name__ == "__main__":
    main()
