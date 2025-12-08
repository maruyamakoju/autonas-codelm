#!/usr/bin/env python3
"""
dry_run モードの切り替えスクリプト

使い方:
    python toggle_dryrun.py

現在のモードを表示し、反転させます。
"""

import yaml

CONFIG_PATH = "config.yaml"

def toggle_dry_run():
    # 設定を読み込み
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    current = config["safety"]["dry_run"]
    new_value = not current

    print(f"現在の dry_run: {current}")
    print(f"新しい dry_run: {new_value}")

    # 確認
    if new_value is False:
        response = input("\n⚠️  本番モード（実際にクリック・入力を実行）に切り替えますか？ [y/N]: ")
        if response.lower() != 'y':
            print("キャンセルしました。")
            return

    # 更新
    config["safety"]["dry_run"] = new_value

    # 保存
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n✅ config.yaml を更新しました: dry_run = {new_value}")

    if new_value is False:
        print("\n⚠️  本番モードに切り替わりました！")
        print("   - 実際にマウス・キーボード操作が実行されます")
        print("   - Claude Code が開いていることを確認してください")
        print("   - 緊急停止: Ctrl+C または マウスを左上(0,0)に移動")

if __name__ == "__main__":
    toggle_dry_run()
