#!/usr/bin/env python3
"""
クロップ領域の調整ヘルパー

使い方:
1. Claude Code ブラウザを配置
2. このスクリプトを実行
3. 画面をクリックして座標を確認
4. config.yaml の crop_region を調整
"""

import pyautogui
import time

print("=" * 60)
print("クロップ領域調整ヘルパー")
print("=" * 60)
print()
print("Claude Code ブラウザの四隅をクリックして座標を確認してください。")
print()
print("左上隅をクリック...")
time.sleep(2)

try:
    # マウス位置を5秒間監視
    for i in range(5):
        x, y = pyautogui.position()
        print(f"現在のマウス位置: x={x}, y={y}")
        time.sleep(1)

    print()
    print("使い方:")
    print("1. このスクリプトを実行")
    print("2. Claude Code ブラウザの左上隅にマウスを移動")
    print("3. 表示される座標をメモ → config.yaml の x, y に設定")
    print("4. 右下隅にマウスを移動")
    print("5. width = (右下のx - 左上のx), height = (右下のy - 左上のy)")

except KeyboardInterrupt:
    print("\n中断されました。")
