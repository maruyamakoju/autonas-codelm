"""
アクション抽出ロジックのユニットテスト
"""

import re


# claude_coding_agent.py からコピー
ACTION_RE = re.compile(
    r"pyautogui\.(click|write|press)\("
    r"(?:x=(\d+),\s*y=(\d+)|"  # click(x=..., y=...)
    r"['\"]([^'\"]*)['\"]|"    # write("...") or press("...")
    r"([a-z]+))"               # press(enter) など
    r"\)"
)


def extract_actions(text: str):
    """
    モデル出力から pyautogui アクションを抽出。
    戻り値: [("click", x, y), ("write", text), ("press", key), ...]
    """
    # NO_ACTION 判定
    if "NO_ACTION" in text and "pyautogui" not in text:
        return []

    # ```python ... ``` ブロックを抽出
    code_blocks = re.findall(r"```python\s+(.*?)\s+```", text, re.DOTALL)
    if code_blocks:
        code = "\n".join(code_blocks)
    else:
        code = text

    actions = []
    for match in ACTION_RE.finditer(code):
        action_type = match.group(1)

        if action_type == "click":
            x = match.group(2)
            y = match.group(3)
            if x and y:
                actions.append(("click", int(x), int(y)))

        elif action_type == "write":
            text_content = match.group(4)
            if text_content:
                actions.append(("write", text_content))

        elif action_type == "press":
            key = match.group(4) or match.group(5)
            if key:
                actions.append(("press", key))

    return actions


def test_extract_actions():
    """テストケース"""

    # Test 1: NO_ACTION
    text1 = "NO_ACTION"
    result1 = extract_actions(text1)
    assert result1 == [], f"Test 1 failed: {result1}"
    print("[OK] Test 1 passed: NO_ACTION")

    # Test 2: Single click
    text2 = """```python
pyautogui.click(x=1234, y=567)
```"""
    result2 = extract_actions(text2)
    assert result2 == [("click", 1234, 567)], f"Test 2 failed: {result2}"
    print("[OK] Test 2 passed: Single click")

    # Test 3: Click + Write + Press
    text3 = """```python
pyautogui.click(x=100, y=200)
pyautogui.write("テストが失敗しています。表示されている主なエラーを修正して、テストを再実行してください。")
pyautogui.press("enter")
```"""
    result3 = extract_actions(text3)
    assert result3 == [
        ("click", 100, 200),
        ("write", "テストが失敗しています。表示されている主なエラーを修正して、テストを再実行してください。"),
        ("press", "enter")
    ], f"Test 3 failed: {result3}"
    print("[OK] Test 3 passed: Click + Write + Press")

    # Test 4: Multiple clicks
    text4 = """```python
pyautogui.click(x=10, y=20)
pyautogui.click(x=30, y=40)
```"""
    result4 = extract_actions(text4)
    assert result4 == [("click", 10, 20), ("click", 30, 40)], f"Test 4 failed: {result4}"
    print("[OK] Test 4 passed: Multiple clicks")

    # Test 5: Write with single quotes
    text5 = """```python
pyautogui.write('Hello World')
```"""
    result5 = extract_actions(text5)
    assert result5 == [("write", "Hello World")], f"Test 5 failed: {result5}"
    print("[OK] Test 5 passed: Write with single quotes")

    # Test 6: Press without quotes
    text6 = """```python
pyautogui.press(enter)
```"""
    result6 = extract_actions(text6)
    assert result6 == [("press", "enter")], f"Test 6 failed: {result6}"
    print("[OK] Test 6 passed: Press without quotes")

    # Test 7: NO_ACTION with explanation
    text7 = "NO_ACTION - tests are still running"
    result7 = extract_actions(text7)
    assert result7 == [], f"Test 7 failed: {result7}"
    print("[OK] Test 7 passed: NO_ACTION with explanation")

    # Test 8: Code without markdown block
    text8 = "pyautogui.click(x=500, y=600)"
    result8 = extract_actions(text8)
    assert result8 == [("click", 500, 600)], f"Test 8 failed: {result8}"
    print("[OK] Test 8 passed: Code without markdown block")

    print("\n*** All tests passed! ***")


if __name__ == "__main__":
    test_extract_actions()
