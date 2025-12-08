"""
Phase 2 機能のユニットテスト
- 状態判定
- エラーループ検知
"""

import sys
from collections import deque
from enum import Enum


# claude_coding_agent.py から必要な部分をインポート
class AgentState(Enum):
    """エージェントの状態"""
    UNKNOWN = "unknown"
    PROCEED_DIALOG = "proceed_dialog"
    ALLOW_EDITS_DIALOG = "allow_edits"
    TESTS_RUNNING = "tests_running"
    TESTS_FAILED = "tests_failed"
    WAITING_FOR_INPUT = "waiting"
    NO_ACTION_NEEDED = "no_action"


def determine_state(output_text: str, actions: list) -> AgentState:
    """モデル出力とアクションから現在の状態を判定。"""
    if not actions:
        if any(keyword in output_text.lower() for keyword in ["running", "executing", "in progress"]):
            return AgentState.TESTS_RUNNING
        return AgentState.NO_ACTION_NEEDED

    for action in actions:
        if action[0] == "write":
            text_content = action[1].lower()
            if "失敗" in text_content or "エラー" in text_content or "修正" in text_content:
                return AgentState.TESTS_FAILED
            if "いいね" in text_content or "素晴らしい" in text_content or "進めて" in text_content:
                return AgentState.WAITING_FOR_INPUT

    output_lower = output_text.lower()
    if "proceed" in output_lower or "1. yes" in output_lower:
        return AgentState.PROCEED_DIALOG
    if "allow all edits" in output_lower or "alt+m" in output_lower:
        return AgentState.ALLOW_EDITS_DIALOG

    return AgentState.UNKNOWN


def get_action_summary(actions: list) -> str:
    """アクションのサマリー文字列を生成"""
    if not actions:
        return "NO_ACTION"
    summary_parts = []
    for action in actions:
        if action[0] == "click":
            summary_parts.append(f"click({action[1]},{action[2]})")
        elif action[0] == "write":
            text = action[1][:20] + "..." if len(action[1]) > 20 else action[1]
            summary_parts.append(f"write({repr(text)})")
        elif action[0] == "press":
            summary_parts.append(f"press({action[1]})")
    return "|".join(summary_parts)


def check_loop_detection(state_history: deque, max_repetitions: int = 5) -> tuple[bool, int]:
    """エラーループを検知。"""
    if len(state_history) < max_repetitions:
        return False, 0

    latest_state, latest_hash, latest_action = state_history[-1]
    repetition_count = 0
    for state, screen_hash, action_summary in reversed(state_history):
        if state == latest_state and screen_hash == latest_hash and action_summary == latest_action:
            repetition_count += 1
        else:
            break

    is_loop = repetition_count >= max_repetitions
    return is_loop, repetition_count


# ========== テスト ==========

def test_determine_state():
    """状態判定のテスト"""
    print("=== Test: determine_state ===")

    # Test 1: NO_ACTION (実行中)
    state1 = determine_state("Tests are running...", [])
    assert state1 == AgentState.TESTS_RUNNING, f"Test 1 failed: {state1}"
    print("[OK] Test 1: TESTS_RUNNING")

    # Test 2: NO_ACTION (何もしない)
    state2 = determine_state("Everything looks good.", [])
    assert state2 == AgentState.NO_ACTION_NEEDED, f"Test 2 failed: {state2}"
    print("[OK] Test 2: NO_ACTION_NEEDED")

    # Test 3: PROCEED_DIALOG
    actions3 = [("click", 100, 200)]
    state3 = determine_state("Click on 1. Yes to proceed", actions3)
    assert state3 == AgentState.PROCEED_DIALOG, f"Test 3 failed: {state3}"
    print("[OK] Test 3: PROCEED_DIALOG")

    # Test 4: ALLOW_EDITS_DIALOG
    actions4 = [("click", 100, 200)]
    state4 = determine_state("Click on allow all edits during this session", actions4)
    assert state4 == AgentState.ALLOW_EDITS_DIALOG, f"Test 4 failed: {state4}"
    print("[OK] Test 4: ALLOW_EDITS_DIALOG")

    # Test 5: TESTS_FAILED
    actions5 = [("write", "テストが失敗しています。エラーを修正してください。"), ("press", "enter")]
    state5 = determine_state("Writing error message", actions5)
    assert state5 == AgentState.TESTS_FAILED, f"Test 5 failed: {state5}"
    print("[OK] Test 5: TESTS_FAILED")

    # Test 6: WAITING_FOR_INPUT
    actions6 = [("write", "いいね！素晴らしい次もばんばん進めて。"), ("press", "enter")]
    state6 = determine_state("Writing encouragement", actions6)
    assert state6 == AgentState.WAITING_FOR_INPUT, f"Test 6 failed: {state6}"
    print("[OK] Test 6: WAITING_FOR_INPUT")

    # Test 7: UNKNOWN
    actions7 = [("click", 500, 600)]
    state7 = determine_state("Some random text", actions7)
    assert state7 == AgentState.UNKNOWN, f"Test 7 failed: {state7}"
    print("[OK] Test 7: UNKNOWN")

    print()


def test_action_summary():
    """アクションサマリーのテスト"""
    print("=== Test: get_action_summary ===")

    # Test 1: NO_ACTION
    summary1 = get_action_summary([])
    assert summary1 == "NO_ACTION", f"Test 1 failed: {summary1}"
    print("[OK] Test 1: NO_ACTION")

    # Test 2: Single click
    summary2 = get_action_summary([("click", 100, 200)])
    assert summary2 == "click(100,200)", f"Test 2 failed: {summary2}"
    print("[OK] Test 2: Single click")

    # Test 3: Click + Write + Press
    summary3 = get_action_summary([
        ("click", 100, 200),
        ("write", "テストが失敗しています。"),
        ("press", "enter")
    ])
    assert "click(100,200)" in summary3, f"Test 3 failed: {summary3}"
    assert "write(" in summary3, f"Test 3 failed: {summary3}"
    assert "press(enter)" in summary3, f"Test 3 failed: {summary3}"
    print("[OK] Test 3: Click + Write + Press")

    # Test 4: Long text truncation
    long_text = "これは非常に長いテキストメッセージです。20文字を超えているので切り捨てられるはずです。"
    summary4 = get_action_summary([("write", long_text)])
    assert "..." in summary4, f"Test 4 failed: {summary4}"
    print("[OK] Test 4: Long text truncation")

    print()


def test_loop_detection():
    """ループ検知のテスト"""
    print("=== Test: check_loop_detection ===")

    # Test 1: ループなし（履歴が少ない）
    history1 = deque(maxlen=10)
    history1.append((AgentState.PROCEED_DIALOG, "hash1", "click(100,200)"))
    history1.append((AgentState.PROCEED_DIALOG, "hash2", "click(100,200)"))
    is_loop1, count1 = check_loop_detection(history1, max_repetitions=5)
    assert not is_loop1, f"Test 1 failed: {is_loop1}"
    print("[OK] Test 1: No loop (insufficient history)")

    # Test 2: ループなし（状態が異なる）
    history2 = deque(maxlen=10)
    for i in range(6):
        state = AgentState.PROCEED_DIALOG if i % 2 == 0 else AgentState.WAITING_FOR_INPUT
        history2.append((state, f"hash{i}", "click(100,200)"))
    is_loop2, count2 = check_loop_detection(history2, max_repetitions=5)
    assert not is_loop2, f"Test 2 failed: {is_loop2}"
    print("[OK] Test 2: No loop (different states)")

    # Test 3: ループ検知（同じ状態が5回連続）
    history3 = deque(maxlen=10)
    for i in range(5):
        history3.append((AgentState.TESTS_FAILED, "hash_same", "write('エラー')|press(enter)"))
    is_loop3, count3 = check_loop_detection(history3, max_repetitions=5)
    assert is_loop3, f"Test 3 failed: is_loop={is_loop3}, count={count3}"
    assert count3 == 5, f"Test 3 failed: count={count3}"
    print("[OK] Test 3: Loop detected (5 repetitions)")

    # Test 4: ループ検知（6回連続）
    history4 = deque(maxlen=10)
    for i in range(6):
        history4.append((AgentState.TESTS_FAILED, "hash_same", "write('エラー')|press(enter)"))
    is_loop4, count4 = check_loop_detection(history4, max_repetitions=5)
    assert is_loop4, f"Test 4 failed: is_loop={is_loop4}"
    assert count4 == 6, f"Test 4 failed: count={count4}"
    print("[OK] Test 4: Loop detected (6 repetitions)")

    # Test 5: ループなし（途中で状態が変わる）
    history5 = deque(maxlen=10)
    for i in range(4):
        history5.append((AgentState.TESTS_FAILED, "hash1", "write('エラー')|press(enter)"))
    history5.append((AgentState.WAITING_FOR_INPUT, "hash2", "write('いいね')|press(enter)"))
    is_loop5, count5 = check_loop_detection(history5, max_repetitions=5)
    assert not is_loop5, f"Test 5 failed: {is_loop5}"
    print("[OK] Test 5: No loop (state changed)")

    print()


def main():
    """全テストを実行"""
    print("\n" + "="*60)
    print("Phase 2 機能テスト")
    print("="*60 + "\n")

    test_determine_state()
    test_action_summary()
    test_loop_detection()

    print("="*60)
    print("*** All Phase 2 tests passed! ***")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
