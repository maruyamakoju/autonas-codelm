# Test Project for Claude Coding Autopilot Agent

This is a minimal Python project for testing the agent.

## Files

- `calculator.py` - Simple calculator module
- `test_calculator.py` - Tests (includes 1 intentionally failing test)

## Usage

```bash
# Run tests
pytest test_calculator.py -v

# Expected: 5 passed, 1 failed (test_add_intentional_fail)
```

## Testing the Agent

1. Open this project in Claude Code
2. Ask Claude to "Run the tests"
3. Claude will see the failing test
4. The agent should:
   - Click "Yes" on proceed dialogs
   - Allow edits when prompted
   - Send error fix instructions when tests fail
   - Send encouragement when tests pass
