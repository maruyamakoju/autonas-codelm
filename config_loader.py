"""
Configuration loader for Claude Coding Autopilot Agent
Loads and validates config.yaml
"""

import os
import yaml
from typing import Dict, Any


DEFAULT_CONFIG = {
    "version": 1,
    "loop_protection": {
        "max_same_state_repetitions": 5
    },
    "capture": {
        "use_crop": False,
        "crop_region": {
            "x": 0,
            "y": 0,
            "width": 1920,
            "height": 1080
        },
        "save_debug_screenshot": True,
        "debug_screenshot_path": "screen.png"
    },
    "messages": {
        "on_test_failed": "テストが失敗しています。表示されている主なエラーを修正して、テストを再実行してください。",
        "on_test_success": "いいね！素晴らしい次もばんばん進めて。"
    },
    "states": {
        "tests_running_keywords": ["Running", "running", "in progress", "executing"],
        "tests_failed_keywords": ["FAILED", "failed", "Error", "error", "Exception"],
        "proceed_dialog_keywords": ["Do you want to proceed?", "1. Yes"],
        "allow_edits_dialog_keywords": ["Do you want to make this edit", "allow all edits", "alt+m"]
    },
    "logging": {
        "enable_json_log": True,
        "json_log_path": "logs/steps.jsonl",
        "verbose": True,
        "log_dir": "logs"
    },
    "model": {
        "model_id": "xlangai/OpenCUA-32B",
        "max_new_tokens": 256,
        "sleep_between_steps": 5.0
    },
    "safety": {
        "enable_failsafe": True,
        "max_steps": 1000000,
        "dry_run": False
    }
}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Falls back to default config if file doesn't exist.
    """
    if not os.path.exists(config_path):
        print(f"[CONFIG] Config file not found: {config_path}")
        print("[CONFIG] Using default configuration")
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        print(f"[CONFIG] Loaded configuration from: {config_path}")

        # Merge with defaults (in case some keys are missing)
        merged_config = _merge_configs(DEFAULT_CONFIG, config)
        return merged_config

    except Exception as e:
        print(f"[CONFIG] Error loading config file: {e}")
        print("[CONFIG] Using default configuration")
        return DEFAULT_CONFIG.copy()


def _merge_configs(default: Dict, custom: Dict) -> Dict:
    """
    Recursively merge custom config with default config.
    Custom values override defaults.
    """
    merged = default.copy()

    for key, value in custom.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration.
    Returns True if valid, False otherwise.
    """
    try:
        # Check version
        if config.get("version") != 1:
            print(f"[CONFIG] Warning: Unknown config version: {config.get('version')}")

        # Check required keys
        required_keys = ["loop_protection", "capture", "messages", "states", "logging", "model", "safety"]
        for key in required_keys:
            if key not in config:
                print(f"[CONFIG] Error: Missing required key: {key}")
                return False

        # Validate crop region if use_crop is enabled
        if config["capture"]["use_crop"]:
            crop = config["capture"]["crop_region"]
            if not all(k in crop for k in ["x", "y", "width", "height"]):
                print("[CONFIG] Error: Incomplete crop_region specification")
                return False
            if crop["width"] <= 0 or crop["height"] <= 0:
                print("[CONFIG] Error: Invalid crop_region dimensions")
                return False

        # Validate loop protection
        max_reps = config["loop_protection"]["max_same_state_repetitions"]
        if not isinstance(max_reps, int) or max_reps < 1:
            print("[CONFIG] Error: Invalid max_same_state_repetitions")
            return False

        print("[CONFIG] Configuration validation passed")
        return True

    except Exception as e:
        print(f"[CONFIG] Validation error: {e}")
        return False


def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    """Save configuration to YAML file."""
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"[CONFIG] Configuration saved to: {config_path}")
        return True
    except Exception as e:
        print(f"[CONFIG] Error saving config: {e}")
        return False


if __name__ == "__main__":
    # Test config loader
    config = load_config()
    if validate_config(config):
        print("\n[CONFIG] Test: Configuration loaded and validated successfully")
        print(f"[CONFIG] Loop protection threshold: {config['loop_protection']['max_same_state_repetitions']}")
        print(f"[CONFIG] Use crop: {config['capture']['use_crop']}")
        print(f"[CONFIG] Model: {config['model']['model_id']}")
        print(f"[CONFIG] JSON logging: {config['logging']['enable_json_log']}")
    else:
        print("\n[CONFIG] Test: Configuration validation failed")
