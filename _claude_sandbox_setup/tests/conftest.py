"""
Pytest configuration for sandbox verification tests.

Adds the project root to sys.path so tests can find the setup scripts.
"""
import os
import sys

# Project root is two levels up from this file:
# project_root/_claude_sandbox_setup/tests/conftest.py
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SETUP_DIR = os.path.dirname(TESTS_DIR)
PROJECT_ROOT = os.path.dirname(SETUP_DIR)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SETUP_DIR)
