#!/usr/bin/env python3
"""
Setup script for Face Recognition System
Automates the complete setup process
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {command}")
            retur