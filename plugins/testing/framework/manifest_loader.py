#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tutorial manifest loader and validator"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class TestConfig:
    """Configuration for a specific test type"""
    enabled: bool = True
    command: str = ""
    timeout: int = 300
    requires_services: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TestConfig':
        return cls(
            enabled=data.get('enabled', True),
            command=data.get('command', ''),
            timeout=data.get('timeout', 300),
            requires_services=data.get('requires_services', [])
        )


@dataclass
class TutorialManifest:
    """Tutorial manifest data structure (simplified)"""
    name: str
    description: str
    path: Path
    tests: Dict[str, TestConfig] = field(default_factory=dict)
    enabled: bool = True

    @classmethod
    def load_from_file(cls, manifest_path: Path) -> 'TutorialManifest':
        """Load tutorial manifest from YAML file"""
        with open(manifest_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse test configurations
        tests = {}
        if 'tests' in data:
            for test_type, test_data in data['tests'].items():
                tests[test_type] = TestConfig.from_dict(test_data)

        return cls(
            name=data['name'],
            description=data.get('description', ''),
            path=manifest_path.parent,
            tests=tests,
            enabled=data.get('enabled', True)
        )

    def get_test_types(self) -> List[str]:
        """Get list of enabled test types"""
        return [test_type for test_type, config in self.tests.items() if config.enabled]

    def validate(self) -> List[str]:
        """Validate manifest and return list of errors"""
        errors = []

        if not self.name:
            errors.append("Tutorial name is required")

        if not self.version:
            errors.append("Tutorial version is required")

        if not self.description:
            errors.append("Tutorial description is required")

        for test_type, config in self.tests.items():
            if config.enabled and not config.command:
                errors.append(f"Test '{test_type}' is enabled but has no command")

        return errors


class TutorialRegistry:
    """Tutorial registry manager - auto-discovers tutorials by scanning subfolders"""

    def __init__(self, tutorials_dir: Path):
        self.tutorials_dir = tutorials_dir
        self.tutorials: Dict[str, TutorialManifest] = {}
        self._discover_tutorials()

    def _discover_tutorials(self):
        """Discover tutorials by scanning subfolders for tutorial.yaml"""
        if not self.tutorials_dir.exists():
            raise FileNotFoundError(f"Tutorials directory not found: {self.tutorials_dir}")

        for subdir in sorted(self.tutorials_dir.iterdir()):
            if not subdir.is_dir():
                continue

            manifest_path = subdir / 'tutorial.yaml'
            if not manifest_path.exists():
                continue

            try:
                manifest = TutorialManifest.load_from_file(manifest_path)
                self.tutorials[manifest.name] = manifest
            except Exception as e:
                print(f"Error loading manifest for {subdir.name}: {e}")

    def get_tutorial(self, name: str) -> Optional[TutorialManifest]:
        """Get tutorial manifest by name"""
        return self.tutorials.get(name)

    def get_all_tutorials(self, enabled_only: bool = True) -> List[TutorialManifest]:
        """Get all tutorial manifests"""
        tutorials = list(self.tutorials.values())

        if enabled_only:
            tutorials = [t for t in tutorials if t.enabled]

        return tutorials

    def get_tutorials_by_category(self, category: str) -> List[TutorialManifest]:
        """Get tutorials by category"""
        return [t for t in self.tutorials.values() if t.category == category]

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all tutorial manifests"""
        results = {}
        for name, tutorial in self.tutorials.items():
            errors = tutorial.validate()
            if errors:
                results[name] = errors
        return results

