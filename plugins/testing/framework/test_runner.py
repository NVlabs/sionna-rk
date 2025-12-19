#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Centralized tutorial test runner"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from manifest_loader import TutorialRegistry, TutorialManifest


class TestResult(Enum):
    """Test result status"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TestRunResult:
    """Result of a test run"""
    tutorial_name: str
    test_type: str
    result: TestResult
    duration: float
    output: str = ""
    error: str = ""

    def is_success(self) -> bool:
        return self.result == TestResult.PASSED

    def __str__(self) -> str:
        emoji = {
            TestResult.PASSED: "✅",
            TestResult.FAILED: "❌",
            TestResult.SKIPPED: "⏭️",
            TestResult.TIMEOUT: "⏱️",
            TestResult.ERROR: "💥"
        }
        return f"{emoji[self.result]} {self.tutorial_name}/{self.test_type}: {self.result.value} ({self.duration:.2f}s)"


class TutorialTestRunner:
    """Test runner for tutorials"""

    def __init__(self, tutorials_dir: Path, verbose: bool = False, plan_postfix: str = ""):
        self.registry = TutorialRegistry(tutorials_dir)
        self.verbose = verbose
        self.plan_postfix = plan_postfix
        self.results: List[TestRunResult] = []

    def run_command(self, command: str, cwd: Path, timeout: int) -> TestRunResult:
        """Run a test command"""
        start_time = time.time()

        try:
            # Set up environment with PLAN_POSTFIX if specified
            env = os.environ.copy()
            if self.plan_postfix:
                env['PLAN_POSTFIX'] = self.plan_postfix
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                return TestRunResult(
                    tutorial_name="",
                    test_type="",
                    result=TestResult.PASSED,
                    duration=duration,
                    output=result.stdout
                )
            else:
                return TestRunResult(
                    tutorial_name="",
                    test_type="",
                    result=TestResult.FAILED,
                    duration=duration,
                    output=result.stdout,
                    error=result.stderr
                )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestRunResult(
                tutorial_name="",
                test_type="",
                result=TestResult.TIMEOUT,
                duration=duration,
                error=f"Test timed out after {timeout}s"
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestRunResult(
                tutorial_name="",
                test_type="",
                result=TestResult.ERROR,
                duration=duration,
                error=str(e)
            )

    def run_tutorial_tests(self, tutorial: TutorialManifest, test_types: Optional[List[str]] = None) -> List[TestRunResult]:
        """Run tests for a specific tutorial"""
        results = []

        if not tutorial.enabled:
            print(f"⏭️  Skipping disabled tutorial: {tutorial.name}")
            return results

        # Get test types to run
        if test_types is None:
            test_types = tutorial.get_test_types()

        if not test_types:
            print(f"⏭️  No tests enabled for tutorial: {tutorial.name}")
            return results

        print(f"\n{'='*60}")
        print(f"Running tests for: {tutorial.name}")
        print(f"Description: {tutorial.description}")
        print(f"Test types: {', '.join(test_types)}")
        print(f"{'='*60}\n")

        for test_type in test_types:
            if test_type not in tutorial.tests:
                print(f"⚠️  Test type '{test_type}' not found in manifest")
                continue

            test_config = tutorial.tests[test_type]

            if not test_config.enabled:
                result = TestRunResult(
                    tutorial_name=tutorial.name,
                    test_type=test_type,
                    result=TestResult.SKIPPED,
                    duration=0
                )
                results.append(result)
                print(result)
                continue

            print(f"🔄 Running {test_type} tests...")
            if self.verbose:
                print(f"   Command: {test_config.command}")
                print(f"   Working directory: {tutorial.path}")

            test_result = self.run_command(
                test_config.command,
                tutorial.path,
                test_config.timeout
            )

            test_result.tutorial_name = tutorial.name
            test_result.test_type = test_type
            results.append(test_result)

            print(test_result)

            if self.verbose or not test_result.is_success():
                if test_result.output:
                    print(f"\n--- Output ---")
                    print(test_result.output)
                if test_result.error:
                    print(f"\n--- Error ---")
                    print(test_result.error)

            print()

        return results

    def run_all_tests(self, tutorial_names: Optional[List[str]] = None) -> List[TestRunResult]:
        """Run tests for all or specified tutorials"""
        tutorials = self.registry.get_all_tutorials(enabled_only=True)

        if tutorial_names:
            tutorials = [t for t in tutorials if t.name in tutorial_names]

        if not tutorials:
            print("No tutorials to test")
            return []

        print(f"\n{'='*60}")
        print(f"Tutorial Test Suite")
        print(f"Tutorials: {len(tutorials)}")
        print(f"{'='*60}")

        all_results = []
        for tutorial in tutorials:
            results = self.run_tutorial_tests(tutorial)
            all_results.extend(results)
            self.results.extend(results)

        return all_results

    def print_summary(self):
        """Print test results summary"""
        if not self.results:
            print("\nNo tests were run")
            return

        passed = sum(1 for r in self.results if r.result == TestResult.PASSED)
        failed = sum(1 for r in self.results if r.result == TestResult.FAILED)
        skipped = sum(1 for r in self.results if r.result == TestResult.SKIPPED)
        timeout = sum(1 for r in self.results if r.result == TestResult.TIMEOUT)
        error = sum(1 for r in self.results if r.result == TestResult.ERROR)
        total = len(self.results)

        total_duration = sum(r.duration for r in self.results)

        print(f"\n{'='*60}")
        print(f"Test Summary")
        print(f"{'='*60}")
        print(f"Total tests:   {total}")
        print(f"✅ Passed:     {passed}")
        print(f"❌ Failed:     {failed}")
        print(f"⏭️  Skipped:    {skipped}")
        print(f"⏱️  Timeout:    {timeout}")
        print(f"💥 Error:      {error}")
        print(f"⏱️  Duration:   {total_duration:.2f}s")
        print(f"{'='*60}")

        if failed > 0 or timeout > 0 or error > 0:
            print("\n❌ Failed/Error Tests:")
            for result in self.results:
                if result.result in [TestResult.FAILED, TestResult.TIMEOUT, TestResult.ERROR]:
                    print(f"   {result}")

        return failed == 0 and timeout == 0 and error == 0


def main():
    parser = argparse.ArgumentParser(description='Run tutorial tests')
    parser.add_argument('--tutorials-dir', type=str,
                       default=None,
                       help='Path to tutorials directory (default: parent of framework)')
    parser.add_argument('--tutorial', type=str, nargs='+',
                       help='Specific tutorial(s) to test')
    parser.add_argument('--type', type=str, nargs='+',
                       help='Specific test type(s) to run (unit, integration)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--validate', action='store_true',
                       help='Only validate manifests without running tests')
    parser.add_argument('--host', action='store_true',
                       help='Use host-specific TensorRT plans (sets PLAN_POSTFIX=.host)')

    args = parser.parse_args()

    # Find tutorials directory
    if args.tutorials_dir:
        tutorials_dir = Path(args.tutorials_dir)
    else:
        tutorials_dir = Path(__file__).parent.parent.parent

    if not tutorials_dir.exists():
        print(f"❌ Error: Tutorials directory not found: {tutorials_dir}")
        sys.exit(1)

    # Validate manifests if requested
    if args.validate:
        registry = TutorialRegistry(tutorials_dir)
        validation_errors = registry.validate_all()

        if validation_errors:
            print("❌ Validation errors found:")
            for tutorial_name, errors in validation_errors.items():
                print(f"\n{tutorial_name}:")
                for error in errors:
                    print(f"  - {error}")
            sys.exit(1)
        else:
            print("✅ All manifests are valid")
            sys.exit(0)

    # Run tests
    plan_postfix = ".host" if args.host else ""
    runner = TutorialTestRunner(tutorials_dir, verbose=args.verbose, plan_postfix=plan_postfix)

    try:
        runner.run_all_tests(tutorial_names=args.tutorial)
        success = runner.print_summary()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        runner.print_summary()
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

