#!/bin/bash

# Script to build all plugins defined in tutorial.yaml files
# Located in plugins/common/

# Exit on error
set -e

# Parse command line arguments
BUILD_TARGET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            BUILD_TARGET="host"
            shift
            ;;
        --container)
            BUILD_TARGET="container"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host|--container]"
            echo "  --host       Build using build.host section from tutorial.yaml"
            echo "  --container  Build using build.container section from tutorial.yaml"
            echo "  (no option)  Build using build section from tutorial.yaml"
            exit 1
            ;;
    esac
done

# Export so it's available to build scripts
export BUILD_TARGET

# Get the absolute path of the script and plugins directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLUGINS_DIR="$(dirname "$SCRIPT_DIR")"

if [ -n "$BUILD_TARGET" ]; then
    echo "Building plugins using build.$BUILD_TARGET section"
else
    echo "Building plugins using build section"
fi
echo "Searching for plugins in: $PLUGINS_DIR"

# Find all tutorial.yaml files in immediate subdirectories of the plugins directory
find "$PLUGINS_DIR" -maxdepth 2 -name "tutorial.yaml" | while read config_file; do
    plugin_dir=$(dirname "$config_file")
    plugin_name=$(basename "$plugin_dir")

    echo "Processing plugin: $plugin_name"

    # Parse YAML using python to get build configuration
    # We output shell variable assignments to be evaluated
    eval $(python3 -c "
import yaml
import sys
import os

try:
    with open('$config_file', 'r') as f:
        data = yaml.safe_load(f)
        if data and 'build' in data:
            build_target = os.environ.get('BUILD_TARGET', '')
            
            # Try to get the specific build section (build.host or build.container)
            if build_target:
                section_name = f'{build_target}'
                if section_name in data['build'] and isinstance(data['build'][section_name], dict):
                    build = data['build'][section_name]
                else:
                    # Fall back to main build section
                    build = data['build']
            else:
                # No target specified, use main build section
                build = data['build']
            
            enabled = str(build.get('enabled', False)).lower()
            command = build.get('command', '')
            # Escape single quotes for shell safety
            command = command.replace(\"'\", \"'\\''\")
            print(f'BUILD_ENABLED={enabled}')
            print(f'BUILD_COMMAND=\'{command}\'')
        else:
            print('BUILD_ENABLED=false')
            print('BUILD_COMMAND=\'\'')
except Exception as e:
    print(f'Error parsing yaml: {e}', file=sys.stderr)
    print('BUILD_ENABLED=false')
    print('BUILD_COMMAND=\'\'')
")

    if [ "$BUILD_ENABLED" == "true" ] && [ -n "$BUILD_COMMAND" ]; then
        echo "  Build enabled. Executing: $BUILD_COMMAND"

        # Run the build command inside the plugin directory
        # We use a subshell to not affect the current directory
        (
            cd "$plugin_dir"
            echo "  Working directory: $(pwd)"
            eval "$BUILD_COMMAND"
        )

        # check exit code of the subshell
        if [ $? -ne 0 ]; then
            echo "  Error: Build failed for $plugin_name"
            exit 1
        else
            echo "  Build successful for $plugin_name"
        fi
    else
        echo "  Build skipped (not enabled or no command configured)"
    fi
    echo "----------------------------------------"
done

echo "All plugins processed."

if [ -n "$BUILD_TARGET" ]; then
    echo ""
    echo "=========================================="
    echo "Plugins were built using build.$BUILD_TARGET section."
    echo "To use them in unit tests, ensure your environment"
    echo "matches the build target ($BUILD_TARGET)."
    echo "=========================================="
fi

