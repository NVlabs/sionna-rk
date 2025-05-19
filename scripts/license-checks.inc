#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

function check-license() {
    # File with date and time when the license was accepted
    license_accepted_file="${HOME}/.NVIDIA_SIONNA_RK_LICENSE_ACCEPTED"

    if [ ! -f "$license_accepted_file" ]; then
        # Define the text file to display
        license_file="$(dirname "${BASH_SOURCE[0]}")/../LICENSE"

        # Check if the file exists
        if [[ ! -f "$license_file" ]]; then
            echo "File '$license_file' not found!"
            exit 1
        fi

        # Display the contents of the file
        cat "$license_file"
        echo -e "\n\n"

        # Prompt for user confirmation
        read -r -p "Do you accept the license terms? [y/N]: " response

        # Handle user input
        case "$response" in
            [yY][eE][sS]|[yY])
                echo "License accepted. Proceeding..."
                echo "$(date)" > "$license_accepted_file"
                ;;
            *)
                echo "License not accepted. Exiting..."
                exit 1
                ;;
        esac
    fi
}
