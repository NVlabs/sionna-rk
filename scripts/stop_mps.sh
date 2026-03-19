#!/bin/bash

echo "Stopping MPS Server"
echo "quit" | nvidia-cuda-mps-control
