#!/bin/bash
date -r build.log || echo "build.log not found"
grep -iC 2 -e "\(^\|\s\)error\([[:space:]:]\|$\)" --color build.log || echo "done scanning build.log"
