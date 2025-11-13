#!/bin/bash
# Clean up generated files

rm -f winner-*.pickle
rm -f winner-*.gv
rm -f winner-*.gv.pdf
rm -f neat-checkpoint-*
rm -f *.svg
rm -rf __pycache__
rm -rf .pytest_cache

echo "Cleanup complete!"
