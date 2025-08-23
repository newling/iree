
#!/usr/bin/env bash
#
# cleanup_artifacts.sh
#
# This script removes all files in the current directory that do NOT end with
# ".sh" or ".hip".

set -euo pipefail

echo "=== Cleanup Script Starting ==="
echo "Working directory: $(pwd)"
echo

# Iterate through items in the current directory.
for item in *; do

  echo "Checking: $item"

  # Skip directories
  if [ -d "$item" ]; then
    echo "  -> Skipping (is a directory)"
    continue
  fi

  # Keep files ending in .sh or .hip
  if [[ "$item" == *.sh || "$item" == *.hip ]]; then
    echo "  -> Keeping (script/source file)"
    continue
  fi

  echo "  -> Removing (unwanted artifact)"
  rm -v -- "$item"
done

echo
echo "=== Cleanup Complete ==="
