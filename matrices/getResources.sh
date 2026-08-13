#!/bin/bash
set -euo pipefail

# Traverse the "mat_resources" file
while IFS= read -r link; do
    # Skip blank lines
    [ -z "$link" ] && continue

    # Download the link using wget
    wget "$link"

    # Extract the tar file
    tar -xf "$(basename "$link")"

    # Delete the tar file
    rm "$(basename "$link")"

    # SuiteSparse extracts to a folder named after the matrix. Keep only the
    # .mtx that matches the folder name and the other variants.
    name="$(basename "$link" .tar.gz)"
    mv "$name/$name.mtx" .
    rm -rf "$name"
done < mat_resources.txt
