#!/bin/bash

# Define the start and end of the numerical range
START=1
END=160

# Loop through the numbers from START to END
for i in $(seq -w $START $END); do
  # Find the matching files for the current number (i)
  file1=$(find . -name "*_${i}_ocr.txt" -print -quit)
  file2=$(find . -name "*_${i}_ocr25.txt" -print -quit)

  # Check if both files exist
  if [[ -f "$file1" && -f "$file2" ]]; then
    echo "--- Comparing "$file1" and "$file2" ---" >> diff_results.log
    # Compare the two files and print the result
    diff "$file1" "$file2" >> diff_results.log
    echo "---------------------------------------" >> diff_results.log
  fi
done
