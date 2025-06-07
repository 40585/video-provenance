#!/bin/bash

# Set input and output directories
INPUT_DIR="./../resources/ffmpeg_original_frames_192_108"
OUTPUT_DIR="./../resources/ffmpeg_resized_frames_48_27"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all jpg, jpeg, and png files in the input directory
for file in "$INPUT_DIR"/*.{jpg,jpeg,png}; do
    # Check if file exists (to handle case of no matches)
    [ -e "$file" ] || continue
    
    # Get the filename without the path
    filename=$(basename "$file")
    
    # Run ffmpeg command
    ffmpeg -i "$file" -vf "scale=48:27:flags=bilinear" -y "$OUTPUT_DIR/$filename"
    
    # Check if ffmpeg command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully resized: $filename"
    else
        echo "Failed to resize: $filename"
    fi
done

echo "Resizing complete!"