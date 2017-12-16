#!/bin/bash
jpegtopnm "$1" > "$(basename "$1" .jpg).pnm"
convert -depth 8 -colorspace gray "$(basename "$1" .jpg).pnm" "$(basename "$1" .jpg).pgm"
