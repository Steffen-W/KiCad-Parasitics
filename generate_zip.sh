#!/bin/bash

rm KiCad-Parasitics.zip

git ls-files  -- '*.json' '*.png' '*.py' | xargs zip KiCad-Parasitics.zip