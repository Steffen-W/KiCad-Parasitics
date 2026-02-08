#!/bin/bash

rm KiCad-Parasitics.zip
mv metadata.json metadata_.json
jq --arg today "$(date +%Y.%m.%d)" '.versions[0].version |= $today' metadata_.json > metadata.json

git ls-files  -- 'metadata.json' 'resources*.png' 'plugins*.png' 'plugins*.py' 'plugins/plugin.json' 'plugins/requirements.txt' | xargs zip KiCad-Parasitics.zip
mv metadata_.json metadata.json
echo "$(realpath KiCad-Parasitics.zip)"

if grep -rq "debug = 1" plugins/*.py; then
    echo "WARNING: 'debug = 1' found in:"
    grep -rl "debug = 1" plugins/*.py
fi