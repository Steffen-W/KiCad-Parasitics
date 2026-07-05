#!/bin/bash
set -euo pipefail

rm -f KiCad-Parasitics.zip
mv metadata.json metadata_.json
# Always put the original metadata.json back, even if a step below fails.
trap 'mv -f metadata_.json metadata.json' EXIT

jq --arg today "$(date +%Y.%m.%d)" '.versions[0].version |= $today' metadata_.json > metadata.json

# Validate the generated metadata.json before it goes into the release ZIP:
# reject empty/invalid JSON (e.g. from a failing jq) and a version string that
# isn't in YYYY.MM.DD format (e.g. if the jq substitution above silently no-ops).
jq empty metadata.json
version=$(jq -r '.versions[0].version' metadata.json)
if ! [[ "$version" =~ ^[0-9]{4}\.[0-9]{2}\.[0-9]{2}$ ]]; then
    echo "ERROR: metadata.json has an invalid version '$version' (expected YYYY.MM.DD)" >&2
    exit 1
fi

git ls-files  -- 'metadata.json' 'resources*.png' 'plugins*.png' 'plugins*.py' 'plugins/plugin.json' 'plugins/requirements.txt' | xargs zip KiCad-Parasitics.zip
echo "$(realpath KiCad-Parasitics.zip)"

if grep -rq "debug = 1" plugins/*.py; then
    echo "WARNING: 'debug = 1' found in:"
    grep -rl "debug = 1" plugins/*.py
fi