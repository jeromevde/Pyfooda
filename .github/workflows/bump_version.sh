#!/bin/bash

# Function to check if version exists on PyPI
check_version_exists() {
    local version=$1
    local response=$(curl -s "https://pypi.org/pypi/pyfooda/$version/json")
    if [[ $response == *"$version"* ]]; then
        return 0  # Version exists
    else
        return 1  # Version doesn't exist
    fi
}

# Bumps patch version (e.g., 0.1.1 -> 0.1.2)
bump_version() {
    local version=$1                    # Input version string
    local major=$(echo $version | cut -d. -f1)    # Get major version
    local minor=$(echo $version | cut -d. -f2)    # Get minor version
    local patch=$(echo $version | cut -d. -f3)    # Get patch version
    local new_patch=$((patch + 1))               # Increment patch
    echo "$major.$minor.$new_patch"              # Return new version
}

# Read current version and trim whitespace
current_version=$(cat pyfooda/VERSION | tr -d ' \t\n\r')
echo "[BUMP] Current version: $current_version" >&2

# Check if version exists on PyPI
echo "[BUMP] Checking if version $current_version exists on PyPI..." >&2
if check_version_exists "$current_version"; then
    new_version=$(bump_version "$current_version")
    echo "[BUMP] Version $current_version exists on PyPI. Bumping to $new_version" >&2
    printf "%s" "$new_version" > pyfooda/VERSION
    echo "[BUMP] Updated pyfooda/VERSION to $new_version" >&2

    if [ -n "$GITHUB_ACTIONS" ]; then
        echo "[BUMP] Committing and pushing version bump..." >&2
        git config --global user.name 'GitHub Actions'
        git config --global user.email 'actions@github.com'
        git remote set-url origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
        git add pyfooda/VERSION
        git commit -m "Bump version to $new_version [skip ci]" >&2
        git push >&2
        echo "[BUMP] Commit and push complete." >&2
    fi
    printf "%s" "$new_version"
    exit 0
else
    echo "[BUMP] Version $current_version does not exist on PyPI. No bump needed. Returning un-bumped version." >&2
    # Output the un-bumped version string to stdout
    printf "%s" "$current_version"
    exit 0
fi
fi 