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

# Function to bump version
bump_version() {
    local version=$1
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)
    local patch=$(echo $version | cut -d. -f3)
    local new_patch=$((patch + 1))
    echo "$major.$minor.$new_patch"
}

# Read current version
current_version=$(cat VERSION)

# Check if version exists on PyPI
if check_version_exists "$current_version"; then
    # Bump version
    new_version=$(bump_version "$current_version")
    echo "Version $current_version exists on PyPI. Bumping to $new_version"
    
    # Update VERSION file
    echo "$new_version" > VERSION
    
    # If running in GitHub Actions, commit and push
    if [ -n "$GITHUB_ACTIONS" ]; then
        git config --global user.name 'GitHub Actions'
        git config --global user.email 'actions@github.com'
        git add VERSION
        git commit -m "Bump version to $new_version [skip ci]"
        git push
    fi
    
    # Return new version
    echo "$new_version"
    exit 0
else
    echo "Version $current_version is available for publishing"
    echo "$current_version"
    exit 0
fi 