#!/usr/bin/env bash
set -euo pipefail

# Download selected Figshare datasets into ../data relative to this script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
mkdir -p "${DATA_DIR}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but not found." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "This script requires jq to parse Figshare API JSON. Install it and re-run." >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required to extract downloaded .zip files. Install it and re-run." >&2
  exit 1
fi

# Ordered list of Figshare article_id|folder_name pairs
ARTICLES=(
  "20449410|Acheron_rock_avalanche"
  "20454936|Acheron_rock_avalanche_Validation"
  "20449395|Synthetic_case"
  "20454933|Synthetic_case_Validation"
)

download_article() {
  local article_id="$1"
  local out_name="$2"
  local out_dir="${DATA_DIR}/${out_name}"
  mkdir -p "${out_dir}"

  echo "Fetching file list for article ${article_id} -> ${out_name} ..."
  local json
  json="$(curl -sS "https://api.figshare.com/v2/articles/${article_id}")"

  # Check if there are files to download
  local file_count
  file_count="$(printf '%s' "$json" | jq -r '(.files | length) // 0')"
  if [[ "$file_count" -eq 0 ]]; then
    echo "No files found for article ${article_id}." >&2
    return 1
  fi

  # Stream URLs and names and download each
  printf '%s' "$json" | jq -r '.files[]? | [.download_url, .name] | @tsv' | while IFS=$'\t' read -r url name; do
    echo "  - Downloading ${name} ..."
    curl -L --fail --retry 3 --retry-delay 2 -o "${out_dir}/${name}" "${url}"

    # If the file is a ZIP, unzip it
    case "${name}" in
      *.zip|*.ZIP|*.Zip)
        echo "Unzipping ${name} ..."
        unzip -q -o "${out_dir}/${name}" -d "${out_dir}"
        rm -f "${out_dir}/${name}"
        ;;
    esac
  done
}

main() {
  local pair id name
  for pair in "${ARTICLES[@]}"; do
    IFS='|' read -r id name <<< "$pair"
    download_article "$id" "$name"
  done

  echo "All downloads complete. Files saved in: ${DATA_DIR}"
}

main "$@"


