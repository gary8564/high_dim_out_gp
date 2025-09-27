#!/usr/bin/env bash
set -euo pipefail

# Download emulator inputs (CSV from GitHub) and simulator outputs (TIF from Figshare)
# into a structured data layout under ../data relative to this script:
#
# data/
#   acheron/
#     train/
#       input/   # CSVs from GitHub
#       output/  # .tif from Figshare article 20449410
#     test/
#       input/   # CSVs from GitHub (validation)
#       output/  # .tif from Figshare article 20454936
#   synthetic/
#     train/
#       input/   # CSVs from GitHub
#       output/  # .tif from Figshare article 20449395
#     test/
#       input/   # CSVs from GitHub (validation)
#       output/  # .tif from Figshare article 20454933

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

# Desired folder structure
ACHERON_TRAIN_INPUT_DIR="${DATA_DIR}/acheron/train/input"
ACHERON_TRAIN_OUTPUT_DIR="${DATA_DIR}/acheron/train/output"
ACHERON_TEST_INPUT_DIR="${DATA_DIR}/acheron/test/input"
ACHERON_TEST_OUTPUT_DIR="${DATA_DIR}/acheron/test/output"

SYN_TRAIN_INPUT_DIR="${DATA_DIR}/synthetic/train/input"
SYN_TRAIN_OUTPUT_DIR="${DATA_DIR}/synthetic/train/output"
SYN_TEST_INPUT_DIR="${DATA_DIR}/synthetic/test/input"
SYN_TEST_OUTPUT_DIR="${DATA_DIR}/synthetic/test/output"

mkdir -p \
    "${ACHERON_TRAIN_INPUT_DIR}" "${ACHERON_TRAIN_OUTPUT_DIR}" \
    "${ACHERON_TEST_INPUT_DIR}"  "${ACHERON_TEST_OUTPUT_DIR}" \
    "${SYN_TRAIN_INPUT_DIR}"     "${SYN_TRAIN_OUTPUT_DIR}" \
    "${SYN_TEST_INPUT_DIR}"      "${SYN_TEST_OUTPUT_DIR}"

# GitHub CSV sources: web_url|desired_dir
CSV_SOURCES=(
  "https://github.com/yildizanil/frontiers_yildizetal/blob/main/src/frontiers_yildizetal/utilities/input/acheron_emulator.csv|${ACHERON_TRAIN_INPUT_DIR}"
  "https://github.com/yildizanil/frontiers_yildizetal/blob/main/src/frontiers_yildizetal/utilities/input/acheron_validation_emulator.csv|${ACHERON_TEST_INPUT_DIR}"
  "https://github.com/yildizanil/frontiers_yildizetal/blob/main/src/frontiers_yildizetal/utilities/input/synth_emulator.csv|${SYN_TRAIN_INPUT_DIR}"
  "https://github.com/yildizanil/frontiers_yildizetal/blob/main/src/frontiers_yildizetal/utilities/input/synth_validation_emulator.csv|${SYN_TEST_INPUT_DIR}"
)

# Figshare articles mapped to desired output directories: article_id|desired_dir
FIGSHARE_MAP=(
  "20449410|${ACHERON_TRAIN_OUTPUT_DIR}"
  "20454936|${ACHERON_TEST_OUTPUT_DIR}"
  "20449395|${SYN_TRAIN_OUTPUT_DIR}"
  "20454933|${SYN_TEST_OUTPUT_DIR}"
)

# Convert a GitHub blob URL to its raw URL
to_raw_github_url() {
  local url="$1"
  printf '%s' "$url" | sed -E 's#https://github.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)#https://raw.githubusercontent.com/\1/\2/\3/\4#'
}

download_github_csv() {
  local web_url="$1"
  local desired_dir="$2"
  local raw_url filename
  raw_url="$(to_raw_github_url "$web_url")"
  filename="$(basename "$raw_url")"
  echo "Downloading CSV $(basename "$filename") -> ${desired_dir} ..."
  curl -L --fail --retry 3 --retry-delay 2 -o "${desired_dir}/${filename}" "${raw_url}"
}

download_article() {
  local article_id="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}"

  echo "Fetching file list for article ${article_id} -> ${out_dir} ..."
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
      (*.zip|*.ZIP|*.Zip)
        echo "Unzipping ${name} ..."
        unzip -q -o "${out_dir}/${name}" -d "${out_dir}"
        rm -f "${out_dir}/${name}"
        ;;
    esac
  done
}

main() {
  # Download GitHub CSV inputs
  local csv_pair csv_url csv_dest
  for csv_pair in "${CSV_SOURCES[@]}"; do
    IFS='|' read -r csv_url csv_dest <<< "$csv_pair"
    download_github_csv "$csv_url" "$csv_dest"
  done

  # Download Figshare outputs
  local fs_pair fs_id fs_dest
  for fs_pair in "${FIGSHARE_MAP[@]}"; do
    IFS='|' read -r fs_id fs_dest <<< "$fs_pair"
    download_article "$fs_id" "$fs_dest"
  done

  echo "All downloads complete. Files saved in: ${DATA_DIR}"
}

main "$@"


