#!/usr/bin/env bash
# Build the Brightspace submission: one zip with the report, all source code and readme.txt.
# Only files tracked by git go in (no .git, envs/, caches or local outputs).
#
#   scripts/make_submission_zip.sh              # full: code, results and traces (~600 MB)
#   scripts/make_submission_zip.sh --no-traces  # without traces/ (~40 MB), if the upload
#                                               # limit is too small; the traces stay in the repo
# Output: vla-gpu-latency_submission[_notraces].zip in the repo's parent directory.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=vla-gpu-latency_submission
EXCL=()
if [[ ${1:-} == --no-traces ]]; then
  NAME=${NAME}_notraces
  EXCL=(':!traces')
fi
OUT="$(cd .. && pwd)/$NAME.zip"
rm -f "$OUT"
git ls-files -z -- . "${EXCL[@]}" | xargs -0 zip -q -9 "$OUT"
[[ -f readme.txt && -f paper/main.pdf ]] || { echo "missing readme.txt or paper/main.pdf"; exit 1; }
listing=$(unzip -l "$OUT")
for f in readme.txt paper/main.pdf bench/Makefile; do
  grep -qE " $f\$" <<<"$listing" || { echo "$f not in zip"; exit 1; }
done
echo "$OUT: $(du -h "$OUT" | cut -f1), $(tail -1 <<<"$listing" | awk '{print $2}') files"
