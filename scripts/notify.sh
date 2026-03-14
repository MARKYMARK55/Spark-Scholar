#!/bin/bash
set -euo pipefail
# ────────────────────────────────────────────────────────────────────────────────
# Generic email notification for DGX Spark long-running jobs
# Usage: notify.sh "Subject" "Body message"
# ────────────────────────────────────────────────────────────────────────────────
EMAIL_TO="griffith.mark@gmail.com"
EMAIL_FROM="griffith.mark@gmail.com"
APP_PASSWORD="thyt zxzj fefa sqbm"

SUBJECT="${1:-DGX Spark Job Complete}"
BODY="${2:-Job finished.}"
TS="$(date '+%Y-%m-%d %H:%M:%S')"
FULL_SUBJECT="$SUBJECT ($TS)"

printf "From: %s\nTo: %s\nSubject: %s\n\n%s\n" \
  "$EMAIL_FROM" "$EMAIL_TO" "$FULL_SUBJECT" "$BODY" | \
curl --silent --show-error \
  --url 'smtps://smtp.gmail.com:465' \
  --ssl-reqd \
  --mail-from "$EMAIL_FROM" \
  --mail-rcpt "$EMAIL_TO" \
  --user "$EMAIL_FROM:$APP_PASSWORD" \
  --upload-file -

echo "Notification sent: $FULL_SUBJECT"
