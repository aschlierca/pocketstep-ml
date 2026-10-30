#!/bin/bash
# Sets up code signing so Xcode can install PocketStep on your iPhone. Run once after cloning:
#
#   ios/setup_signing.sh            # detect your team from your Apple Development certificate
#   ios/setup_signing.sh TEAM_ID    # or name the team explicitly
#
# Writes ios/Config/Signing.xcconfig (git-ignored). The bundle identifier gets your team ID as a
# suffix, because two teams cannot register the same identifier.
set -euo pipefail
cd "$(dirname "$0")"

team="${1:-}"
if [ -z "$team" ]; then
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT
  security find-certificate -a -c "Apple Development" -p 2>/dev/null \
    | awk -v dir="$tmp" '/-----BEGIN CERTIFICATE-----/ { n++ } n { print > (dir "/cert" n ".pem") }'
  teams=()
  labels=()
  for cert in "$tmp"/cert*.pem; do
    [ -e "$cert" ] || continue
    openssl x509 -in "$cert" -noout -checkend 0 >/dev/null 2>&1 || continue   # skip expired
    subject=$(openssl x509 -in "$cert" -noout -subject 2>/dev/null)
    id=$(sed -n 's/.*OU *= *\([A-Z0-9]\{10\}\).*/\1/p' <<<"$subject")
    name=$(sed -n 's/.*O *= *\([^,]*\).*/\1/p' <<<"$subject")
    [ -n "$id" ] || continue
    if [[ ! " ${teams[*]-} " =~ " $id " ]]; then
      teams+=("$id")
      labels+=("$id ($name)")
    fi
  done

  if [ ${#teams[@]} -eq 0 ]; then
    cat <<'MSG'
No Apple Development certificate found. In Xcode:
  1. Settings > Accounts > + > Apple ID, and sign in (a free Apple ID works).
  2. Select the team > Manage Certificates... > + > Apple Development.
Then run this script again.
MSG
    exit 1
  elif [ ${#teams[@]} -eq 1 ]; then
    team="${teams[0]}"
    echo "Using team ${labels[0]}"
  else
    echo "Several development teams found:"
    PS3="Choose a team: "
    select choice in "${labels[@]}"; do
      [ -n "${choice:-}" ] && { team="${choice%% *}"; break; }
    done
  fi
fi

if [[ ! "$team" =~ ^[A-Z0-9]{10}$ ]]; then
  echo "\"$team\" does not look like a 10-character Apple team ID." >&2
  exit 1
fi

cat > Config/Signing.xcconfig <<CONFIG
// Written by ios/setup_signing.sh for this computer. Not committed.
DEVELOPMENT_TEAM = $team
POCKETSTEP_BUNDLE_SUFFIX = .$team
CONFIG

echo "Wrote ios/Config/Signing.xcconfig (team $team, bundle ID org.pocketstep.PocketStep.$team)."
echo "Next: open ios/PocketStep.xcodeproj, choose your iPhone as the destination and press Run."
