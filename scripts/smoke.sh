#!/bin/zsh
# Runs smoke.ts with node's type stripping and pi's node_modules on the
# resolution path. argv[1] selects the case: "model-first" (config override),
# "quota" (quota awareness), or default (ladder/rotation defaults). HOME is
# redirected to a temp dir for every case so config/quota files never touch
# the real ~/.pi/agent.
set -e
cd "$(dirname "$0")/.."

PI_NM="$HOME/.pi/agent/install/releases/1.5.2/node_modules"
SMOKE_DIR=$(mktemp -d /tmp/gemini-router-smoke.XXXXXX)
trap 'rm -rf "$SMOKE_DIR"' EXIT

mkdir -p "$SMOKE_DIR/node_modules/@earendil-works" "$SMOKE_DIR/home/.pi/agent"
ln -s "$PI_NM/@earendil-works/pi-ai" "$SMOKE_DIR/node_modules/@earendil-works/pi-ai"
cp index.ts signal.ts smoke.ts "$SMOKE_DIR/"

if [[ "$1" == "model-first" ]]; then
  cat > "$SMOKE_DIR/home/.pi/agent/gemini-router.json" <<'JSON'
{
  "ladders": { "gemini-flash": ["gemini-3.5-flash", "gemini-3.8-flash"] },
  "order": "model-first"
}
JSON
  HOME="$SMOKE_DIR/home" GEMINI_ROUTER_SMOKE=model-first \
    node --experimental-strip-types "$SMOKE_DIR/smoke.ts"
elif [[ "$1" == "quota" ]]; then
  HOME="$SMOKE_DIR/home" GEMINI_ROUTER_SMOKE=quota \
    node --experimental-strip-types "$SMOKE_DIR/smoke.ts"
else
  HOME="$SMOKE_DIR/home" node --experimental-strip-types "$SMOKE_DIR/smoke.ts"
fi
