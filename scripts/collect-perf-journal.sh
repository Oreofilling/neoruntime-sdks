#!/bin/sh
# collect-perf-journal.sh — perf-run journal evidence (run ON the device).
#
# Greps the systemd journal for the camera-daemon / device-control log
# lines whose counters are NOT RPC-exposed (the known observability
# gap, 2026-09-12 perf matrix): strict-gate milestones with avg_wait,
# FdPublisher quota/lease rejects and client drops, FrameRouter and
# Watchdog force-reclaims, DspService warnings, device event hub
# backlog drops, and OOM history.
#
# Usage (from the test host):
#   ssh root@<device> 'sh -s' < scripts/collect-perf-journal.sh \
#       --since "2026-09-12 10:00:00" [--until "..."] > journal-evidence.txt
#
# Output: one section per pattern group — match count, derived numbers
# (avg_wait min/mean/max, cumulative reject counters), and up to 5
# sample lines. Absence of a section means zero matches, which for the
# Debug-level hub drops also means "not logged", not "did not happen".
set -u

SINCE=""
UNTIL=""
while [ $# -gt 0 ]; do
    case "$1" in
        --since) SINCE="$2"; shift 2 ;;
        --until) UNTIL="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[ -n "$SINCE" ] || SINCE="-2h"

JBASE="journalctl --no-pager --since $SINCE"
[ -n "$UNTIL" ] && JBASE="$JBASE --until $UNTIL"

echo "# perf journal evidence  (since: $SINCE${UNTIL:+  until: $UNTIL})"
echo "# host: $(hostname)  generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo

section() {
    # section <label> <pattern> [journalctl-extra-args...]
    label="$1"; pat="$2"; shift 2
    # shellcheck disable=SC2086
    matches=$($JBASE "$@" 2>/dev/null | grep -E "$pat")
    n=$(printf '%s\n' "$matches" | grep -c .)
    echo "## $label : $n match(es)"
    [ "$n" -gt 0 ] && printf '%s\n' "$matches" | head -5 | cut -c1-220
    echo
}

# -- strict gate milestones (INFO level): extract avg_wait series ------
$JBASE 2>/dev/null | grep -F 'AiOverlaySubscriber: strict gate' > /tmp/.pf_strict.$$
if [ -s /tmp/.pf_strict.$$ ]; then
    echo "## strict_gate_milestones : $(wc -l < /tmp/.pf_strict.$$) line(s)"
    head -3 /tmp/.pf_strict.$$ | cut -c1-220
    awk -F'avg_wait=' '{split($2, a, " "); if (a[1] != "") print a[1]}' \
        /tmp/.pf_strict.$$ | sort -n | awk '
        {v[NR]=$1; s+=$1}
        END {printf "avg_wait_ms: n=%d min=%.1f mean=%.1f max=%.1f\n", \
             NR, v[1], s/NR, v[NR]}'
else
    echo "## strict_gate_milestones : 0 match(es)"
fi
rm -f /tmp/.pf_strict.$$
echo

# -- FdPublisher quota/lease rejects: report the cumulative maxima -----
$JBASE -p warning 2>/dev/null | grep -E 'FdPublisher: (quota|lease) reject' \
    > /tmp/.pf_rej.$$
if [ -s /tmp/.pf_rej.$$ ]; then
    echo "## fdpublisher_rejects : $(wc -l < /tmp/.pf_rej.$$) warn line(s) (rate-limited 1/s)"
    head -3 /tmp/.pf_rej.$$ | cut -c1-220
    q=$(grep -oE 'quota_rejected=[0-9]+' /tmp/.pf_rej.$$ | cut -d= -f2 | sort -n | tail -1)
    l=$(grep -oE 'lease_rejected=[0-9]+' /tmp/.pf_rej.$$ | cut -d= -f2 | sort -n | tail -1)
    echo "cumulative maxima: quota_rejected=${q:-0} lease_rejected=${l:-0}"
else
    echo "## fdpublisher_rejects : 0 match(es)"
fi
rm -f /tmp/.pf_rej.$$
echo

# -- remaining warn/error groups ----------------------------------------
section fd_client_drops     'FdPublisher: Dropping desynced client'      -p warning
section frame_router        'FrameRouter:'                              -p warning
section frame_watchdog      'Watchdog: Frame'                           -p warning
section dsp_service_warn    'DspService:'                               -p warning
section event_hub_drops     'SubscribeEvents: subscriber backlog full'
section oom_history         'Out of memory|oom-killer|invoked oom-killer'

# -- journal window bounds ----------------------------------------------
echo "## journal_window"
$JBASE -o short 2>/dev/null | sed -n '1p;$p' | cut -c1-40
