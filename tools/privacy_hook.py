"""Codex lifecycle entry point. Never echoes tool inputs or private findings."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embeddinggemma_tools.privacy import github_profile_findings, reviews, scan_objects, scan_staged
from embeddinggemma_tools.privacy_hook import decide

try:
    # Hook JSON is UTF-8 even when Windows text pipes default to a legacy codec.
    event = json.load(sys.stdin.buffer)
    if event.get('hook_event_name') == 'SessionStart':
        print(json.dumps({'hookSpecificOutput':{'hookEventName':'SessionStart','additionalContext':
            'Before publishing, use the project-privacy skill, check Git hook installation, scan outgoing history and actual payloads, and visually/OCR-review image bytes. Never bypass privacy checks or echo private values.'}}))
    else:
        cwd = event.get('cwd', str(Path.cwd()))
        root = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=cwd,text=True).strip())
        def checker():
            return not scan_staged(root) and not scan_objects(root, ['HEAD'], reviews(root))
        print(json.dumps(decide(event, checker, profile_checker=lambda:not github_profile_findings())))
except Exception:
    print(json.dumps({'hookSpecificOutput':{'hookEventName':'PreToolUse','permissionDecision':'deny',
        'permissionDecisionReason':'Privacy hook could not validate the publication; fix the checker before retrying.'}}))
