"""Install this repository's hooks, without overwriting an unrelated hook setup."""
from pathlib import Path
import subprocess

root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
hooks = root / '.githooks'
current = subprocess.run(['git', 'config', '--local', '--get', 'core.hooksPath'], capture_output=True, text=True).stdout.strip()
if current and Path(current).resolve() != hooks.resolve():
    raise SystemExit('An existing hooksPath needs explicit review; it was preserved.')
for name in ('pre-commit', 'commit-msg', 'pre-push'):
    if not (hooks / name).is_file():
        raise SystemExit('A required privacy hook is missing.')
subprocess.check_call(['git', 'config', '--local', 'core.hooksPath', str(hooks)])
print('Privacy Git hooks installed for this repository and its worktrees.')
