"""Real local Git operations; no network or personal input is used."""
from pathlib import Path
import os
import shutil
import subprocess

SOURCE = Path(__file__).resolve().parents[2]

def git(root, *args, check=True):
    return subprocess.run(['git', *args], cwd=root, text=True, capture_output=True, check=check)

def fixture_repo(tmp_path, private_ancestor=False):
    root = tmp_path/'repository'; root.mkdir()
    git(root, 'init', '-b', 'main')
    git(root, 'config', 'user.name', 'fixture')
    git(root, 'config', 'user.email', ('private'+'@'+'gmail.com') if private_ancestor else '123+fixture@users.noreply.github.com')
    (root/'safe.txt').write_text('safe fixture\n')
    git(root, 'add', 'safe.txt'); git(root, 'commit', '-m', 'initial fixture')
    git(root, 'config', 'user.email', '123+fixture@users.noreply.github.com')
    (root/'tools/embeddinggemma_tools').mkdir(parents=True)
    shutil.copyfile(SOURCE/'tools/embeddinggemma_tools/privacy.py', root/'tools/embeddinggemma_tools/privacy.py')
    (root/'tools/embeddinggemma_tools/__init__.py').write_text('')
    shutil.copytree(SOURCE/'.githooks', root/'.githooks')
    git(root, 'add', '.githooks', 'tools'); git(root, 'commit', '-m', 'install public checker')
    git(root, 'config', 'core.hooksPath', str(root/'.githooks'))
    return root

def test_real_commit_reads_staged_blob_even_if_worktree_was_overwritten(tmp_path):
    root=fixture_repo(tmp_path)
    value='C:'+'\\'+'Users'+'\\'+'fixture-user'+'\\data'
    (root/'receipt.txt').write_text(value)
    git(root,'add','receipt.txt')
    (root/'receipt.txt').write_text('safe working copy')
    result=git(root,'commit','-m','must reject staged private data',check=False)
    assert result.returncode != 0
    assert 'fixture-user' not in result.stdout+result.stderr
    assert 'home-path' in result.stdout+result.stderr

def test_real_commit_message_trailer_is_rejected(tmp_path):
    root=fixture_repo(tmp_path)
    result=git(root,'commit','--allow-empty','-m','contact '+'fixture'+'@'+'gmail.com',check=False)
    assert result.returncode != 0
    assert 'personal-email' in result.stdout+result.stderr

def test_real_pre_push_rejects_private_ancestor_even_with_clean_head(tmp_path):
    root=fixture_repo(tmp_path,private_ancestor=True)
    sink=tmp_path/'sink.git'; git(tmp_path,'init','--bare',str(sink))
    result=git(root,'push',str(sink),'HEAD:refs/heads/main',check=False)
    assert result.returncode != 0
    assert 'private-identity' in result.stdout+result.stderr
    assert not git(tmp_path,'--git-dir='+str(sink),'for-each-ref','--format=%(refname)').stdout.strip()

def test_real_clean_commit_and_local_push_are_allowed(tmp_path):
    root=fixture_repo(tmp_path)
    git(root,'commit','--allow-empty','-m','public identity and safe content')
    sink=tmp_path/'sink.git'; git(tmp_path,'init','--bare',str(sink))
    git(root,'push',str(sink),'HEAD:refs/heads/main')
    assert git(tmp_path,'--git-dir='+str(sink),'show-ref','--verify','refs/heads/main').returncode==0


def test_explicit_ci_source_revision_still_rejects_private_ancestors(tmp_path):
    root=fixture_repo(tmp_path,private_ancestor=True)
    revision=git(root,'rev-parse','HEAD').stdout.strip()
    import sys
    result=subprocess.run([sys.executable,'-m','embeddinggemma_tools.privacy','history',
        '--root',str(root),'--revision',revision],cwd=root/'tools',capture_output=True,text=True)
    assert result.returncode==1
    assert 'private-identity' in result.stdout


def test_linked_checkout_uses_the_installed_validator_for_all_three_hooks(tmp_path):
    root=fixture_repo(tmp_path)
    linked=tmp_path/'older-checkout'
    git(root,'worktree','add','--detach',str(linked))
    # This checkout's checker is deliberately older/broken and is not staged.
    # Installed hooks must still audit this checkout's actual index and history.
    (linked/'tools/embeddinggemma_tools/privacy.py').write_text('raise RuntimeError("older checkout checker")\n')
    (linked/'safe.txt').write_text('safe linked-worktree change\n')
    git(linked,'add','safe.txt')
    committed=git(linked,'commit','-m','public linked checkout',check=False)
    assert committed.returncode==0,'Installed commit hooks must use their own reviewed validator'
    sink=tmp_path/'sink.git';git(tmp_path,'init','--bare',str(sink))
    pushed=git(linked,'push',str(sink),'HEAD:refs/heads/main',check=False)
    assert pushed.returncode==0,'Installed push hook must use its own reviewed validator'
