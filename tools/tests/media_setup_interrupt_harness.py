"""An owned hidden console isolates real Windows CTRL_C_EVENT delivery.

The relay and target alone share that console. No attachment or signal is sent
to the caller's console or unrelated processes. POSIX uses the owned PID only.
"""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, value):
    path = Path(path)
    pending = path.with_suffix('.writing')
    pending.write_text(json.dumps(value), encoding='utf-8')
    os.replace(pending, path)


def relay(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    root = config_path.parent
    signal.signal(signal.SIGINT, lambda *_: None)
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        assert kernel.SetConsoleCtrlHandler(None, False)
    child = None
    record = dict(relay_pid=os.getpid(), signal_sent=False, forced_cleanup=False)
    try:
        with (root/'stdout.txt').open('wb') as out, (root/'stderr.txt').open('wb') as err:
            child = subprocess.Popen(config['command'], cwd=config['cwd'], env=config['env'],
                                     stdout=out, stderr=err)
            record['child_pid'] = child.pid
            save(root/'ready.json', record)
            deadline = time.monotonic()+45
            while child.poll() is None:
                if time.monotonic() > deadline or (root/'cleanup').exists():
                    record['forced_cleanup'] = True
                    child.kill()
                    break
                if (root/'interrupt').exists() and not record['signal_sent']:
                    if os.name == 'nt':
                        members = (ctypes.c_uint32*8)()
                        count = kernel.GetConsoleProcessList(members, len(members))
                        record['console_process_count'] = count
                        record['console_processes'] = sorted(members[:count])
                        assert count == 2 and set(members[:count]) == {os.getpid(),child.pid}, ('Never signal a shared or unrelated console.',record)
                        assert kernel.GenerateConsoleCtrlEvent(0,0), ctypes.get_last_error()
                        record['signal_kind'] = 'GenerateConsoleCtrlEvent(CTRL_C_EVENT,0) in isolated owned console'
                    else:
                        os.kill(child.pid, signal.SIGINT)
                        record['signal_kind'] = 'SIGINT to owned child PID'
                    record['signal_sent'] = True
                    save(root/'signal.json', record)
                time.sleep(.01)
            record['exit_code'] = child.wait(timeout=10)
    finally:
        if child is not None and child.poll() is None:
            record['forced_cleanup'] = True
            child.kill()
            child.wait(timeout=10)
        record['terminal'] = True
        save(root/'done.json', record)


class InterruptedCli:
    def __init__(self, command, cwd, env, directory):
        self.root = Path(directory)
        self.root.mkdir()
        config = self.root/'config.json'
        save(config, dict(command=command,cwd=str(cwd),env=env))
        options = {}
        if os.name == 'nt':
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            options.update(creationflags=subprocess.CREATE_NEW_CONSOLE,startupinfo=startup)
        # A Windows venv redirector would itself join the console. Use the same
        # underlying interpreter directly; both relay and tested CLI need stdlib.
        executable = getattr(sys, '_base_executable', sys.executable)
        self.process = subprocess.Popen([executable,str(Path(__file__).resolve()),str(config)],
                                        stdout=subprocess.PIPE,stderr=subprocess.PIPE,**options)

    def wait_file(self, name, timeout=10):
        deadline = time.monotonic()+timeout
        while not (self.root/name).exists():
            if self.process.poll() is not None:
                out,err = self.process.communicate()
                raise AssertionError(f'Interrupt relay exited before {name}: {out!r} {err!r}')
            if time.monotonic() >= deadline:
                raise TimeoutError('Owned signal relay did not produce '+name)
            time.sleep(.02)
        return json.loads((self.root/name).read_text(encoding='utf-8'))

    def interrupt(self):
        (self.root/'interrupt').touch()
        return self.wait_file('signal.json')

    def finish(self):
        out,err = self.process.communicate(timeout=15)
        assert self.process.returncode == 0, (out,err)
        return self.wait_file('done.json')

    def close(self):
        if self.process.poll() is None:
            (self.root/'cleanup').touch()
            self.process.communicate(timeout=15)


if __name__ == '__main__':
    relay(sys.argv[1])
