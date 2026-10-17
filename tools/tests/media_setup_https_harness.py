"""Local HTTPS fixture only; never publishes assets or changes system trust."""
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import threading
from urllib.parse import unquote, urlsplit


def create_certificate(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    executable = shutil.which('openssl')
    if not executable:
        bundled = Path('C:/Program Files/Git/usr/bin/openssl.exe')
        executable = str(bundled) if bundled.is_file() else None
    if not executable:
        raise RuntimeError('The local HTTPS test fixture requires an existing OpenSSL executable')
    certificate, private_key = directory/'certificate.pem', directory/'private-key.pem'
    result = subprocess.run([executable, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
        '-keyout', str(private_key), '-out', str(certificate), '-days', '1',
        '-subj', '/CN=EmbeddingGemma local HTTPS fixture',
        '-addext', 'subjectAltName=IP:127.0.0.1',
        '-addext', 'basicConstraints=critical,CA:TRUE'], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('Could not create the ephemeral local HTTPS fixture certificate')
    return certificate, private_key


class LocalBundleServer:
    def __init__(self, root, certificate, private_key):
        self.root = Path(root).resolve()
        self.certificate = Path(certificate).resolve()
        self.private_key = Path(private_key).resolve()
        self.lock = threading.Lock()
        self.reset()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                name = unquote(urlsplit(self.path).path).lstrip('/')
                path = (owner.root/name).resolve()
                if not path.is_relative_to(owner.root) or name != 'bundle.json' and not name.startswith('objects/'):
                    self.send_error(404); return
                with owner.lock:
                    fault = dict(owner.fault)
                    if name != 'bundle.json':
                        owner._object_requests.append(name)
                    ordinal = len(owner._object_requests)
                row = dict(utc=datetime.now(timezone.utc).isoformat(), path=name, bytes_sent=0)
                if name == 'bundle.json':
                    status = fault.get('descriptor_status', 200)
                else:
                    status = 404 if name == fault.get('missing_object') else 200
                if not path.is_file(): status = 404
                row['status'] = status
                if status != 200:
                    with owner.lock: owner._requests.append(row)
                    self.send_error(status); return
                body = fault.get('descriptor_body') if name == 'bundle.json' else None
                length = len(body) if body is not None else path.stat().st_size
                truncated = name != 'bundle.json' and ordinal == fault.get('truncate_ordinal')
                limit = min(1024*1024, length//2) if truncated else length
                row.update(declared_bytes=length, truncated=truncated)
                self.send_response(200)
                self.send_header('Content-Length', str(length))
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                try:
                    if body is not None:
                        self.wfile.write(body); row['bytes_sent'] = len(body)
                    else:
                        with path.open('rb') as source:
                            while row['bytes_sent'] < limit:
                                block = source.read(min(1024*1024, limit-row['bytes_sent']))
                                if not block: break
                                self.wfile.write(block); row['bytes_sent'] += len(block)
                                if (name == fault.get('pause_object') and not row.get('paused')
                                        and row['bytes_sent'] >= fault.get('pause_after_bytes', 2*1024*1024)):
                                    row['paused'] = True
                                    self.wfile.flush()
                                    if not fault['resume_event'].wait(timeout=20):
                                        raise TimeoutError('Owned HTTPS test pause was not released')
                    self.wfile.flush()
                    if truncated:
                        try: self.connection.shutdown(socket.SHUT_RDWR)
                        except OSError: pass
                except (OSError, ssl.SSLError) as error:
                    row['connection_error'] = type(error).__name__
                finally:
                    self.close_connection = True
                    with owner.lock: owner._requests.append(row)

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.httpd.daemon_threads = True
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(self.certificate), str(self.private_key))
        self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)
        self.url = f'https://127.0.0.1:{self.httpd.server_address[1]}/bundle.json'
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def object_requests(self):
        with self.lock: return list(self._object_requests)

    @property
    def requests(self):
        with self.lock: return [dict(row) for row in self._requests]

    def reset(self, **fault):
        with self.lock:
            self.fault = fault
            self._object_requests = []
            self._requests = []

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.httpd.shutdown(); self.httpd.server_close(); self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('The owned local HTTPS fixture did not stop')
