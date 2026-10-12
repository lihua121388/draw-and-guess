"""Mirror a growing UTF-8 log without changing carriage-return progress updates."""
import codecs
import sys
import time


def emit(text):
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding))
    sys.stdout.flush()


class LogConsole:
    def __init__(self, path, tail_bytes=0):
        self.stream = path.open("rb")
        if tail_bytes:
            self.stream.seek(max(0, path.stat().st_size - tail_bytes))
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def pump(self):
        while True:
            chunk = self.stream.read(65536)
            if not chunk:
                return
            emit(self.decoder.decode(chunk))

    def close(self):
        self.pump()
        emit(self.decoder.decode(b"", final=True))
        self.stream.close()


def wait_with_console(child, log):
    console = LogConsole(log)
    try:
        while child.poll() is None:
            console.pump()
            time.sleep(.1)
        return child.wait()
    finally:
        console.close()


def child_environment():
    import os
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment
