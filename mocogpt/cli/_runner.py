import json
import signal
import time

from ._args import StartArgs
from ._parser import ConfigParser


class CliRunner:
    @staticmethod
    def run(config, port):
        with open(config) as f:
            settings = json.load(f)
            args = StartArgs(port, settings)
            server = ConfigParser().parse(args)

            stopwatch = int(time.time())

            def cleanup_before_exit(signum, frame):
                server.stop_server(int(time.time()) - stopwatch)

            signal.signal(signal.SIGINT, cleanup_before_exit)
            server.start_server()

