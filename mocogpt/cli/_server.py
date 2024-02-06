import sys

from loguru import logger

from mocogpt.core.actual_server import ActualGptServer, Monitor

logger.remove()
logger.add(sys.stdout, colorize=True,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
                  "<red>|</red> {level: <8} "
                  "<red>|</red> <level>{message}</level>")


class LogMonitor(Monitor):
    def on_server_start(self, server):
        logger.info("Server started on port {port}", port=server.port)

    def on_server_end(self, server, elapsed):
        logger.info("Server stopped")
        logger.info("Total time: {elapsed} seconds", elapsed=elapsed)

    async def on_session_start(self, request):
        logger.info(f"Request: {request}")

    async def on_session_end(self, response):
        logger.info(f"Response: {response}")


def console_server(port):
    return ActualGptServer(port, monitor=LogMonitor())
