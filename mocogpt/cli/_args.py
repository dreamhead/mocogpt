class StartArgs:
    def __init__(self, port, settings):
        self._port = port
        self._settings = settings

    @property
    def port(self):
        return self._port

    @property
    def settings(self):
        return self._settings


