import os
import sys


_ENABLE = True
if not sys.stdout.isatty() or os.name == 'nt':
    # Disable when not printing on STDOUT
    # Disable on Windows until we test this
    _ENABLE = False


class Color:
    _COLORS = {
        'red': '\x1b[31m{msg}\x1b[39m',
        'blue': '\x1b[34m{msg}\x1b[39m',
        'cyan': '\x1b[36m{msg}\x1b[39m',
        'black': '\x1b[30m{msg}\x1b[39m',
        'green': '\x1b[32m{msg}\x1b[39m',
        'white': '\x1b[37m{msg}\x1b[39m',
        'yellow': '\x1b[33m{msg}\x1b[39m',
        'magenta': '\x1b[35m{msg}\x1b[39m',
    }

    @staticmethod
    def format(msg: str, color: str) -> str:
        """Returns a colored version of the given message string.

        :param msg: The string to colorify.
        :param color: The color specifier i.e. 'red', 'blue', 'green', etc.
        :return: A colored version of the string if the output is a terminal
        and the platform is not Windows.
        """
        if _ENABLE:
            return Color._COLORS.get(color.lower(), '{msg}').format(msg=msg)
        else:
            return msg
