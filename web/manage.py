#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from app.res2lights import Res2lights, EncoderCNN, Model

sys.path.append('/home/r8v10/git/InvCo')
# response = requests.get('https://api.telegram.org/bot986672647:AAF3FZZqoG6HA1N_wd04ZXTr657iygqW_0I/sendMessage?chat_id=954201468&text='+str(sys.path))
# print('---- ',sys.path)
from demo2 import Demo
from ingrs_vocab import Vocabulary
from args import get_parser


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proj.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
