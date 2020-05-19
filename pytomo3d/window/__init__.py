#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Wenjie Lei (lei@princeton.edu), 2016
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

from __future__ import (absolute_import, division, print_function)
import logging

# Setup the logger.
logger = logging.getLogger("pytomo3d.window")
#logger.setLevel(logging.WARNING)      # NOQA
logger.setLevel(logging.DEBUG)         # NOQA
# Prevent propagating to higher loggers.
logger.propagate = 0
# Console log handler.
ch = logging.StreamHandler()
# Add formatter
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s " + \
    "[%(filename)s:%(lineno)d]: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
logger.addHandler(ch)
