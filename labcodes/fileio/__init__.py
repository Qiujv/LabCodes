import logging

logger = logging.getLogger(__name__)

from labcodes.fileio.base import LogFile, LogName
from labcodes.fileio.data_logger import capture

try:
    from labcodes.fileio.labber import read_labber
except:
    logger.exception("Fail to import fileio.labber.")

try:
    from labcodes.fileio.labrad import read_labrad, LabradRead, LabradDirectory
    from labcodes.fileio import labrad
except:
    logger.exception("Fail to import fileio.labrad.")

# try:
#     from labcodes.fileio.ltspice import LTSpiceRead
# except:
#     logger.exception('Fail to import fileio.ltspice.')

try:
    from labcodes.fileio.misc import data_to_json, data_from_json
except:
    logger.exception("Fail to import fileio.misc.")
