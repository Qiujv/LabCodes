from labcodes.fileio.base import LogFile, LogName

# try:
#     from labcodes.fileio.labber import LabberRead, LabberWrite
# except:
#     print('WARNING: Fail to import fileio.labber. It may because python version >3.8.')

try:
    from labcodes.fileio.labrad import read_labrad, LabradRead
except:
    print('WARNING: Fail to import fileio.labrad.')

# try:
#     from labcodes.fileio.ltspice import LTSpiceRead
# except:
#     print('WARNING: Fail to import fileio.ltspice.')

try:
    from labcodes.fileio.misc import data_to_json, data_from_json
except:
    print('WARNING: Fail to import fileio.misc.')
