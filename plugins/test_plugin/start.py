
logger = setupData["logger"]
isCPUonly = setupData["isCPUonly"]

# Example importing base modules
import os
print(os.listdir("./"))

# Example importing custom modules
from plugins.test_plugin.test import doImportableFunction
print(doImportableFunction(1,2))


def start_pre(data=None):
    # Example using modules
    global os, doImportableFunction, logger, isCPUonly
    logger.log(f'Start pre: {data} (isCPUonly: {isCPUonly})')

    logger.log(f'doImportableFunction(1,2) = {doImportableFunction(1,2)}')
    logger.log(", ".join(os.listdir("./")))




def start_post(data=None):
    global logger, isCPUonly
    logger.log(f'Start post: {data}')

