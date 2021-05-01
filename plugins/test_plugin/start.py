logger = setupData["logger"]
isCPUonly = setupData["isCPUonly"]


def start_pre(data=None):
    global logger, isCPUonly
    logger.log(f'Start pre: {data} (isCPUonly: {isCPUonly})')


def start_post(data=None):
    global logger, isCPUonly
    logger.log(f'Start post: {data}')


register_function(start_pre)
register_function(start_post)