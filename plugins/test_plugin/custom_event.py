logger = setupData["logger"]

def custom_event_fn(data=None):
    global logger
    logger.log(f'custom_event_fn: {data}')
