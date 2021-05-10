logger = setupData["logger"]

import time

def custom_event_fn(data=None):
    global logger, time
    print(f'custom_event_fn: {data}')
    logger.log(f'custom_event_fn: {data}')
    time.sleep(2)

