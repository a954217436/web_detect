import logging
import time

def init_log(output_dir):
    if not os.path.exists(output_dir):
        os.mkdirs(output_dir)
    log_name = '{}.log'.format(time.strftime('%Y%m%d_%H%M%S'))

    logging.basicConfig(level = logging.DEBUG,
                        format = '%(asctime)s %(message)s',
                        datefmt = '%Y-%m-%d %H-%M-%S',
                        filename = os.path.join(output_dir, log_name),
                        filemode = 'w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging


log_dir = 'E:/DL/web_detect/log/'
log = init_log('E:/DL/web_detect/log/')
_print = log.info
_print(log_dir)

