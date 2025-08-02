def release_all():
    """简单释放内存和资源"""
    import gc

    # 关闭所有matplotlib图形
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except:
        pass

    # 关闭日志
    try:
        import logging
        logging.shutdown()
    except:
        pass

    # 清空全局变量
    globals().clear()

    # 强制垃圾回收
    gc.collect()

    # Linux下尽量把内存还给操作系统
    try:
        import sys, ctypes
        if sys.platform.startswith('linux'):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass
