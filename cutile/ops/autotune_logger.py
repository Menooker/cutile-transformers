import logging
import cuda.tile_experimental as ct_experimental


def setup_autotune_logger():
    """
    配置日志系统，只显示 ct_experimental._autotuner 的 DEBUG 信息。
    其他模块的日志保持 INFO 级别。
    """
    # 设置根日志级别为 DEBUG（这样所有 DEBUG 信息都能传递到 handler）
    logging.basicConfig(
        level=logging.DEBUG,
    )
    
    # 设置根 logger 的级别为 INFO（默认情况下只显示 INFO 及以上）
    logging.getLogger().setLevel(logging.INFO)
    
    # 单独为 ct_experimental._autotuner 设置 DEBUG 级别
    ct_experimental._autotuner.logger.setLevel(logging.DEBUG)
