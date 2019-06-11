from model.fifo import FIFO
from model.lifo import LIFO
from model.lru_dntm import LRU_DNTM
from model.r_emr import R_EMR
from model.t_emr import T_EMR
from model.uniform import UNIFORM


def create_a3c_model(cfg, vocab, stats):
    models = dict(
        FIFO=FIFO,
        LIFO=LIFO,
        LRU_DNTM=LRU_DNTM,
        R_EMR=R_EMR,
        T_EMR=T_EMR,
        UNIFORM=UNIFORM,
    )
    model_cls = models.get(cfg.model)
    max_ctx_len = cfg.memory_size
    model = model_cls(cfg=cfg,
                      num_hops=cfg.num_hops,
                      memory_size=cfg.memory_size,
                      max_ctx_len=max_ctx_len,
                      max_sent_len=stats['max_sent_len'],
                      memory_dim=cfg.memory_dim,
                      vocab_size=len(vocab))

    return model
