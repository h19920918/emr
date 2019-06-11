from model.fifo import FIFO
from model.lifo import LIFO
from model.lru_dntm import LRU_DNTM
from model.r_emr import R_EMR
from model.t_emr import T_EMR
from model.uniform import UNIFORM


def create_a3c_model(cfg):
    models = dict(
        FIFO=FIFO,
        LIFO=LIFO,
        LRU_DNTM=LRU_DNTM,
        R_EMR=R_EMR,
        T_EMR=T_EMR,
        UNIFORM=UNIFORM,
    )
    model_cls = models.get(cfg.model)
    model = model_cls(cfg=cfg)
    return model
