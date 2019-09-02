from model.lru import LRU
from model.fifo import FIFO
from model.t_emr import T_EMR
from model.r_emr import R_EMR

def create_model(cfg):
    models=dict(
        FIFO=FIFO,
		LIFO=FIFO,
		UNIFORM=FIFO,
        T_EMR=T_EMR,
        R_EMR=R_EMR,
        LRU_DNTM=LRU,
    )
    model_cls = models.get(cfg.model)
    model = model_cls(cfg)

    return model
