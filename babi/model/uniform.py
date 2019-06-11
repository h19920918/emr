from model.base import Base


class UNIFORM(Base):
    def mem_forward(self, **batch):
        attn_logit_mvavg = batch['attn_logit_mvavg']
        hidden_state = batch['hidden_state']

        logit = attn_logit_mvavg.new(1, self.cfg.memory_size).fill_(0.0)
        value = logit.new(1, 1).fill_(0.0)
        zeros = logit.new(1, self.memory_size).fill_(0.0)
        return dict(logit=logit,
                    value=value,
                    attn_logit_mvavg=attn_logit_mvavg,
                    hidden_state=hidden_state,
                    )
