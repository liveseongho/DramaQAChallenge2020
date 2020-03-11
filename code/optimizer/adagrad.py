from torch import optim


class Adagrad(optim.Adagrad):
    @classmethod
    def resolve_args(cls, args, params):
        options = {}
        options['lr'] = args.get("learning_rate", 0.01)
        options['lr_decay'] = args.get("lr_decay", 0)
        options['weight_decay'] = args.get("weight_decay", 0)

        return cls(params, **options)

class Adam(optim.Adam):
    @classmethod
    def resolve_args(cls,args,params):
        options = {}
        options['lr'] = args.get("learning_rate", 3e-4)
        options['weight_decay'] = args.get("weigh_decay", 1e-5)
        return cls(params, **options)