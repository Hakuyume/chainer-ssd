import chainer.functions as F
from chainer.training import ParallelUpdater
from chainer import Variable


class CustomUpdater(ParallelUpdater):

    def __init__(self, iterator, optimizer, devices=None, loss_func=None):
        super().__init__(
            iterator=iterator, optimizer=optimizer, devices=devices)
        self.loss_func = loss_func

    def update_core(self):
        optimizer = self.get_optimizer('main')
        model_main = optimizer.target
        models_others = {
            k: v for k, v in self._models.items()
            if v is not model_main}

        batch = self.get_iterator('main').next()

        n = len(self._models)
        in_arrays_list = {}
        for i, key in enumerate(self._models.keys()):
            in_arrays_list[key] = self.converter(
                batch[i::n], self._devices[key])

        for model in self._models.values():
            model.cleargrads()

        hs = None
        for model_key, model in self._models.items():
            in_arrays = in_arrays_list[model_key]
            in_vars = tuple(Variable(x) for x in in_arrays)
            out_vars = model(*in_vars)
            if hs is None:
                hs = [list() for _ in range(len(out_vars))]
            for i, out_var in enumerate(out_vars):
                hs[i].append(F.copy(out_var, self._devices['main']))

        loss = self.loss_func(*(F.concat(h, axis=0) for h in hs))
        loss.backward()

        for model in models_others.values():
            model_main.addgrads(model)

        optimizer.update()

        for model in models_others.values():
            model.copyparams(model_main)
