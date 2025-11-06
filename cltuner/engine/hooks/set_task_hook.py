from mmengine.hooks import Hook


class SetTaskHook(Hook):
    def before_train(self, runner) -> None:
        runner.model.cur_task = runner.cur_task

    def before_val(self, runner) -> None:
        runner.model.cur_task = runner.cur_task

    def before_test(self, runner) -> None:
        runner.model.cur_task = runner.cur_task

