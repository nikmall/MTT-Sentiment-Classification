import optuna


def show_best_results(study):

    print("Best params: ", study.best_params)
    print("Best f1-score: ", study.best_value)
    print("Best Trial: ", study.best_trial)


study_name = 'study_mtt_fuse'
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0), direction="maximize",
                            study_name=study_name, storage=storage_name, load_if_exists=True)

show_best_results(study)