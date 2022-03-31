class ValidationConfig:
    default_config_dict = {
        "loading": {
            "add_supplemental": False,
        },
        "preprocessing": {
            "preprocessor_type": "naive",
            "preprocessor_args": {},
        },
        "model": {
            "model_type": "random",
            "model_args": {},
        },
        "validator_args": {
            "n_folds": 5,
            "n_timesteps_per_fold": 100,
        },
    }

    def __init__(self, config_dict):
        self.validate_config_dict(config_dict)
        self.config_dict = self.default_config_dict.copy()

        for k in config_dict:
            self.config_dict[k].update(config_dict[k])

    def validate_config_dict(self, config_dict):

        assert isinstance(config_dict, dict)
        for k in config_dict:
            assert isinstance(config_dict[k], dict)
            assert k in self.default_config_dict, k

        for k in config_dict:
            for sub_k in config_dict[k]:
                assert sub_k in self.default_config_dict[k], sub_k

    def __getitem__(self, item):
        return self.config_dict[item]
