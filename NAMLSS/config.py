from nam.config.base import Config


def defaults():
    config = Config(
        num_epochs=100,
        lr=0.01,
        feature_dropout=0.1,
        dropout=0.1,
        batch_size=512,
        name_scope="model",
        weight_decay=0.995,
        num_basis_functions=1000,
        units_multiplier=2,
        activation="exu",
        shallow=False
    )
    return config
