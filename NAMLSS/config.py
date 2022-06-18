from nam.config.base import Config


def defaults():
    config = Config(
        num_epochs=100,
        lr=0.001,
        feature_dropout=0.1,
        dropout=0.1,
        batch_size=512,
        name_scope="model",
        decay_rate=0.995,
        num_basis_functions=1000,
        units_multiplier=2,
        activation="exu",
        shallow=False,
        output_regularization1=0.1,
        output_regularization2=0.1,
        l2_regularization1=0.1,
        l2_regularization2=0.1,
        logdir="output",
        wandb=False,
        save_frequency=5,
        early_stopping_patience=10,
    )
    return config
