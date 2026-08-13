import logging
import signal
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .config import validate_config
from .runner import ApplicationRunner


CONFIG_DIRECTORY = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base="1.3", config_path=CONFIG_DIRECTORY, config_name="config")
def main(config: DictConfig) -> None:
    validate_config(config)
    logging.getLogger().setLevel(str(config.runtime.log_level).upper())
    runner = ApplicationRunner(config, Path(HydraConfig.get().runtime.output_dir))
    signal.signal(signal.SIGINT, runner.stop)
    signal.signal(signal.SIGTERM, runner.stop)
    runner.run()


def entrypoint() -> None:
    main()


if __name__ == "__main__":
    entrypoint()
