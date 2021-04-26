import argparse
import hydra
import matplotlib.pyplot as plt
import numpy as np
import yaml
import logging

from image_constructor import ImageConstructor

CONFIG_PATH = "config.yaml"

@hydra.main(config_path=CONFIG_PATH, strict=False)
def main(cfg):
  logging.basicConfig(filename='gpr_processing.log', encoding='utf-8', level=logging.INFO)

  logging.info(cfg.pretty())

  # if config.extract_training_images:
  tic = ImageConstructor(cfg)

  if cfg.radargram.create:
    tic.create_radargram()

  if cfg.submaps.create:
    tic.create_submaps()

  if cfg.visualize.show:
    plt.show()

if __name__ == "__main__":
  main()