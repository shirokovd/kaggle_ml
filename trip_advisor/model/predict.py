from learning.predictors import Predictor

if __name__ == '__main__':
    config_file = 'configs/config.yml'
    trainer = Predictor(config_file)
    trainer.predict()
