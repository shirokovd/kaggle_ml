from learning.trainers import Trainer

if __name__ == '__main__':
    config_file = 'configs/config.yml'
    trainer = Trainer(config_file)
    trainer.train()
