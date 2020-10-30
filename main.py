from tools.neural import Neural
from tools.test import Test
from pathlib import Path
import os

if __name__ == "__main__":
    data_train = 'train.csv'
    data_test = 'test.csv'
    model_type='resnet'
    base_dir = Path(__file__).resolve().parent.parent

    data_path = os.path.join(base_dir, 'data', data_train)
    trainer = Neural(model_type=model_type, batch_size=32, epochs=20, data_path=data_path)
    trainer.train()

    test_epoch = 20
    data_path = os.path.join(base_dir, 'data', data_test)
    tester = Test(model_type=model_type, batch_size=100, data_path=data_path, start_epoch=test_epoch)
    result = tester.test()
    print('Test accuracy: {}'.format(result))