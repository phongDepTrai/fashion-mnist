import datetime
import tensorflow as tf
from tensorflow import keras
from models import mlp, lenet, alexnet, restnet, googlenet
from datasets.dataset import DataLoader


class Test(object):
    def __init__(self, model_type='mlp', batch_size=32, data_path='./', start_epoch=0):
        self.is_reshape = True
        self.model_type = model_type
        self.model = self.get_model()
        self.batch_size = batch_size

        self.test_dataset = DataLoader(
            data_path=data_path,
            is_reshape=self.is_reshape,
            batch_size=self.batch_size)

        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        self.acc = keras.metrics.SparseCategoricalAccuracy()
        self.acc_tracker = keras.metrics.Mean()
        # load pretrained weight:
        self.start_epoch = start_epoch
        self.load_weights = True
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(self.start_epoch + 1, name="step"), optimizer=self.optimizer,
                                        net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts/' + self.model_type, max_to_keep=1)
        if self.load_weights:
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Loading weights from {}".format(self.manager.latest_checkpoint))
            else:
                print("Loading weights failed. Initializing from scratch.")

    def get_model(self):
        if self.model_type == 'mlp':
            self.is_reshape = False
            inputs = keras.Input(shape=(784,), name="fashion_mlp")
            return mlp.create_model(inputs, name=self.model_type)
        elif self.model_type == 'mlp_2':
            self.is_reshape = False
            inputs = keras.Input(shape=(784,), name="fashion_mlp_2")
            return mlp.create_model_2(inputs, name=self.model_type)
        elif self.model_type == 'mlp_sm':
            self.is_reshape = False
            inputs = keras.Input(shape=(784,), name="fashion_mlp_sm")
            return mlp.create_model_sm(inputs, name=self.model_type)

        elif self.model_type == 'lenet' or self.model_type == 'lenet-64':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_lenet")
            return lenet.create_model(inputs)
        elif self.model_type == 'lenet-dropout':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_lenet_dropout")
            return lenet.create_model_do(inputs)
        elif self.model_type == 'lenet-bn-8' or self.model_type == 'lenet-bn-64':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_bn")
            return lenet.create_model_bn(inputs)

        elif self.model_type == 'alexnet':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion-alexnet")
            return alexnet.create_model(inputs, name=self.model_type)
        elif self.model_type == 'alexnet_2':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_alexnet_2")
            return alexnet.create_model_2(inputs, name=self.model_type)

        elif self.model_type == 'resnet':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_restnet")
            return restnet.create_model(inputs, name=self.model_type)
        elif self.model_type == 'resnet_2':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_restnet_2")
            return restnet.create_model_2(inputs, name=self.model_type)
        elif self.model_type == 'resnet_3':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_restnet_3")
            return restnet.create_model_3(inputs, name=self.model_type)

        elif self.model_type == 'googlenet':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_googlenet")
            return googlenet.create_model(inputs, name=self.model_type)
        elif self.model_type == 'googlenet_2':
            self.is_reshape = True
            inputs = keras.Input(shape=(28, 28, 1), name="fashion_googlenet_2")
            return googlenet.create_model_2(inputs, name=self.model_type)

    def test(self):
        for i, (x_batch, y_batch) in enumerate(self.test_dataset):
            predicts = self.model(x_batch)

            self.acc.update_state(y_batch, predicts)
            self.acc_tracker.update_state(self.acc.result())

        return self.acc_tracker.result()
