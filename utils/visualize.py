import matplotlib.pyplot as plt
from datasets.dataset import DataLoader


class Visualization(object):
    def __init__(self, images, labels):
        self.label_ann = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                          'Ankle boot']
        self.images = images
        self.labels = labels
        self.num_col = 5
        self.num_row = images.shape[0] // self.num_col
        self.fig_size = (1.5 * self.num_col, 2 * self.num_row)
        self.num_samples = self.num_col * self.num_row

    def plt_draw(self):
        figure, axes = plt.subplots(self.num_row, self.num_col, figsize=self.fig_size)
        for i in range(self.num_samples):
            ax = axes[i // self.num_col, i % self.num_col]
            ax.imshow(self.images[i], cmap='gray')
            ax.set_title('{}'.format(self.label_ann[self.labels[i]]))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    fashion_mnist = DataLoader(data_path='D:\InterShip\Other_project\\nam_4\data\\fashion_mnist\\fashion-mnist_train.csv',
                               split_factor=0.2)
    x_train, y_train, x_valid, y_valid = fashion_mnist.data

    visualize = Visualization(x_valid[20:30], y_valid[20:30])
    visualize.plt_draw()
