import numpy as np
import struct
import matplotlib.pyplot as plt
from typing import Tuple, List
from numpy.typing import NDArray

class MnistData:
    """Contains all MNIST images and labels."""
    
    DIGIT_COUNT = 10
    
    def __init__(self, images_path: str, labels_path: str) -> None:
        self._read_images(images_path)
        self._read_labels(labels_path)

    def get_split_data(self, validation_percent: float = 0.2, seed: int = 42) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
        if not 0.0 < validation_percent < 1.0:
            raise ValueError("val_ratio must be between 0 and 1.")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.size)
        split = int(self.size * (1 - validation_percent))
        training_idx, validation_idx = indices[:split].tolist(), indices[split:].tolist()

        training = self.__subset(training_idx)
        validation = self.__subset(validation_idx)
        return training.get_data(), validation.get_data()

    def get_data(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        labels = [label.reshape(-1, 1) for label in self.labels]
        images = [image.flatten().reshape(-1, 1) for image in self.images]
        return list(zip(images, labels))

    @staticmethod
    def show_image(data: Tuple[np.ndarray, np.ndarray]) -> None:
        image, label = data
        """Displays a single MNIST image from a (784, 1) vector."""
        if image.shape != (784, 1):
            raise ValueError("Image must be a (784, 1) column vector.")
        
        image_reshaped = image.reshape(28, 28)

        plt.imshow(image_reshaped, cmap='gray')
        if label is not None:
            plt.title(f"Label: {label.argmax()}")
        plt.axis('off')
        plt.show()

    def _read_images(self, path: str) -> None:
        with open(path, 'rb') as f:
            header = struct.unpack('>IIII', f.read(16))
            magic_number, self.size, self.img_rows, self.img_cols = header

            if magic_number != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic_number}')

            raw_images = f.read()
        
        images = np.frombuffer(raw_images, dtype=np.uint8)
        self.images = images.astype(np.float32).reshape(self.size, self.img_rows, self.img_cols) / 255.0

    def _read_labels(self, path: str) -> None:
        with open(path, 'rb') as f:
            header = struct.unpack('>II', f.read(8))
            magic_number, total_labels = header

            if magic_number != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic_number}')
            if self.size != total_labels:
                raise ValueError(f'Number of images does not match number of labels. There are {self.size} images and {total_labels} labels.')

            raw_labels = f.read()
        
        labels = np.frombuffer(raw_labels, dtype=np.uint8)
        self.labels = np.eye(self.DIGIT_COUNT)[labels]

    def __subset(self, indices: List[int]) -> 'MnistData':
        data = object.__new__(MnistData)
        
        data.size = len(indices)
        data.img_rows = self.img_rows
        data.img_cols = self.img_cols
        data.images = self.images[indices]
        data.labels = self.labels[indices]

        return data