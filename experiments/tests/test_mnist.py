import unittest
import data.mnist as mnist

class TestMNISTLOAD(unittest.TestCase):
    def test_load(self):
        loader = mnist.get_mnist_train_loader()
        for batch_idx, (data, target) in enumerate(loader):
            self.assertEqual(len(data),64)
            break
        loader = mnist.get_mnist_train_loader(batch_size=10)
        for batch_idx, (data, target) in enumerate(loader):
            self.assertEqual(len(data),10)
            break
        loader = mnist.get_mnist_test_loader()
        for batch_idx, (data, target) in enumerate(loader):
            self.assertEqual(len(data),64)
            break
        loader = mnist.get_mnist_test_loader(batch_size=10)
        for batch_idx, (data, target) in enumerate(loader):
            self.assertEqual(len(data),10)
            break
        
if __name__ == '__main__':
    unittest.main()
