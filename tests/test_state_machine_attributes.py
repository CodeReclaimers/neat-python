import unittest

from neat.state_machine_attributes import BiasesAttribute, WeightsAttribute
from tests.config_generation import init_config


class TestBiasAttribute(unittest.TestCase):

    def test_init_value(self):
        config = init_config()
        attribute = BiasesAttribute("bias")

        value = attribute.init_value(config)
        self.assertEqual(2, len(value))

    def test_mutate_value(self):
        config = init_config()
        attribute = BiasesAttribute("bias")

        value = attribute.init_value(config)
        self.assertEqual(2, len(value))

        value = attribute.mutate_value(value, config)
        self.assertEqual(2, len(value))


class TestWeightsAttribute(unittest.TestCase):

    def test_init_value(self):
        config = init_config()
        attribute = WeightsAttribute("weights")

        value = attribute.init_value(config)
        self.assertEqual((2, 3), value.shape)

    def test_mutate_value(self):
        config = init_config()
        attribute = WeightsAttribute("weights")

        value = attribute.init_value(config)
        self.assertEqual((2, 3), value.shape)

        value = attribute.mutate_value(value, config)
        self.assertEqual((2, 3), value.shape)


if __name__ == '__main__':
    unittest.main()
