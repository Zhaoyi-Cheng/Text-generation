from unittest import TestCase


class InstallationTestCase(TestCase):

    def test_pytorch(self):
        import torch
        torch_version = torch.__version__
        self.assertIsNotNone(torch_version)
        print(f"Installed PyTorch version: {torch_version} (this assignment was created with version 2.2.2)")

    def test_transformers(self):
        from transformers import __version__
        self.assertIsNotNone(__version__)
        print(f"Installed transformers version: {__version__} (this assignment was created with version 4.39.3)")
