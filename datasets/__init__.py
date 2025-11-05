# datasets/__init__.py
from .inpainting_dataset import ImageFolderWithMask
from .image_dataset import ImageExtendDataset

__all__ = ['ImageFolderWithMask', 'ImageExtendDataset']
