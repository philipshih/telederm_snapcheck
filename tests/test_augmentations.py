import numpy as np
from PIL import Image

from snapcheck.augmentations import QualityAugmentor


def test_quality_augmentor_produces_labels():
    rng = np.random.default_rng(0)
    image = Image.fromarray((rng.uniform(0, 255, (224, 224, 3))).astype("uint8"))
    config = {
        "quality_defects": {
            "blur": {"kernel_sizes": [3], "sigma_range": [1.0, 1.0]},
            "noise": {"std_range": [0.05, 0.05]},
        },
        "max_augmentations_per_image": 2,
    }
    augmentor = QualityAugmentor(config, seed=123)
    result = augmentor.apply(image)
    assert result.labels["blur"] == 1 or result.labels["noise"] == 1
    assert result.labels["overall_fail"] == 1
    assert result.image.size == image.size
