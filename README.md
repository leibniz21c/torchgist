# torchgist

Python reimplementation of "Modeling the shape of the scene: a holistic representation of the spatial envelope" using pytorch.

# Installation

```
$ pip install torchgist
```

# Usage

```python
from torchgist.models import GIST

model = GIST(
    orientations_per_scale: Tuple = (8, 8, 8, 8), 
    num_blocks=4, 
    fc_prefilt=4,
    image_size: Tuple[int, int] = (256, 256),
    boundary_extension=32
)
```

# TODO

1. Process-based parallel processing
2. CUDA-based parallel processing
