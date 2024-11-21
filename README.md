# utils_gradients

### Examples

To install the package:
```
pip install git+https://github.com/arekavandi/utils_gradients.git
```

To use
```
from RobustPCA.rpca import RobustPCA
from RobustPCA.spcp import StablePCP

rpca = RobustPCA()
spcp = StablePCP()

rpca.fit(M)
L = rpca.get_low_rank()
S = rpca.get_sparse()

spcp.fit(M)
L = spcp.get_low_rank()
S = spcp.get_sparse()
```
