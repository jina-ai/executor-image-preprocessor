# ImagePreprocessor

An executor that performs standard pre-processing and normalization on images.


## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://ImagePreprocessor')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://ImagePreprocessor')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
