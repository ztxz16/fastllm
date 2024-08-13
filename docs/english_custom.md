### Custom Models

For models that are not supported by the Fastllm framework, you can support them by customizing the model structure.

A custom Python model requires only a Python file to describe the model structure. You can refer to the implementation in [QWEN](../example/python/qwen2.py).

### Using Python Custom Models

When using `ftllm.chat`, `ftllm.webui`, or `ftllm.server`, you can add the `--custom` parameter to specify the custom model file.

Assuming our model is located in the `~/Qwen2-7B-Instruct/` directory and the custom model is located in `~/qwen2.py`, you can use the command:

```sh
python3 -m ftllm.chat -t 16 -p ~/Qwen2-7B-Instruct/ --custom ~/qwen2.py
```

to load the Qwen2 model using the custom model file. The usage for `server` and `webui` is similar.

### Writing Python Custom Models

When creating a custom model, you need to implement a model description class that inherits from `ftllm.llm.ComputeGraph`.

Refer to the code in [QWEN](../example/python/qwen2.py):

```python
from ftllm.llm import ComputeGraph
class Qwen2Model(ComputeGraph):
```

At the end of the file, you need to define the `__model__` variable to specify the class corresponding to the custom model structure, with the corresponding code:

```python
__model__ = Qwen2Model
```

The model description class needs to implement the `build` method to obtain model parameters and describe the computation flow.

Here is an example based on the sample code:

```python
class Qwen2Model(ComputeGraph):
    def build(self):
        # 1. Get weight, data, config
        weight, data, config = self.weight, self.data, self.config

        # 2. Set some config
        config["max_positions"] = 128000

        # 3. Describe the computation flow
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        self.Embedding(data["inputIds"], weight["model.embed_tokens.weight"], data["hiddenStates"]);
        # The following is the computation flow, see the example code for details
```

#### `self.config`

The model configuration, which by default is read from the `config.json` file in the model folder.

You can modify parameters in the `config` within the `build` method, such as changing `max_positions` to modify the context length.

For some models, the variable names used in `config.json` may differ and need to be manually assigned during the `build` process.

For example, in the TeleChat7B model configuration, there is no `max_positions` variable but instead uses `seq_length` to represent the length. In the `build` method, you need to assign it with the following code:

```python
self.config["max_positions"] = self.config["seq_length"]
```

In the `config`, the following variables must be assigned (if the variable names in `config.json` are consistent, no action is needed):

```python
self.config["max_positions"] # Represents the maximum context length
```

#### `self.weight`

Represents the weight data.

`self.weight[weightName]` represents the parameter named `weightName` in the model file (corresponding to the parameter names in the `.safetensors` file in the HF model folder).

#### `self.data`

Represents the intermediate variables and input variables of the computation flow.

`self.data[dataName]` represents the intermediate variable named `dataName`. `dataName` can be any string except for the following input variable names:

Input variables:

```python
data["inputIds"] # Input tokens
data["positionIds"] # Position information
data["attentionMask"] # Mask information
data["sin"] # Sin for rotary encoding
data["cos"] # Cos for rotary encoding
data["atype"] # Data type in inference
data["pastKey."][i] # Key cache for the i-th block
data["pastValue."][i] # Value cache for the i-th block
```

#### Computation Flow and Operators

Use the functions of the base class `ComputeGraph` to describe the computation flow.

The currently supported operators are documented in [Custom Model Operators](./custom_op.md).

### Custom Models in C++

(The interface for custom models in C++ is still under modification...)