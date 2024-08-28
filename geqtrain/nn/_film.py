from typing import List, Optional
from geqtrain.nn import ScalarMLPFunction


class FiLMFunction(ScalarMLPFunction):
    """
        FiLMFunction applies Feature-wise Linear Modulation (FiLM) conditioning to its input.
        
        This class extends the ScalarMLPFunction to perform feature-wise affine transformations
        based on conditioning inputs. It uses a multi-layer perceptron (MLP) to generate scaling 
        and shifting parameters that are applied to the input features.

        Attributes:
            mlp_input_dimension (Optional[int]): The input dimension size for the MLP.
            mlp_latent_dimensions (List[int]): A list of dimensions for the hidden layers of the MLP.
            mlp_output_dimension (Optional[int]): The output dimension size for the MLP.
            mlp_nonlinearity (Optional[str]): The non-linearity activation function to be used in the MLP. Defaults to "silu".
            _dim (int): The dimension size for the output features, which is half of the MLP output dimension.
        
        Parameters for Initialization:
            mlp_input_dimension (Optional[int]): The input dimension size for the MLP.
            mlp_latent_dimensions (List[int]): A list of dimensions for the hidden layers of the MLP.
            mlp_output_dimension (Optional[int]): The output dimension size for the MLP.
            mlp_nonlinearity (Optional[str]): The non-linearity activation function to be used in the MLP. Defaults to "silu".
    """

    def __init__(
        self,
        mlp_input_dimension: Optional[int],
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
    ):
        super().__init__(
            mlp_input_dimension=mlp_input_dimension,
            mlp_latent_dimensions=mlp_latent_dimensions,
            mlp_output_dimension=mlp_output_dimension * 2, # weights + bias
            mlp_nonlinearity=mlp_nonlinearity,
            use_norm_layer=True,
            has_bias=True,
            zero_init_last_layer_weights=True,
        )

        self._dim = mlp_output_dimension

    def forward(self, x, conditioning, batch):
        """
        Applies the FiLM conditioning to the input tensor.

        Args:
            x (Tensor): The input tensor to be modulated.
            conditioning (Tensor): The conditioning input that generates the FiLM parameters.
            batch (int): The batch index to select the corresponding FiLM parameters.

        Returns:
            Tensor: The modulated input tensor.
        """
        
        _wb = self.sequential(conditioning)
        return _wb[batch, :self._dim] * x + _wb[batch, self._dim:]