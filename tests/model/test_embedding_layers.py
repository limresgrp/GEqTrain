from geqtrain.model._embedding import buildEmbeddingLayers
from geqtrain.utils.config import Config


def test_build_embedding_layers_graph_attributes():
    config = Config.from_dict(
        {
            "graph_attributes": {
                "graph_cat": {
                    "attribute_type": "categorical",
                    "embedding_mode": "one_hot",
                    "num_types": 2,
                }
            }
        }
    )

    layers = buildEmbeddingLayers(config)
    assert "graph_input_attrs" in layers


def test_build_embedding_layers_graph_eq_attributes():
    config = Config.from_dict(
        {
            "eq_graph_attributes": {
                "graph_eq": {
                    "attribute_type": "numerical",
                    "irreps": "1x1o",
                    "embedding_dimensionality": 3,
                }
            }
        }
    )

    layers = buildEmbeddingLayers(config)
    assert "graph_input_attrs" in layers
