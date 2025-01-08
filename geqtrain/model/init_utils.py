from e3nn import o3
from geqtrain.utils import Config


def update_config(config: Config):
    if "l_max" in config:
        l_max = int(config["l_max"])
        parity_setting = config.get("parity", "o3_full")
        assert parity_setting in ("o3_full", "so3")
        irreps_edge_sh = repr(
            o3.Irreps.spherical_harmonics(
                l_max, p=(1 if parity_setting == "so3" else -1)
            )
        )
        # check consistency
        assert config.get("irreps_edge_sh", irreps_edge_sh) == irreps_edge_sh
        config["irreps_edge_sh"] = irreps_edge_sh