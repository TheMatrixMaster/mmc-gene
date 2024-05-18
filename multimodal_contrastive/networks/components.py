from tqdm import tqdm
import numpy as np
from typing import Iterable, List, Optional, Set, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import linear, logsigmoid, relu, softmax
from torch_geometric.nn import (
    GATConv,
    global_mean_pool,
    global_add_pool,
    GINEConv,
    GlobalAttention,
    JumpingKnowledge,
    GraphMultisetTransformer,
    GCNConv,
)
from torch.nn import (
    ModuleList,
    ModuleDict,
    Sequential,
    Linear,
    BatchNorm1d,
    ReLU,
    Dropout,
)

from omegaconf import ListConfig, OmegaConf
from multimodal_contrastive.networks.utils import move_batch_input_to_device
from ..networks.loss import get_loss, MultiTaskLoss


class MultiLayerPerceptron(nn.Module):
    """Standard multi-layer perceptron with non-linearity and potentially dropout.

    Parameters
    ----------
    num_input_features : int hidden_channels
        input dimension
    hidden_layer_dimensions : List[int]
        list of hidden layer dimensions.
    output_size: int
        Number of output classes
    nonlin : Union[str, nn.Module]
        name of a nonlinearity in torch.nn, or a pytorch Module. default is relu
    p_dropout : float
        dropout probability for dropout layers. default is 0.0
    """

    def __init__(
        self,
        num_input_features: int,
        hidden_layer_dimensions: Optional[List[int]],
        output_size: int,
        nonlin: Union[str, nn.Module] = "ReLU",
        p_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(hidden_layer_dimensions, ListConfig):
            hidden_layer_dimensions = OmegaConf.to_object(hidden_layer_dimensions)

        if isinstance(nonlin, str):
            nonlin = getattr(torch.nn, nonlin)()

        hidden_layer_dimensions = [dim for dim in hidden_layer_dimensions if dim != 0]
        layer_inputs = [num_input_features] + hidden_layer_dimensions
        modules = []
        for i in range(len(hidden_layer_dimensions)):
            modules.extend(
                [
                    nn.Linear(layer_inputs[i], layer_inputs[i + 1]),
                    nn.BatchNorm1d(layer_inputs[i + 1]),
                    nonlin,
                    nn.Dropout(p=p_dropout),
                ]
            )

        modules.append(nn.Linear(layer_inputs[-1], output_size))

        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


class CommonEncoder(torch.nn.Module):
    """Class for setting up CommonEncoder for GMC_PL module.

    Parameters
    ----------
    common_dim : dimensions for input features
    hidden_layer_dimensions: dimensions for hidden layers
    latent_dim: dimensions for final embeddings
    """

    def __init__(self, common_dim, hidden_layer_dimensions, latent_dim):
        super().__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        if isinstance(hidden_layer_dimensions, ListConfig):
            hidden_layer_dimensions = OmegaConf.to_object(hidden_layer_dimensions)

        self.feature_extractor = MultiLayerPerceptron(
            num_input_features=self.common_dim,
            hidden_layer_dimensions=hidden_layer_dimensions,
            output_size=self.latent_dim,
        )

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


class JointEncoder(torch.nn.Module):
    """Class for setting up joint encoder for GMC_PL module.

    Parameters
    ----------
    encoders_mod (List[torch.nn.Module]): list of modules for each modality
    """
    def __init__(self, encoders_mod):
        super().__init__()
        self.encoders = ModuleDict()
        if isinstance(encoders_mod, Dict):
            encoders_mod = OmegaConf.to_object(encoders_mod)
        self.encoders.update(encoders_mod)

    def forward(self, x_dict):
        list_emb_mod_ = []
        for mod_name, x in x_dict.items():
            list_emb_mod_.append(self.encoders[mod_name](x))

        return torch.cat(list_emb_mod_, dim=-1)



class GNEpropGIN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        edge_dim,
        hidden_channels,
        ffn_hidden_channels,
        num_layers,
        out_channels,
        num_readout_layers,
        dropout,
        mol_features_size,
        aggr="mean",
        jk="cat",
        gmt_args=None,
        use_proj_head=False,
        proj_dims=(512, 256),
        skip_last_relu=False,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        if num_classes is None:
            self.has_final_layer = False
        else:
            self.has_final_layer = True

        # graph encoderhidden_channels
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(edge_dim, hidden_channels)

        self.convs = ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1 and skip_last_relu:
                mlp = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm1d(2 * hidden_channels),
                    ReLU(inplace=True),
                    Linear(2 * hidden_channels, hidden_channels),
                    BatchNorm1d(hidden_channels),
                    Dropout(p=dropout),
                )
            else:
                mlp = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm1d(2 * hidden_channels),
                    ReLU(inplace=True),
                    Linear(2 * hidden_channels, hidden_channels),
                    BatchNorm1d(hidden_channels),
                    ReLU(inplace=True),
                    Dropout(p=dropout),
                )
            conv = GINEConv(mlp, train_eps=True)
            self.convs.append(conv)

        self.jk_mode = jk
        if self.jk_mode == "none":
            self.jk = None
        else:
            self.jk = JumpingKnowledge(
                mode=self.jk_mode, channels=hidden_channels, num_layers=num_layers
            )

        # classifier
        self.classifier = ModuleList()

        if self.jk_mode == "none":
            hidden_channels_mol = hidden_channels + mol_features_size
        elif self.jk_mode == "cat":
            hidden_channels_mol = hidden_channels * (num_layers + 1) + mol_features_size
        else:
            raise NotImplementedError

        ffn_hidden_size = (
            int(ffn_hidden_channels)
            if ffn_hidden_channels is not None
            else int(hidden_channels_mol)
        )

        for layer in range(num_readout_layers):
            input_dim = int(hidden_channels_mol) if layer == 0 else int(ffn_hidden_size)
            mlp = Sequential(
                Linear(input_dim, ffn_hidden_size),
                BatchNorm1d(ffn_hidden_size),
                ReLU(inplace=True),
                Dropout(p=dropout),
            )
            self.classifier.append(mlp)

        # last layer (classifier)
        input_dim = hidden_channels_mol if num_readout_layers == 0 else ffn_hidden_size
        self.classifier.append(
            Linear(input_dim, int(out_channels)),
        )

        self.aggr = aggr
        self.global_pool = None
        if aggr == "mean":
            self.global_pool = global_mean_pool
        elif aggr == "sum":
            self.global_pool = global_add_pool
        elif aggr == "global_attention":
            hidden_channels_without_mol = hidden_channels_mol - mol_features_size
            hidden_ga_channels = int(hidden_channels_without_mol / 2)
            gate_nn = Sequential(
                Linear(hidden_channels_without_mol, hidden_ga_channels),
                BatchNorm1d(hidden_ga_channels),
                ReLU(inplace=True),
                Linear(hidden_ga_channels, 1),
            )
            self.global_pool = GlobalAttention(gate_nn)
        elif aggr == "gmt":
            assert gmt_args is not None

            gmt_sequences = [
                ["GMPool_I"],
                ["GMPool_G"],
                ["GMPool_G", "GMPool_I"],
                ["GMPool_G", "SelfAtt", "GMPool_I"],
                ["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"],
            ]

            gmt_sequence = gmt_sequences[gmt_args["gmt_sequence"]]

            self.global_pool = GraphMultisetTransformer(
                in_channels=hidden_channels_mol,
                hidden_channels=gmt_args["hidden_channels"],
                out_channels=hidden_channels_mol,
                Conv=GCNConv,
                num_nodes=200,
                pooling_ratio=gmt_args["gmt_pooling_ratio"],
                pool_sequences=gmt_sequence,
                num_heads=gmt_args["gmt_num_heads"],
                layer_norm=gmt_args["gmt_layer_norm"],
            )

        self.use_proj_head = use_proj_head
        self.proj_dims = proj_dims
        if self.use_proj_head and self.proj_dims is not None:
            self.proj_head = ModuleList()

            input_dim = hidden_channels_mol
            for proj_dim in self.proj_dims[:-1]:
                mlp = Sequential(
                    Linear(input_dim, proj_dim),
                    BatchNorm1d(proj_dim),
                    ReLU(inplace=True),
                    Dropout(p=dropout),
                )
                self.proj_head.append(mlp)
                input_dim = proj_dim

            # last proj layer
            self.proj_head.append(
                Linear(input_dim, proj_dims[-1]),
            )

    def compute_representations(self, x, edge_index, edge_attr, batch, perturb=None):
        list_graph_encodings = []

        x_encoded = self.node_encoder(x)

        edge_attr = self.edge_encoder(edge_attr)

        if perturb is not None:
            if "perturb_a" in perturb:
                x_encoded += perturb["perturb_a"]
            if "perturb_b" in perturb:
                edge_attr += perturb["perturb_b"]

        if self.jk_mode != "none":
            list_graph_encodings.append(x_encoded)

        for conv in self.convs:
            x_encoded = conv(x_encoded, edge_index, edge_attr)
            if self.jk_mode != "none":
                list_graph_encodings.append(x_encoded)

        if self.jk_mode != "none":
            x_encoded = self.jk(list_graph_encodings)
        # x_encoded = torch.stack(list_graph_encodings, dim=1)  # [num_nodes, num_layers, num_channels] # for dnaconv
        # x_encoded = F.relu(self.dna(x_encoded, edge_index)) # for dnaconv

        if self.aggr in ["gmt"]:
            out = self.global_pool(x_encoded, batch, edge_index)
        else:
            out = self.global_pool(x_encoded, batch)  # [batch_size, hidden_channels]

        if perturb is not None:
            if "perturb_graph" in perturb:
                out += perturb["perturb_graph"]
        return out

    def forward(
        self,
        x_struct,
        mol_features=None,
        restrict_output_layers=0,
        output_type=None,
        perturb=None,
        **kwargs
    ):
        x = x_struct.x
        edge_index = x_struct.edge_index
        edge_attr = x_struct.edge_attr
        batch = x_struct.batch

        # compute graph emb
        out_repr = self.compute_representations(
            x, edge_index, edge_attr, batch, perturb=perturb
        )

        if mol_features is not None:
            out_repr = torch.cat((out_repr, mol_features), dim=1)

        # compute classifier
        out = out_repr

        if self.has_final_layer:
            for mlp in self.classifier[
                : None if restrict_output_layers == 0 else restrict_output_layers
            ]:
                out = mlp(out)

            # (optionally) compute proj head
            if self.use_proj_head:
                if self.proj_dims is None:
                    out_proj = out_repr[-500:]
                else:
                    out_proj = out_repr
                    for mlp in self.proj_head:
                        out_proj = mlp(out_proj)

            if output_type is not None:
                if output_type == "prob":
                    out = torch.sigmoid(out)
                elif output_type == "log_prob":
                    out = torch.nn.functional.logsigmoid(out)
                elif output_type == "prob_multiclass":
                    out = torch.sigmoid(out)
                    return torch.cat((1 - out, out), dim=-1)
                elif output_type == "log_prob_multiclass":
                    return torch.cat(
                        (logsigmoid(-out), logsigmoid(out)), dim=-1
                    )  # equivalent to log(1-sigmoid(x)), log(sigmoid(x))
                else:
                    raise NotImplementedError

        if self.use_proj_head:
            return out, out_proj
        else:
            return out

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for mlp in self.classifier:
            mlp.reset_parameters()
        self.input_batch_norm.reset_parameters()


class MultiTask_model(torch.nn.Module):
    def __init__(
        self, backbone, loss_name, num_tasks, mod_name, lr=0.001, freeze_backbone=True
    ):
        super().__init__()
        self.backbone = backbone
        self.optimizer = None
        input_dim = self.backbone.hparams.latent_dim
        self.linear_layer = nn.Linear(input_dim, num_tasks)

        self.loss = get_loss(name=loss_name)
        self.mod_name = mod_name

        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone is True:
            self.optimizer = torch.optim.Adam(self.linear_layer.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, batch):
        probs = self._forward_with_sigmoid(batch, device="cuda", return_mod=None)
        return probs

    def _forward_with_sigmoid(
        self, batch, device="cuda", mod_name=None, return_mod=None
    ):
        label = batch["labels"]
        label = label.to(device)
        x_dict = batch["inputs"]
        x_dict = move_batch_input_to_device(x_dict, device=device)

        #TODO gen reprs with encoder

        if self.freeze_backbone is True:
            self.backbone.freeze()
        shared_reprs = self.backbone.compute_representations(x_dict, mod_name=mod_name)

        logits = self.linear_layer(shared_reprs)
        probs = torch.sigmoid(logits)

        if return_mod == "label":
            return probs, label
        elif return_mod == "logits":
            return probs, logits
        elif return_mod is None:
            return probs

    def predict_probs_dataloader(
        self,
        dataloader,
        device="cuda",
        return_mod="label",
        mod_name=None,
        disable_progress_bar=False,
        return_mol=False,
    ):
        probs_list = []
        label_list = []
        mols = []
        with torch.no_grad():
            for batch in tqdm(
                dataloader, position=0, leave=True, disable=disable_progress_bar
            ):
                probs, label = self._forward_with_sigmoid(
                    batch, device=device, mod_name=mod_name, return_mod=return_mod
                )
                probs_list.append(probs.cpu().data.numpy())
                label_list.append(label.cpu().data.numpy())
                if return_mol:
                    mols.extend([*batch["inputs"]["struct"].mols])
        if return_mol:
            return np.vstack(probs_list), np.vstack(label_list), mols
        else:
            return np.vstack(probs_list), np.vstack(label_list)


class FP_MLP(torch.nn.Module):
    def __init__(self, input_dim=2048, hidden_layer_dimensions=[64, 64], output_size=270):
        super().__init__()
        self.input_dim = input_dim
        if isinstance(hidden_layer_dimensions, ListConfig):
            hidden_layer_dimensions = OmegaConf.to_object(hidden_layer_dimensions)

        self.mlp = MultiLayerPerceptron(
            num_input_features=self.input_dim,
            hidden_layer_dimensions=hidden_layer_dimensions,
            output_size=output_size,
        )

    def forward(self, batch, device="cuda", return_mod="logits", out_act="sigmoid"):
        label = batch["labels"]
        label = label.to(device)
        x_dict = batch["inputs"]
        x_dict = move_batch_input_to_device(x_dict, device=device)

        logits = self.mlp(x_dict['struct'])

        if out_act == "sigmoid":
            probs = torch.sigmoid(logits)
        elif out_act == "softmax":
            probs = torch.softmax(logits, dim=-1)

        if return_mod == "label":
            return probs, label
        elif return_mod == "logits":
            return probs, logits
        elif return_mod is None:
            return probs
        