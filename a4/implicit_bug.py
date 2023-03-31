import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import autograd

from ray_utils import RayBundle

from a4.lighting_functions import relighting_dict

# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)


# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)


sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)



class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            # layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
            layers.append(torch.nn.Sequential(linear, torch.nn.ELU(True)))


        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # TODO (Q2): Implement Neural Surface MLP to output per-point SDF

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim

        self.volsdf_mlp = MLPWithInputSkips(n_layers=8,
                          input_dim=embedding_dim_xyz,
                          output_dim=256,
                          skip_dim=embedding_dim_xyz,
                          hidden_dim=256,
                          input_skips=[4])

        self.dist_fc = torch.nn.Linear(256, 1)
        self.feature_fc = torch.nn.Linear(256, 256)


        # TODO (Q3): Implement Neural Surface MLP to output per-point color

        self.radiance_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )

        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        self.volsdf_mlp.apply(weight_init)
        self.radiance_mlp.apply(weight_init)

        nn.init.xavier_normal_(self.dist_fc.weight.data)
        nn.init.zeros_(self.dist_fc.bias.data)
        nn.init.xavier_normal_(self.feature_fc.weight.data)
        nn.init.zeros_(self.feature_fc.bias.data)

        # nn.init.kaiming_normal_(self.dist_fc.weight.data)
        # nn.init.kaiming_normal_(self.feature_fc.weight.data)

    def get_3Dfeatures(self, points):
        points = points.view(-1, 3)
        ip_emb = self.harmonic_embedding_xyz(points)
        encoding = self.volsdf_mlp(ip_emb, ip_emb)
        return encoding

    def get_distance(
        self,
        encoding
    ):
        '''
        TODO: Q2
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        # points = points.view(-1, 3)
        # ip_emb = self.harmonic_embedding_xyz(points)
        # encoding = self.volsdf_mlp(ip_emb, ip_emb)
        dist = self.dist_fc(encoding)
        return dist

    
    def get_color(
        self,
        encoding
    ):
        '''
        TODO: Q3
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        features = self.feature_fc(encoding)
        color = self.radiance_mlp(features)
        return color


    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q3
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''

        encoding = self.get_3Dfeatures(points)
        dist = self.get_distance(encoding)
        color = self.get_color(encoding)
        
        return dist,color
        
    def forward(self, points):
        dist, color = self.get_distance_color(points)

        out = {
                'dist': dist,
                'color': color 
            }
        return out
    
    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            # distance = self.get_distance(points)
            distance, color = self.get_distance_color(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient

    def get_surface_normal(
        self,
        points
    ):
        '''
        TODO: Q4
        Input:
            points: N X 3 Tensor, where N is number of input points
        Output:
            surface_normal: N X 3 Tensor, where N is number of input points
        '''
        pass


implicit_dict = {
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
}
