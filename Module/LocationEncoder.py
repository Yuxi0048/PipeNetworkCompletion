import torch
import math
"""
Author: Gengchen Mai
GitHub Repository: https://github.com/gengchenmai/space2vec

If you use this code, please cite the following paper:

@inproceedings{space2vec_iclr2020,
    title={Multi-Scale Representation Learning for Spatial Feature Distributions using Grid Cells},
    author={Mai, Gengchen and Janowicz, Krzysztof and Yan, Bo and Zhu, Rui and Cai, Ling and Lao, Ni},
    booktitle={The Eighth International Conference on Learning Representations},
    year={2020},
    organization={openreview}
}
"""

def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        freq_list = torch.rand([frequency_num]) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) /
          (frequency_num*1.0 - 1))

        timescales = min_radius * torch.exp(
            torch.arange(frequency_num).float() * log_timescale_increment)

        freq_list = 1.0/timescales

    return freq_list

class TheoryGridCellSpatialRelationEncoder(torch.nn.Module):
    def __init__(self, spa_embed_dim, device, coord_dim = 2, frequency_num = 16,
        max_radius = 10000,  min_radius = 1000, freq_init = "geometric", ffn = None):
        super(TheoryGridCellSpatialRelationEncoder, self).__init__()
        self.device = device
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.spa_embed_dim = spa_embed_dim
        self.freq_init = freq_init

        self.cal_freq_list()
        self.cal_freq_mat()

        self.unit_vec1 = torch.tensor([1.0, 0.0]).to(self.device)
        self.unit_vec2 = torch.tensor([-1.0/2.0, math.sqrt(3)/2.0]).to(self.device)
        self.unit_vec3 = torch.tensor([-1.0/2.0, -math.sqrt(3)/2.0]).to(self.device)

        self.input_embed_dim = self.cal_input_dim()
        self.ffn = ffn
        

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        freq_mat = torch.unsqueeze(self.freq_list, 1).to(self.device)
        self.freq_mat = torch.repeat_interleave(freq_mat, 6, dim=1).to(self.device)

    def cal_input_dim(self):
        return int(6 * self.frequency_num)

    def make_input_embeds(self, coords):
        batch_size = coords.shape[0]
        num_context_pt = coords.shape[1]

        angle_mat1 = torch.unsqueeze(torch.matmul(coords, self.unit_vec1), -1)
        angle_mat2 = torch.unsqueeze(torch.matmul(coords, self.unit_vec2), -1)
        angle_mat3 = torch.unsqueeze(torch.matmul(coords, self.unit_vec3), -1)

        angle_mat = torch.cat([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], dim=-1)
        angle_mat = torch.unsqueeze(angle_mat, -2)
        angle_mat = angle_mat.repeat(1, 1, self.frequency_num, 1)
        angle_mat = angle_mat * self.freq_mat
        spr_embeds = torch.reshape(angle_mat, (batch_size, num_context_pt, -1))

        spr_embeds[:, :, 0::2] = torch.sin(spr_embeds[:, :, 0::2].clone())
        spr_embeds[:, :, 1::2] = torch.cos(spr_embeds[:, :, 1::2].clone())

        return spr_embeds

    def forward(self, coords):
        spr_embeds = self.make_input_embeds(coords)
        spr_embeds = spr_embeds

        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds

