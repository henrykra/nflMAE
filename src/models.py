import torch
import torch.nn as nn

"""
Learnings: 

remove the final pooling layer from the backbone, ssl uses an embedding for each point
each points embedding is mapped to a prototype vector, which is weight normalized (only direction)

The student and the teacher models have their own online cluster heads which do this mapping. The cluster heads' parameters are updated in the same way the student and teacher models are
The prototype is the mapping between the embedding dimension and the num_prototypes dimension. The num_prototypes dimension is weight normalized, so can be thought of a probability distribution
The simliarty (to each cluster centroid) is the value for each entry in n_protoypes, and that similiary is determined by the online cluster head

I probably won't have to worry about feature upcasting to get multi-scale feautres since there is no intermediate pooling in the model
I also won't need to do the k-means matching for that same reason, I can match each players easily



To fit into applying the SSL wrapper, it might be best to clean up the model a little bit, and write some more helper classes

sonata handles masking by making a change in the embedding layer, so the masked points map to a mask_token
"""




class SslWrapper(nn.Module): 
    def __init__(
            self,
            cfg: dict,
    ): 
        super().__init__()

        # self.mask_loss_weight = mask_loss_weight
        # self.roll_mask_loss_weight = roll_mask_loss_weight
        # self.unmask_loss_weight = unmask_loss_weight

        # self.num_global_view = num_global_view
        # self.num_local_view = num_local_view

        # # masking and scheduler

        # masked point jitter
        self.mask_jitter = cfg["mask_jitter"]

        # # temperature and scheduler
        self.student_temp = cfg["student_temp"]


        # # dynamic matching
        # self.match_max_k = match_max_k
        # self.match_max_r = match_max_r

        # # up cast level
        # self.up_cast_level = up_cast_level

        # one of unmask, mask, roll mask loss enable
        assert unmask_loss_weight + mask_loss_weight + roll_mask_loss_weight > 0
        # roll mask loss needs more than one global view
        assert num_global_view > 1 or roll_mask_loss_weight == 0
        # corrent roll mask only supports two global views
        assert num_global_view == 1 or num_global_view == 2


        student_model_dict = dict()
        teacher_model_dict = dict()
        if teacher_custom is None: 
            teacher_custom = {}
        # backbone is a model dict, i.e. config
        student_backbone = backbone()

        # update backbone with teacher custom, like turning down drop out
        teacher_backbone = backbone()

        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        head = partial(
            ClusterHead,
            in_channels=head_in_channels,
            hidden_channels=head_hidden_channels,
            embed_channels=head_embed_channels,
            num_prototypes=num_prototypes
        )
        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0: 
            student_model_dict["mask_head"] = head()
            teacher_model_dict["mask_head"] = head()
        if self.unmask_loss_weight > 0: 
            student_model_dict["unmask_head"] = head()
            teacher_model_dict["unmask_head"] = head()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        # identical initialization?
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False


    def before_train(self): 
        ...



    def update_teacher(self):
        # EMA update teacher
        with torch.no_grad(): 
            m = self.momentum,
            student_param_list = list(self.student.parameters())
            teacher_param_list = list(self.teacher.parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)



    @staticmethod
    def sinkhorn_knopp(feat, temp, num_iter=3): 
        feat = feat.float()
        world_size = 1 # not distributed yet
        Q = torch.exp(feat / temp).t() # K by B
        B = Q.shape[1] * world_size # num sampels to assign
        K = Q.shape[0] # num prototypes

        sum_Q = torch.sum(Q)
        Q /= sum_Q
        for it in range(num_iter): 
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B

        return Q.t()


    def generate_mask(self, coord: torch.Tensor): 
        mask_size = self.mask_size
        mask_ratio = self.mask_ratio

        # anchor each play to min point
        min_coord = coord.min(dim=1)
        # find patch num for each dimension
        grid_coord = ((coord - min_coord) // mask_size).int()
        # grab all points in the same patch
        unique, cluster = torch.unique(grid_coord, dim=1, sorted=True, return_inverse=True)

        num_patches = unique.shape[1]
        mask_patch_num = int(num_patches * mask_ratio)
        # shuffle indexes, take a slice
        patch_index = torch.randperm(num_patches, device=coord.device)
        mask_patch_index = patch_index[:mask_patch_num]

        # bool mask - false not masked, true=masked
        point_mask = torch.isin(cluster, mask_patch_index)

        # bool mask, patch num
        return point_mask, cluster



    def student_backprop(self, losses: dict):
        ...


    def forward(self, data_dict: dict, step_dict: dict):
        ...