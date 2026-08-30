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



# models don't track training steps, let that happen with the trainer

import backbones

class REST(nn.Module): 
    """Rich Embeddings from Sports Transformers"""

    def __init__(self, cfg, channels): 
        super().__init__()
        self.cfg = cfg

        # TODO jitter masked locations

        #self.masked_jitter = cfg['masked_jitter']

        self.embed_layer = nn.Sequential(
            nn.Linear(channels, self.cfg['encoder']['model_dimension']),
            nn.LayerNorm(self.cfg['encoder']['model_dimension'])
        )

        self.encoder = backbones.MyModel(**cfg['encoder']) 

        assert not cfg['decoder']['add_cls_token']
        self.decoder = backbones.MyModel(**cfg['decoder']) 

    
        self.decoder_head = nn.Linear(cfg['decoder']['model_dimension'], channels)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg['decoder']['model_dimension']))

        # self.n_positions = n_positions
        # self.position_softmax = nn.Softmax(n_positions)



    def forward(self, data_dict: dict):
        bs = data_dict['feat'].shape[0]
        n = data_dict['feat'].shape[1]
        c = data_dict['feat'].shape[-1]
        d = self.cfg['encoder']['model_dimension']


        if self.cfg['encode_masks']: 
            # add the mask tokens to the encoder input

            # still only embed the unmasked players, don't want leakage
            embeds = self.embed(data_dict['feat'][~data_dict['mask']].view(bs, -1, c))
            
            feat, coord = self.add_mask_tokens(
                embeds, data_dict['coord'], data_dict['centroid'], data_dict['mask']
            )

        else: 
            # only give the unmasked players to the encoder input
            n_masked = int((~data_dict['mask']).sum() / bs)
            raw_feat = data_dict['feat'][~data_dict['mask']].reshape(bs, n_masked, c)
            coord = data_dict['coord'][~data_dict['mask']].reshape(bs, n_masked, 2)

            feat = self.embed_layer(raw_feat)



        encoded = self.encoder(feat, coord)
        # remove cls token
        if self.cfg['encoder']['add_cls_token']:
            encoded = encoded[:, 1:, :]


        # re-add mask tokens to the sequence?
        if self.cfg['encode_masks']: 
            # don't need to 
            pass

        else: 
            # yes
            encoded, coord = self.add_mask_tokens(encoded, data_dict['coord'], data_dict['centroid'], data_dict['mask'])
            #mask_tokens = self.mask_token.repeat(bs, n, 1)
            # align the mask tokens with their correct coordinate positions
            #mask_tokens[~data_dict['mask']] = encoded.reshape(-1, d)
            #encoded = mask_tokens.reshape(bs, n, d)

            #coord = data_dict['coord'].where(~data_dict['mask'].unsqueeze(-1), data_dict['centroid'])
    
            
        predictions = self.decoder_head(self.decoder(encoded, coord))

        # remove cls token
        if self.cfg['decoder']['add_cls_token']: 
            predictions = predictions[:, 1:, :]

        return predictions




    def add_mask_tokens(self, embedded_features: torch.Tensor, coords: torch.Tensor, centroids, mask: torch.Tensor):
        B, n_masked, D = embedded_features.shape
        P = coords.shape[1]

        # whole play's worth of mask tokens
        mask_tokens = self.mask_token.repeat(B, P, 1)

        #assert mask_tokens.shape == embedded_features.shape, f"{mask_tokens.shape} != {embedded_features.shape}"

        mask_tokens[~mask] = embedded_features.reshape(-1, D)
        masked_features = mask_tokens.reshape(B, P, D)
        # masked_features = embedded_features.where(~mask.unsqueeze(-1), mask_tokens)

        # do the same for coordinates
        masked_coords = coords.where(~mask.unsqueeze(-1), centroids)

        return masked_features, masked_coords
    
    
    def embed(self, raw_feat: torch.Tensor):
        # take unmasked input, raise to embed dimension

        return self.embed_layer(raw_feat)



    def encode(self, data_dict: dict, cls_only=True):
        bs = data_dict['feat'].shape[0]
        c = data_dict['feat'].shape[-1]

        unmasked_embed = self.encoder(data_dict['feat'], data_dict['coord'])

        if cls_only: 
        # return cls token
            return unmasked_embed[:, 0, :]
        else: 
            return unmasked_embed


    @torch.no_grad()
    def generate_mask(self, data_dict: dict):
        """Modifies data dict in place"""

        # determine grid centroids no matter the masking strategy
        min_coord, _ = data_dict['coord'].min(dim=1)
        # find patch num for each dimension
        grid_coord = ((data_dict['coord'] - min_coord.unsqueeze(1)) // self.cfg['grid_size']).int()
        # grab all points in the same patch
        unique, cluster = torch.unique(grid_coord.cpu(), dim=1, sorted=True, return_inverse=True)
        cluster = cluster.to('mps')
        data_dict['centroid'] = grid_coord * self.cfg['grid_size'] + self.cfg['grid_size'] / 2

        assert not (self.cfg['mask_regions'] and not self.cfg['encode_masks'])# not ready yet, can't handle uneven masking

        if self.cfg['mask_regions']: 
            # mask some grid patches
            num_patches = unique.shape[1]
            mask_patch_num = int(num_patches * self.cfg['mask_ratio'])
            # shuffle indexes, slice
            patch_index = torch.randperm(num_patches)
            mask_patch_index = patch_index[:mask_patch_num]

            # boolean mask
            player_mask = torch.isin(cluster, mask_patch_index)

            # save mask information for decoder (true = masked)
            data_dict['mask'] = player_mask


        # TODO: mask the football?
        if self.cfg['mask_players']:
            N, P, D = data_dict['feat'].shape  

            len_keep = int(P * (1 - self.cfg['mask_ratio']))
            
            noise = torch.rand(N, P - 1, device='mps')  # noise in [0, 1], don't noise for the football
            
            # sort noise for each sample
            # = random order of indices - truncate and set those be non-masks
            ids_shuffle = torch.cat((torch.zeros(N, 1, device="mps", dtype=torch.int32), torch.argsort(noise, dim=1) + 1), dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]

            mask = torch.ones([N, P], device='mps')
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
            # mask.scatter_(1, ids_keep, 0)

            data_dict['mask'] = mask.bool()


        return
