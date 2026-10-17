import torch
import torch.nn as nn

from nflMAE.models import backbones

class NflMaeModel(nn.Module): 

    def __init__(self, cfg, channels): 
        super().__init__()
        self.cfg = cfg

        # TODO jitter masked locations

        # self.masked_jitter = cfg['masked_jitter']

        self.embed_layer = nn.Sequential(
            nn.Linear(channels, self.cfg['encoder']['model_dimension']),
            nn.LayerNorm(self.cfg['encoder']['model_dimension'])
        )

        self.encoder = backbones.MyModel(**cfg['encoder']) 

        # might not have decoder when just trying to get embeds
        if "decoder" in cfg:
            assert not cfg['decoder']['add_cls_token']
            self.decoder = backbones.MyModel(**cfg['decoder']) 

            self.decoder_head = nn.Linear(cfg['decoder']['model_dimension'], channels)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg['decoder']['model_dimension']))
        return 


    def forward(self, data_dict: dict):
        bs, n, c = data_dict['feat'].shape
        d = self.cfg['encoder']['model_dimension']

        feat = data_dict['feat']
        mask = data_dict['mask']
        # add in the reference frame
        if 'reference_feat' in data_dict:
            n_reference = data_dict['reference_feat'].shape[1]
            # assert n_reference == n
            feat = torch.cat( (data_dict['reference_feat'], feat), dim=1)
            mask = torch.cat( (torch.zeros(bs, n_reference, dtype=torch.bool, device='mps'), mask), dim=1)

        coord = data_dict.get('coord')
        centroid = data_dict.get('centroid')
        if 'reference_coord' in data_dict:
            coord = torch.cat( (data_dict['reference_coord'], coord), dim=1)
            centroid = torch.cat( (data_dict['reference_coord'], centroid), dim=1)

        # get player vectors to the embedding dimension
        # either pass mask tokens to the encoder, or don't and add them later
        if self.cfg['encode_masks']: 
            # add the mask tokens to the encoder input

            # still only embed the unmasked players, don't want leakage
            unmasked_only_feat = feat[~mask].view(bs, -1, c)        
            embeds = self.embed(unmasked_only_feat)
            
            feat, coord = self.add_mask_tokens(
                embeds, coord, centroid, mask
            )
            fids = data_dict.get('fids')
            pids = data_dict.get('pids')

        else: 
            # only give the unmasked players to the encoder input
            n_masked = int((~mask).sum() / bs)
            raw_feat = feat[~mask].reshape(bs, n_masked, c)

            if coord is not None: 
                coord = coord[~mask].reshape(bs, n_masked, 2)

            fids = data_dict.get('fids')
            pids = data_dict.get('pids')
            if fids is not None:
                fids = fids[~mask].reshape(bs, n_masked)
            if pids is not None: 
                pids = pids[~mask].reshape(bs, n_masked)

            feat = self.embed_layer(raw_feat)

        encoded = self.encoder(feat, coord, fids, pids)
        
        # save cls token
        if self.cfg['encoder']['add_cls_token']:
            encoder_cls_token = encoded[:, :1, :]
            encoded = encoded[:, 1:, :]


        # re-add mask tokens to the sequence?
        if self.cfg['encode_masks']: 
            # don't need to 
            pass

        else: 
            # yes
            encoded, coord = self.add_mask_tokens(encoded, coord, centroid, mask)

        if "reference_feat" in data_dict:
            encoded = torch.cat( (encoder_cls_token, encoded[:, n_reference:, :]), dim=1)
            mask = torch.cat( (torch.zeros(bs, 1, dtype=torch.bool, device='mps'), mask[:, n_reference:]), dim=1)
        if "reference_coord" in data_dict:
            coord = coord[:, n_reference:, :]

        predictions = self.decoder_head(self.decoder(encoded, coord))

        # remove decoder cls token
        if self.cfg['decoder']['add_cls_token']: 
            predictions = predictions[:, 1:, :]

        masked_predictions = predictions[mask].view(bs, -1, c)

        return masked_predictions


    def add_mask_tokens(self, embedded_features: torch.Tensor, coords: torch.Tensor, centroids, mask: torch.Tensor):
        B, n_masked, D = embedded_features.shape
        P = mask.shape[1]

        # whole play's worth of mask tokens
        mask_tokens = self.mask_token.repeat(B, P, 1)

        #assert mask_tokens.shape == embedded_features.shape, f"{mask_tokens.shape} != {embedded_features.shape}"

        mask_tokens[~mask] = embedded_features.reshape(-1, D)
        masked_features = mask_tokens.reshape(B, P, D)
        # masked_features = embedded_features.where(~mask.unsqueeze(-1), mask_tokens)

        # do the same for coordinates
        if coords is not None and centroids is not None: 
            centroids[~mask] = coords[~mask].reshape(-1, 2)
            masked_coords = centroids.reshape(B, P, 2)
            # masked_coords = coords.where(~mask.unsqueeze(-1), centroids)

        return masked_features, masked_coords
    
    
    def embed(self, raw_feat: torch.Tensor):
        # take unmasked input, raise to embed dimension

        return self.embed_layer(raw_feat)



    def encode(self, data_dict: dict, cls_only=True, token_mask=None):
        bs = data_dict['feat'].shape[0]
        c = data_dict['feat'].shape[-1]

        embeded = self.embed(data_dict['feat'])

        unmasked_embed = self.encoder(embeded, data_dict.get('coord'), data_dict.get('fids'), data_dict.get('pids'))
        d = unmasked_embed.shape[-1]

        if cls_only: 
        # return cls token
            return unmasked_embed[:, 0, :]
        else: 
            if token_mask is not None:
                # remove cls_token
                # assert there is only one per mask?
                assert token_mask.shape[1] == unmasked_embed.shape[1] - 1
                return unmasked_embed[:, 1:, :][token_mask].view(bs, d)
            else: 
                return unmasked_embed


    def dimension(self): 
        return self.encoder.model_dimension

