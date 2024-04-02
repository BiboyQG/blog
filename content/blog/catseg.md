+++
title = 'Catseg'
date = 2024-04-02T05:21:09-05:00
tags: ["Open Vocabulary Segmentation", "CLIP", "Swin Transformer"]
categories: ["Computer Vision"]
draft = false
+++

### 1. Model Architecture setup and evaluation data flow(for ade150k)

`CATSeg` setup:

- `backbone`: D2SwinTransformer -> Swintransformer -> BasicLayer(2) -> SwinTransformerBlock -> WindowAttention

- `sem_seg_head`: `CATSegHead.from_config` -> `CATSegPredictor` ->

  - Load CLIP model -> Load text templates -> `class_embeddings(self.class_texts, prompt_templates, clip_model)` -> for each class:

    - bpe encode classname in different templates and save results in variable `texts` **(80(number of templates), 77(number of sentence length))**.
    - CLIP encode `texts` :
      - `texts` go through `token_embedding`(`nn.Embedding`) **(80,77,768(hidden_dim))**
      - `texts` go through a 12 layers of ResidualAttentionBlock **(80,77,768)**
      - take features of `texts` from the `eot_token` **(80,768)**

  - do the above for all classes **(150(number of test classes),80,768)**

  - `Aggregator` -> 2 layers of `AggregatorLayer`:

    - `swin_block`:

      - `SwinTransformerBlockWrapper`:

        ```python
        class SwinTransformerBlockWrapper(nn.Module):
            def __init__(self, dim, appearance_guidance_dim, input_resolution, nheads=4, window_size=5):
                super().__init__()
                self.block_1 = SwinTransformerBlock(dim,
                                                    appearance_guidance_dim,
                                                    input_resolution,
                                                    num_heads=nheads,
                                                    head_dim=None,
                                                    window_size=window_size,
                                                    shift_size=0)
                self.block_2 = SwinTransformerBlock(dim,
                                                    appearance_guidance_dim,
                                                    input_resolution,
                                                    num_heads=nheads,
                                                    head_dim=None,
                                                    window_size=window_size,
                                                    shift_size=window_size // 2)
                self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None

        ```

    - `attention`:

      - `ClassTransformerLayer`:

        ```python
        class ClassTransformerLayer(nn.Module):
            def __init__(self, hidden_dim=64, guidance_dim=64, nheads=8, attention_type='linear', pooling_size=(4, 4)) -> None:
                super().__init__()
                self.pool = nn.AvgPool2d(pooling_size)
                self.attention = AttentionLayer(hidden_dim,
                                                guidance_dim,
                                                nheads=nheads,
                                                attention_type=attention_type)
                self.MLP = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                )

                self.norm1 = nn.LayerNorm(hidden_dim)
                self.norm2 = nn.LayerNorm(hidden_dim)
        ```

        ```python
        class LinearAttention(nn.Module):
            def __init__(self, eps=1e-6):
                super().__init__()
                self.feature_map = elu_feature_map
                self.eps = eps

            def forward(self, queries, keys, values):
                """ Multi-Head linear attention proposed in "Transformers are RNNs"
                Args:
                    queries: [N, L, H, D]
                    keys: [N, S, H, D]
                    values: [N, S, H, D]
                    q_mask: [N, L]
                    kv_mask: [N, S]
                Returns:
                    queried_values: (N, L, H, D)
                """
                Q = self.feature_map(queries)
                K = self.feature_map(keys)

                v_length = values.size(1)
                values = values / v_length  # prevent fp16 overflow
                KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
                Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
                queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

                return queried_values.contiguous()


        class FullAttention(nn.Module):
            def __init__(self, use_dropout=False, attention_dropout=0.1):
                super().__init__()
                self.use_dropout = use_dropout
                self.dropout = nn.Dropout(attention_dropout)

            def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
                """ Multi-head scaled dot-product attention, a.k.a full attention.
                Args:
                    queries: [N, L, H, D]
                    keys: [N, S, H, D]
                    values: [N, S, H, D]
                    q_mask: [N, L]
                    kv_mask: [N, S]
                Returns:
                    queried_values: (N, L, H, D)
                """

                # Compute the unnormalized attention and apply the masks
                QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
                if kv_mask is not None:
                    QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

                # Compute the attention and the weighted average
                softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
                A = torch.softmax(softmax_temp * QK, dim=2)
                if self.use_dropout:
                    A = self.dropout(A)

                queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

                return queried_values.contiguous()


        class AttentionLayer(nn.Module):
            def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type='linear'):
                super().__init__()
                self.nheads = nheads
                self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
                self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
                self.v = nn.Linear(hidden_dim, hidden_dim)

                if attention_type == 'linear':
                    self.attention = LinearAttention()
                elif attention_type == 'full':
                    self.attention = FullAttention()
                else:
                    raise NotImplementedError
        ```

    - Remaining of `Aggregator`

      ```python
      self.guidance_projection = nn.Sequential(
          nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
      ) if appearance_guidance_dim > 0 else None

      self.text_guidance_projection = nn.Sequential(
          nn.Linear(text_guidance_dim, text_guidance_proj_dim),
          nn.ReLU(),
      ) if text_guidance_dim > 0 else None

      self.decoder_guidance_projection = nn.ModuleList([
          nn.Sequential(
              nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
          ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
      ]) if decoder_guidance_dims[0] > 0 else None

      self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
      self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
      self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
      ```

      - `Up`

        ```python
        class Up(nn.Module):
            """Upscaling then double conv"""

            def __init__(self, in_channels, out_channels, guidance_channels):
                super().__init__()

                self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)

            def forward(self, x, guidance=None):
                x = self.up(x)
                if guidance is not None:
                    T = x.size(0) // guidance.size(0)
                    guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
                    x = torch.cat([x, guidance], dim=1)
                return self.conv(x)
        ```

`CATSeg` forward for each image:

- `image` **(3,640,854)** -> `self.inference_sliding_window(batched_inputs)`

  - ```python
    image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
    ```

    -> **(3,640,640)**

  - ```python
    image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
    ```

    -> **(442368(3x384x384(kernel size)),4(number of such patch))** -> **(4,3,384,384)**

  - ```python
    global_image = F.interpolate(images[0].unsqueeze(0),
                                 size=(kernel, kernel),
                                 mode='bilinear',
                                 align_corners=False)
    image = torch.cat((image, global_image), dim=0)
    ```

    -> **(5,3,384,384)** 与下面呼应！

  - ```python
    features = self.backbone(images) # features: a dictionary with length of 3
    clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True) # clip_images: (5, 3, 336, 336)
    # outputs: (5,150,96,96)
    outputs = self.sem_seg_head(clip_features, features)
    ```

    - `features`:
      ![image-20240401003703609](/Users/biboyqg/Library/Application Support/typora-user-images/image-20240401003703609.png)
    - `clip_features`: **(5,577(24x24+1),768)**
    - `outputs`: **(5,150(number of classes),96,96)**

  - After the three steps: `outputs` -> **(5,150,96,96)**

    ```python
    # outputs: (5,150,96,96) -> (5,150,384,384)
    outputs = F.interpolate(outputs,
                            size=kernel,
                            mode="bilinear",
                            align_corners=False)   # -> (5,150,384,384) 与上面呼应！
    outputs = outputs.sigmoid()

    global_output = outputs[-1:]
    global_output = F.interpolate(global_output,
                                  size=out_res,
                                  mode='bilinear',
                                  align_corners=False,)
    outputs = outputs[:-1]												 # -> (4,150,384,384)
    outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
    # fenzi: (4,22118400) -> (22118400,4) -> (150,640,640)
    # fenmu: (1,640,640)
    # This steps normalize the effects brought by the fold operation.
    outputs = (outputs + global_output) / 2.
    # -> (1,150,640,640)

    height = batched_inputs[0].get("height", out_res[0])
    width = batched_inputs[0].get("width", out_res[1])
    output = sem_seg_postprocess(outputs[0], out_res, height, width)
    # -> (150,512,683)
    ```

  ### The workflow within three main steps

- The workflow within `features = self.backbone(images) # features: a dictionary with length of 3`:

  - ```python
    class D2SwinTransformer(SwinTransformer, Backbone):
          def forward(self, x): # x -> (5,3,384,384)
            """
            Args:
                x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
            Returns:
                dict[str->Tensor]: names and the corresponding features
            """
            assert (
                x.dim() == 4
            ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
            outputs = {}
            y = super().forward(x) # y -> a dict of three tensors: {(5,128,96,96), (5,256,48,48), (5,512,24,24)}. Same for shape of the outputs
            for k in y.keys():
                if k in self._out_features:
                    outputs[k] = y[k]
            return outputs
    ```

  - ```python
    class SwinTransformer(nn.Module):
        def forward(self, x):
            """Forward function."""
            x = self.patch_embed(x) # (5,3,384,384) -> (5,128,96,96) 解释在下面

            Wh, Ww = x.size(2), x.size(3)
            if self.ape:
                # interpolate the position embedding to the corresponding size
                absolute_pos_embed = F.interpolate(
                    self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
                )
                x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
            else:
                x = x.flatten(2).transpose(1, 2) # -> (5,9216(96x96),128)
            x = self.pos_drop(x) # no change (5,9216,128)

            outs = {}
            for i in range(self.num_layers):
                layer = self.layers[i]
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww) # x_out -> (5,9216,128)/(5,2304,256)/(5,576,256)

                if i in self.out_indices:
                    norm_layer = getattr(self, f"norm{i}")
                    x_out = norm_layer(x_out) # no change (5,9216,128)/(5,2304,256)/(5,576,512)

                    out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous() # out: (5,128,96,96)/(5,256,48,48)/(5,512,24,24)
                    outs["res{}".format(i + 2)] = out

            return outs
    ```

  - ```python
    class PatchEmbed(nn.Module):
        def forward(self, x):
            """Forward function."""
            # padding
            _, _, H, W = x.size()
            if W % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
            if H % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

            x = self.proj(x)  # B C Wh Ww (5,3,384,384) -> (5,128,96,96)
            if self.norm is not None:
                Wh, Ww = x.size(2), x.size(3)
                x = x.flatten(2).transpose(1, 2)
                x = self.norm(x)
                x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

            return x # (5,128,96,96) 传回上面👆🏻
    ```

  - ```python
    self.layers:
    ModuleList(
      (0): BasicLayer(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=128, out_features=384, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=512, out_features=128, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=128, out_features=384, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.014)
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=512, out_features=128, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          (reduction): Linear(in_features=512, out_features=256, bias=False)
          (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): BasicLayer(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.029)
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=256, out_features=1024, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1024, out_features=256, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=256, out_features=768, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=256, out_features=256, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.043)
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=256, out_features=1024, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=1024, out_features=256, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (downsample): PatchMerging(
          (reduction): Linear(in_features=1024, out_features=512, bias=False)
          (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): BasicLayer(
        (blocks): ModuleList(
          (0): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.057)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (1): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.071)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (2): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.086)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (3): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.100)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (4): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.114)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (5): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.129)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (6): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.143)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (7): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.157)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (8): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.171)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (9): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.186)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (10): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.200)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (11): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.214)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (12): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.229)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (13): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.243)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (14): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.257)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (15): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.271)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (16): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.286)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
          (17): SwinTransformerBlock(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (qkv): Linear(in_features=512, out_features=1536, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=512, out_features=512, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): DropPath(drop_prob=0.300)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=512, out_features=2048, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=2048, out_features=512, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
          )
        )
      )
    )
    ```

  - ```python
    class BasicLayer(nn.Module):
    		def forward(self, x, H, W):
            """Forward function.
            Args:
                x: Input feature, tensor size (B, H*W, C).
                H, W: Spatial resolution of the input feature.
            """

            # calculate attention mask for SW-MSA
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size
            img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )

            for blk in self.blocks:
                blk.H, blk.W = H, W
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, attn_mask)
                else:
                    x = blk(x, attn_mask) # (5,9216,128) -> (5,9216,128)
            if self.downsample is not None:
                x_down = self.downsample(x, H, W)
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
                return x, H, W, x_down, Wh, Ww
            else:
                return x, H, W, x, H, W
    ```

- The workflow within `clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True) # clip_images: (5, 3, 336, 336), clip_features: (5, 577, 768)`:

  - ```python
    class VisualTransformer(nn.Module):
        def forward(self, x: torch.Tensor, dense=False):
            # (5,3,336,336)
            x = self.conv1(x)  # shape = [5, 1024, 24, 24]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [5, 1024, 576(24x24)]
            x = x.permute(0, 2, 1)  # shape = [5, 576, 1024]
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [5, 576+1, 1024]

            if dense and (x.shape[1] != self.positional_embedding.shape[0]):
                x = x + self.resized_pos_embed(self.input_resolution, x.shape[1]).to(x.dtype)
            else:
                x = x + self.positional_embedding.to(x.dtype) # shape = [5, 577, 1024]

            x = self.ln_pre(x) # shape = [5, 577, 1024]

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x, dense)
            x = x.permute(1, 0, 2)  # LND -> NLD shape = [5, 577, 1024]

            if dense:
                x = self.ln_post(x[:, :, :])
            else:
                x = self.ln_post(x[:, 0, :])

            if self.proj is not None:
                x = x @ self.proj # shape -> [5, 577, 768]

            return x # shape = [5, 577, 768]
    ```

- The workflow within `outputs = self.sem_seg_head(clip_features, features)`:

  - ```python
    class CATSegHead(nn.Module):
        def forward(self, features, guidance_features):
            """
            Arguments:
                img_feats: (B, C, HW)
                affinity_features: (B, C, )
            """
            # features: (5,577,768) -> (5,768,24,24)
            img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
            return self.predictor(img_feat, guidance_features)
    ```

  - ```python
    class CATSegPredictor(nn.Module):
      	# self.transformer -> Aggregator!
        def forward(self, x, vis_guidance):
            vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
            # text: (150, 80, 768)
            text = self.text_features if self.training else self.text_features_test
            # text -> (5, 150, 80, 768)
            text = text.repeat(x.shape[0], 1, 1, 1)
            out = self.transformer(x, text, vis) # This Aggregator part: below👇🏻
            return out
    ```

  - `text_feats`: **(5,150,80,768)**, `img_feats`: **(5,768,24,24)**

    ```python
    class Aggregator(nn.Module):
        def __init__(self,
            text_guidance_dim=512,
            text_guidance_proj_dim=128,
            appearance_guidance_dim=512,
            appearance_guidance_proj_dim=128,
            decoder_dims = (64, 32),
            decoder_guidance_dims=(256, 128),
            decoder_guidance_proj_dims=(32, 16),
            num_layers=4,
            nheads=4,
            hidden_dim=128,
            pooling_size=(6, 6),
            feature_resolution=(24, 24),
            window_size=12,
            attention_type='linear',
            prompt_channel=80,
        ) -> None:
            super().__init__()
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim

            self.layers = nn.ModuleList([
                AggregatorLayer(
                    hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim,
                    nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type
                ) for _ in range(num_layers)
            ])

            self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)

            self.guidance_projection = nn.Sequential(
                nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) if appearance_guidance_dim > 0 else None

            self.text_guidance_projection = nn.Sequential(
                nn.Linear(text_guidance_dim, text_guidance_proj_dim),
                nn.ReLU(),
            ) if text_guidance_dim > 0 else None

            self.decoder_guidance_projection = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
            ]) if decoder_guidance_dims[0] > 0 else None

            self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
            self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
            self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

    #---------------------------------------------------------------------------------#

        def feature_map(self, img_feats, text_feats):
            img_feats = F.normalize(img_feats, dim=1) # B C H W
            img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
            text_feats = F.normalize(text_feats, dim=-1) # B T P C
            text_feats = text_feats.mean(dim=-2)
            text_feats = F.normalize(text_feats, dim=-1) # B T C
            text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
            return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

        def correlation(self, img_feats, text_feats):
            img_feats = F.normalize(img_feats, dim=1) # (5,768,24,24)
            text_feats = F.normalize(text_feats, dim=-1) # (5,150,80,768)
            corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
            return corr # corr: (5,80,150,24,24)

        def corr_embed(self, x):
            B = x.shape[0]
            # x: (5,80,150,24,24) -> (750, 80, 24, 24)
            corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
            # x: (750, 80, 24, 24) -> (750, 128, 24, 24)
            corr_embed = self.conv1(corr_embed)
            # x: (750, 128, 24, 24) -> (5, 128, 150, 24, 24)
            corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
            return corr_embed

        def corr_projection(self, x, proj):
            corr_embed = rearrange(x, 'B C T H W -> B T H W C')
            corr_embed = proj(corr_embed)
            corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
            return corr_embed

        def upsample(self, x):
            B = x.shape[0]
            corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
            corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
            corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
            return corr_embed

        def conv_decoder(self, x, guidance):
            B = x.shape[0]
            corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
            corr_embed = self.decoder1(corr_embed, guidance[0])
            corr_embed = self.decoder2(corr_embed, guidance[1])
            corr_embed = self.head(corr_embed)
            corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
            return corr_embed

        def forward(self, img_feats, text_feats, appearance_guidance):
            """
            Arguments:
                img_feats: (B, C, H, W)
                text_feats: (B, T, P, C)
                apperance_guidance: tuple of (B, C, H, W)
            """
            # text_feats: (5,150,80,768), img_feats: (5,768,24,24)
            corr = self.correlation(img_feats, text_feats) # corr: (5,80,150,24,24)
            corr_embed = self.corr_embed(corr)

            projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
            if self.guidance_projection is not None:
              	# projected_guidance: (5,128,24,24)
                projected_guidance = self.guidance_projection(appearance_guidance[0])
            if self.decoder_guidance_projection is not None:
              	# 见下图👇🏻
                projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

            if self.text_guidance_projection is not None:
              	# (5,150,80,768) -> (5,150,768)
                text_feats = text_feats.mean(dim=-2)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                # (5,150,768) -> (5,150,128)
                projected_text_guidance = self.text_guidance_projection(text_feats)

            # corr_embed: (5,80,150,24,24) -> (5,80,150,24,24) -> (5,80,150,24,24)
            for layer in self.layers:
                corr_embed = layer(corr_embed, projected_guidance,
                                   projected_text_guidance)
    				# corr_embed: (5,80,150,24,24), 见最下面👇🏻
            # logit: (5,150,96,96)
            logit = self.conv_decoder(corr_embed, projected_decoder_guidance)

            return logit
    ```

    ![image-20240401182112928](/Users/biboyqg/Library/Application Support/typora-user-images/image-20240401182112928.png)

    ```python
    for layer in self.layers:
    		corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance):
    ```

    with `self.layers`'s structure as follow:

    ```python
    ModuleList(
      (0): AggregatorLayer(
        (swin_block): SwinTransformerBlockWrapper(
          (block_1): SwinTransformerBlock(
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (q): Linear(in_features=256, out_features=128, bias=True)
              (k): Linear(in_features=256, out_features=128, bias=True)
              (v): Linear(in_features=128, out_features=128, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU(approximate='none')
              (drop1): Dropout(p=0.0, inplace=False)
              (fc2): Linear(in_features=512, out_features=128, bias=True)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
          (block_2): SwinTransformerBlock(
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (q): Linear(in_features=256, out_features=128, bias=True)
              (k): Linear(in_features=256, out_features=128, bias=True)
              (v): Linear(in_features=128, out_features=128, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU(approximate='none')
              (drop1): Dropout(p=0.0, inplace=False)
              (fc2): Linear(in_features=512, out_features=128, bias=True)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
          (guidance_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        )
        (attention): ClassTransformerLayer(
          (pool): AvgPool2d(kernel_size=[1, 1], stride=[1, 1], padding=0)
          (attention): AttentionLayer(
            (q): Linear(in_features=256, out_features=128, bias=True)
            (k): Linear(in_features=256, out_features=128, bias=True)
            (v): Linear(in_features=128, out_features=128, bias=True)
            (attention): LinearAttention()
          )
          (MLP): Sequential(
            (0): Linear(in_features=128, out_features=512, bias=True)
            (1): ReLU()
            (2): Linear(in_features=512, out_features=128, bias=True)
          )
          (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): AggregatorLayer(
        (swin_block): SwinTransformerBlockWrapper(
          (block_1): SwinTransformerBlock(
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (q): Linear(in_features=256, out_features=128, bias=True)
              (k): Linear(in_features=256, out_features=128, bias=True)
              (v): Linear(in_features=128, out_features=128, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU(approximate='none')
              (drop1): Dropout(p=0.0, inplace=False)
              (fc2): Linear(in_features=512, out_features=128, bias=True)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
          (block_2): SwinTransformerBlock(
            (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (attn): WindowAttention(
              (q): Linear(in_features=256, out_features=128, bias=True)
              (k): Linear(in_features=256, out_features=128, bias=True)
              (v): Linear(in_features=128, out_features=128, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=128, out_features=128, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
              (softmax): Softmax(dim=-1)
            )
            (drop_path): Identity()
            (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=128, out_features=512, bias=True)
              (act): GELU(approximate='none')
              (drop1): Dropout(p=0.0, inplace=False)
              (fc2): Linear(in_features=512, out_features=128, bias=True)
              (drop2): Dropout(p=0.0, inplace=False)
            )
          )
          (guidance_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        )
        (attention): ClassTransformerLayer(
          (pool): AvgPool2d(kernel_size=[1, 1], stride=[1, 1], padding=0)
          (attention): AttentionLayer(
            (q): Linear(in_features=256, out_features=128, bias=True)
            (k): Linear(in_features=256, out_features=128, bias=True)
            (v): Linear(in_features=128, out_features=128, bias=True)
            (attention): LinearAttention()
          )
          (MLP): Sequential(
            (0): Linear(in_features=128, out_features=512, bias=True)
            (1): ReLU()
            (2): Linear(in_features=512, out_features=128, bias=True)
          )
          (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    ```

    and data flow in each `AggregatorLayer` is as follow (the change of shape is the same for two layers):

    ```python
    for layer in self.layers:
      	# corr_embed: (5,128,150,24,24)
        # projected_guidance: (5,128,24,24)
        # projected_text_guidance: (5,150,128)
    		corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)
    ```

    ```python
    class AggregatorLayer(nn.Module):
        def __init__(self, hidden_dim=64, text_guidance_dim=512,
                     appearance_guidance=512, nheads=4, input_resolution=(20, 20),
                     pooling_size=(5, 5), window_size=(10, 10),
                     attention_type='linear') -> None:
            super().__init__()
            self.swin_block = SwinTransformerBlockWrapper(hidden_dim,
                                                          appearance_guidance,
                                                          input_resolution, nheads,
                                                          window_size)
            self.attention = ClassTransformerLayer(hidden_dim,
                                                   text_guidance_dim,
                                                   nheads=nheads,
                                                   attention_type=attention_type,
                                                   pooling_size=pooling_size)

        def forward(self, x, appearance_guidance, text_guidance):
            """
            Arguments:
                x: B C T H W
            """
            # x: (5,128,150,24,24)
        		# appearance_guidance: (5,128,24,24)
        		# text_guidance: (5,150,128)

            # x: (5,128,150,24,24) -> (5,128,150,24,24)
            x = self.swin_block(x, appearance_guidance)
            # x: (5,128,150,24,24) -> (5,128,150,24,24)
            x = self.attention(x, text_guidance)
            return x
    ```

    For `SwinTransformerBlockWrapper`:

    ```python
    class SwinTransformerBlockWrapper(nn.Module):
        def __init__(self, dim, appearance_guidance_dim,
                     input_resolution, nheads=4, window_size=5):
            super().__init__()
            self.block_1 = SwinTransformerBlock(dim, appearance_guidance_dim,
                                                input_resolution, num_heads=nheads,
                                                head_dim=None, window_size=window_size,
                                                shift_size=0)
            self.block_2 = SwinTransformerBlock(dim, appearance_guidance_dim,
                                                input_resolution, num_heads=nheads,
                                                head_dim=None, window_size=window_size,
                                                shift_size=window_size // 2)
            self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None

        def forward(self, x, appearance_guidance):
            """
            Arguments:
                x: B C T H W
                appearance_guidance: B C H W
            """
            B, C, T, H, W = x.shape
            # x: (5,128,150,24,24) -> (750,576,128)
            x = rearrange(x, 'B C T H W -> (B T) (H W) C')
            if appearance_guidance is not None:
              	# appearance_guidance: (5,128,24,24) -> (750,576,128) -> (750,576,128)
                appearance_guidance = self.guidance_norm(repeat(appearance_guidance, 'B C H W -> (B T) (H W) C', T=T))
            # x: (750,576,128) -> (750,576,128)
            x = self.block_1(x, appearance_guidance)
            # x: (750,576,128) -> (750,576,128)
            x = self.block_2(x, appearance_guidance)
            # x: (750,576,128) -> (5,128,150,24,24)
            x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, T=T, H=H, W=W)
            return x
    ```

    In `SwinTransformerBlock`:

    ```python
    class SwinTransformerBlock(nn.Module):
        def forward(self, x, appearance_guidance):
            H, W = self.input_resolution
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"

            shortcut = x
            x = self.norm1(x)
            # x: (750, 576, 128) -> (750, 24, 24, 128)
            x = x.view(B, H, W, C)
            if appearance_guidance is not None:
              	# appearance_guidance: (750, 576, 128) -> (750, 24, 24, 128)
                appearance_guidance = appearance_guidance.view(B, H, W, -1)
                # x: (750, 24, 24, 128) -> (750, 24, 24, 256)
                x = torch.cat([x, appearance_guidance], dim=-1)

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # num_win*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, x_windows.shape[-1])  # num_win*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # num_win*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x.view(B, H * W, C)

            # FFN
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            # x: (750,576,128)
            return x
    ```

    For `self.attention = ClassTransformerLayer`:

    ```python
    class ClassTransformerLayer(nn.Module):
        def __init__(self, hidden_dim=64, guidance_dim=64, nheads=8,
                     attention_type='linear', pooling_size=(4, 4)) -> None:
            super().__init__()
            self.pool = nn.AvgPool2d(pooling_size)
            self.attention = AttentionLayer(hidden_dim, guidance_dim,
                                            nheads=nheads,
                                            attention_type=attention_type)
            self.MLP = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )

            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

        def pool_features(self, x):
            """
            Intermediate pooling layer for computational efficiency.
            Arguments:
                x: B, C, T, H, W
            """
            B = x.size(0)
            # x: (5,128,150,24,24)
            x = rearrange(x, 'B C T H W -> (B T) C H W')
            x = self.pool(x)
            x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
            # x: (5,128,150,24,24)
            return x

        def forward(self, x, guidance):
            """
            Arguments:
                x: B, C, T, H, W
                guidance: B, T, C
            """
            B, _, _, H, W = x.size()
            # x: (5,128,150,24,24)
            # x_pool: (5,128,150,24,24)
            x_pool = self.pool_features(x)
            *_, H_pool, W_pool = x_pool.size()

            # x_pool: (5,128,150,24,24) -> (2880,150,128)
            x_pool = rearrange(x_pool, 'B C T H W -> (B H W) T C')
            if guidance is not None:
              	# guidance: (5,150,128) -> (2880,150,128)
                guidance = repeat(guidance, 'B T C -> (B H W) T C', H=H_pool, W=W_pool)

            # x_pool: (2880,150,128)
            x_pool = x_pool + self.attention(self.norm1(x_pool), guidance) # 见下面👇🏻
            x_pool = x_pool + self.MLP(self.norm2(x_pool)) # MLP

            # x_pool: (750,128,24,24)
            x_pool = rearrange(x_pool, '(B H W) T C -> (B T) C H W', H=H_pool, W=W_pool)
            # x_pool: (750,128,24,24)
            x_pool = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=True)
            x_pool = rearrange(x_pool, '(B T) C H W -> B C T H W', B=B)

            # x: (5,128,150,24,24)
            x = x + x_pool # Residual
            return x
    ```

    For `self.attention(self.norm1(x_pool), guidance)`:

    ```python
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type='linear'):
            super().__init__()
            self.nheads = nheads
            self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
            self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
            self.v = nn.Linear(hidden_dim, hidden_dim)

            if attention_type == 'linear':
                self.attention = LinearAttention()
            elif attention_type == 'full':
                self.attention = FullAttention()
            else:
                raise NotImplementedError

        def forward(self, x, guidance):
            """
            Arguments:
                x: B, L, C
                guidance: B, L, C
            """
            # q,k,v: (2880,150,128)
            q = self.q(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.q(x)
            k = self.k(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.k(x)
            v = self.v(x)

            # q,k,v: (2880,150,4,32)
            q = rearrange(q, 'B L (H D) -> B L H D', H=self.nheads)
            k = rearrange(k, 'B S (H D) -> B S H D', H=self.nheads)
            v = rearrange(v, 'B S (H D) -> B S H D', H=self.nheads)
    				# out: (2880,150,4,32)
            out = self.attention(q, k, v)
            # out: (2880,150,4,32) -> (2880,150,128)
            out = rearrange(out, 'B L H D -> B L (H D)')
            return out
    ```

    For `self.conv_decoder(corr_embed, projected_decoder_guidance)`:

    ```python
        def conv_decoder(self, x, guidance):
            B = x.shape[0]
          	# corr_embed: (750,128,24,24)
            corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
            # corr_embed: (750,64,48,48)
            corr_embed = self.decoder1(corr_embed, guidance[0])
            # corr_embed: (750,32,96,96)
            corr_embed = self.decoder2(corr_embed, guidance[1])
            # corr_embed: (750,1,96,96)
            corr_embed = self.head(corr_embed)
            # corr_embed: (5,150,96,96)
            corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
            return corr_embed

    class Up(nn.Module):
        """Upscaling then double conv"""

        def __init__(self, in_channels, out_channels, guidance_channels):
            super().__init__()

            self.up = nn.ConvTranspose2d(in_channels,
                                         in_channels - guidance_channels,
                                         kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        def forward(self, x, guidance=None):
          	# x: (750,128,24,24) -> (750,96,48,48)
            x = self.up(x)
            if guidance is not None:
                T = x.size(0) // guidance.size(0)
                # guidance: (5,32,48,48) -> (750,32,48,48)
                guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
                # x: (750,96,48,48) -> (750,128,48,48)
                x = torch.cat([x, guidance], dim=1)
                # x: (750,128,48,48) -> (750,64,48,48)
            return self.conv(x)
    ```

### 2. Unknown stuff

- Loss computation
- GroupNorm
