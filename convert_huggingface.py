# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Config


def convert(model, m0_path, m1_path, save_path):

    # load pretrained weights
    print("loading pretrianed weight")
    m0 = torch.load(m0_path, map_location='cpu')
    m1 = torch.load(m1_path, map_location='cpu')
    print("loading pretrianed weight finished")

    # get state dict
    model_state_dict = model.state_dict()

    # word embedding
    print("filling wte")
    # word embedding
    wte_weight = np.concatenate([
        m0["module"]["word_embeddings.weight"].numpy(),
        m1["module"]["word_embeddings.weight"].numpy(),
    ])

    assert model_state_dict["transformer.wte.weight"].shape == wte_weight.shape

    model_state_dict["transformer.wte.weight"] = torch.as_tensor(wte_weight)
    assert np.array_equal(model_state_dict["transformer.wte.weight"].numpy(), wte_weight)
    print("filling wte finished")

    # positional embedding
    print("filling wpe")
    wpe_weight = m0["module"]["position_embeddings.weight"].numpy()

    assert model_state_dict["transformer.wpe.weight"].shape == wpe_weight.shape

    model_state_dict["transformer.wpe.weight"] = torch.as_tensor(wpe_weight)
    assert np.array_equal(model_state_dict["transformer.wpe.weight"].numpy(), wpe_weight)
    print("filling wpe finished")

    # layer
    for layer in range(32):
        layer_name = "transformer.h.{}".format(layer)
        old_layer_name = "transformer.layers.{}".format(layer)

        # ln_1
        print("filling layer {} ln_1".format(layer))
        ln_1_weight = m0["module"][old_layer_name + ".input_layernorm.weight"].numpy()
        ln_1_bias = m0["module"][old_layer_name + ".input_layernorm.bias"].numpy()

        assert model_state_dict[layer_name + ".ln_1.weight"].shape == ln_1_weight.shape
        assert model_state_dict[layer_name + ".ln_1.bias"].shape == ln_1_bias.shape

        model_state_dict[layer_name + ".ln_1.weight"] = torch.as_tensor(ln_1_weight)
        model_state_dict[layer_name + ".ln_1.bias"] = torch.as_tensor(ln_1_bias)

        assert np.array_equal(model_state_dict[layer_name + ".ln_1.weight"].numpy(), ln_1_weight)
        assert np.array_equal(model_state_dict[layer_name + ".ln_1.bias"].numpy(), ln_1_bias)
        print("filling layer {} ln_1 finished".format(layer))

        # c_attn
        print("filling layer {} attn c_attn".format(layer))
        query_key_value_weight = np.concatenate([
            m0["module"][old_layer_name + ".attention.query_key_value.weight"].numpy()[:1280, :],
            m1["module"][old_layer_name + ".attention.query_key_value.weight"].numpy()[:1280, :],
            m0["module"][old_layer_name + ".attention.query_key_value.weight"].numpy()[1280:2560, :],
            m1["module"][old_layer_name + ".attention.query_key_value.weight"].numpy()[1280:2560, :],
            m0["module"][old_layer_name + ".attention.query_key_value.weight"].numpy()[2560:3840, :],
            m1["module"][old_layer_name + ".attention.query_key_value.weight"].numpy()[2560:3840, :],
        ]).transpose()
        query_key_value_bias = np.concatenate([
            m0["module"][old_layer_name + ".attention.query_key_value.bias"].numpy()[:1280],
            m1["module"][old_layer_name + ".attention.query_key_value.bias"].numpy()[:1280],
            m0["module"][old_layer_name + ".attention.query_key_value.bias"].numpy()[1280:2560],
            m1["module"][old_layer_name + ".attention.query_key_value.bias"].numpy()[1280:2560],
            m0["module"][old_layer_name + ".attention.query_key_value.bias"].numpy()[2560:3840],
            m1["module"][old_layer_name + ".attention.query_key_value.bias"].numpy()[2560:3840],
        ])

        assert model_state_dict[layer_name + ".attn.c_attn.weight"].shape == query_key_value_weight.shape
        assert model_state_dict[layer_name + ".attn.c_attn.bias"].shape == query_key_value_bias.shape

        model_state_dict[layer_name + ".attn.c_attn.weight"] = torch.as_tensor(query_key_value_weight)
        model_state_dict[layer_name + ".attn.c_attn.bias"] = torch.as_tensor(query_key_value_bias)

        assert np.array_equal(model_state_dict[layer_name + ".attn.c_attn.weight"].numpy(), query_key_value_weight)
        assert np.array_equal(model_state_dict[layer_name + ".attn.c_attn.bias"].numpy(), query_key_value_bias)
        print("filling layer {} attn c_attn finished".format(layer))

        # c_proj
        print("filling layer {} attn c_proj".format(layer))
        dense_weight = np.concatenate([
            m0["module"][old_layer_name + ".attention.dense.weight"].numpy(),
            m1["module"][old_layer_name + ".attention.dense.weight"].numpy(),
        ], axis=1).transpose()
        dense_bias = m0["module"][old_layer_name + ".attention.dense.bias"].numpy()

        assert model_state_dict[layer_name + ".attn.c_proj.weight"].shape == dense_weight.shape
        assert model_state_dict[layer_name + ".attn.c_proj.bias"].shape == dense_bias.shape

        model_state_dict[layer_name + ".attn.c_proj.weight"] = torch.as_tensor(dense_weight)
        model_state_dict[layer_name + ".attn.c_proj.bias"] = torch.as_tensor(dense_bias)

        assert np.array_equal(model_state_dict[layer_name + ".attn.c_proj.weight"].numpy(), dense_weight)
        assert np.array_equal(model_state_dict[layer_name + ".attn.c_proj.bias"].numpy(), dense_bias)
        print("filling layer {} attn c_proj finished".format(layer))

        # ln_2
        print("filling layer {} ln_2".format(layer))
        ln_2_weight = m0["module"][old_layer_name + ".post_attention_layernorm.weight"].numpy()
        ln_2_bias = m0["module"][old_layer_name + ".post_attention_layernorm.bias"].numpy()

        assert model_state_dict[layer_name + ".ln_2.weight"].shape == ln_2_weight.shape
        assert model_state_dict[layer_name + ".ln_2.bias"].shape == ln_2_bias.shape

        model_state_dict[layer_name + ".ln_2.weight"] = torch.as_tensor(ln_2_weight)
        model_state_dict[layer_name + ".ln_2.bias"] = torch.as_tensor(ln_2_bias)

        assert np.array_equal(model_state_dict[layer_name + ".ln_2.weight"].numpy(), ln_2_weight)
        assert np.array_equal(model_state_dict[layer_name + ".ln_2.bias"].numpy(), ln_2_bias)
        print("filling layer {} ln_2 finished".format(layer))

        # c_fc
        print("filling layer {} mlp c_fc".format(layer))
        c_fc_weight = np.concatenate([
            m0["module"][old_layer_name + ".mlp.dense_h_to_4h.weight"].numpy(),
            m1["module"][old_layer_name + ".mlp.dense_h_to_4h.weight"].numpy(),
        ], axis=0).transpose()
        c_fc_bias = np.concatenate([
            m0["module"][old_layer_name + ".mlp.dense_h_to_4h.bias"].numpy(),
            m1["module"][old_layer_name + ".mlp.dense_h_to_4h.bias"].numpy(),
        ], axis=0).transpose()

        assert model_state_dict[layer_name + ".mlp.c_fc.weight"].shape == c_fc_weight.shape
        assert model_state_dict[layer_name + ".mlp.c_fc.bias"].shape == c_fc_bias.shape

        model_state_dict[layer_name + ".mlp.c_fc.weight"] = torch.as_tensor(c_fc_weight)
        model_state_dict[layer_name + ".mlp.c_fc.bias"] = torch.as_tensor(c_fc_bias)

        assert np.array_equal(model_state_dict[layer_name + ".mlp.c_fc.weight"].numpy(), c_fc_weight)
        assert np.array_equal(model_state_dict[layer_name + ".mlp.c_fc.bias"].numpy(), c_fc_bias)
        print("filling layer {} mlp c_fc finished".format(layer))

        # c_proj
        print("filling layer {} mlp c_proj".format(layer))
        c_proj_weight = np.concatenate([
            m0["module"][old_layer_name + ".mlp.dense_4h_to_h.weight"].numpy(),
            m1["module"][old_layer_name + ".mlp.dense_4h_to_h.weight"].numpy(),
        ], axis=1).transpose()
        c_proj_bias = m0["module"][old_layer_name + ".mlp.dense_4h_to_h.bias"].numpy()

        assert model_state_dict[layer_name + ".mlp.c_proj.weight"].shape == c_proj_weight.shape
        assert model_state_dict[layer_name + ".mlp.c_proj.bias"].shape == c_proj_bias.shape

        model_state_dict[layer_name + ".mlp.c_proj.weight"] = torch.as_tensor(c_proj_weight)
        model_state_dict[layer_name + ".mlp.c_proj.bias"] = torch.as_tensor(c_proj_bias)

        assert np.array_equal(model_state_dict[layer_name + ".mlp.c_proj.weight"].numpy(), c_proj_weight)
        assert np.array_equal(model_state_dict[layer_name + ".mlp.c_proj.bias"].numpy(), c_proj_bias)
        print("filling layer {} mlp c_proj finished".format(layer))

    # ln_f
    print("filling ln_f")
    ln_f_weight = m0["module"]["transformer.final_layernorm.weight"].numpy()
    ln_f_bias = m0["module"]["transformer.final_layernorm.bias"].numpy()

    assert model_state_dict["transformer.ln_f.weight"].shape == ln_f_weight.shape
    assert model_state_dict["transformer.ln_f.bias"].shape == ln_f_bias.shape

    model_state_dict["transformer.ln_f.weight"] = torch.as_tensor(ln_f_weight)
    model_state_dict["transformer.ln_f.bias"] = torch.as_tensor(ln_f_bias)

    assert np.array_equal(model_state_dict["transformer.ln_f.weight"].numpy(), ln_f_weight)
    assert np.array_equal(model_state_dict["transformer.ln_f.bias"].numpy(), ln_f_bias)
    print("filling ln_f finished")

    # lm_head
    print("filling lm_head")
    assert model_state_dict["lm_head.weight"].shape == wte_weight.shape

    model_state_dict["lm_head.weight"] = torch.as_tensor(wte_weight)

    assert np.array_equal(model_state_dict["lm_head.weight"].numpy(), wte_weight)
    print("filling lm_head finished")

    # load state dict
    model.load_state_dict(model_state_dict)

    # save model
    model.save_pretrained(save_path)


def main():

    config = GPT2Config(
        vocab_size=30000,
        n_positions=1024,
        n_ctx=1024,
        n_embd=2560,
        n_layer=32,
        n_head=32,
        n_inner=4*2560,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        bos_token_id=30000,
        eos_token_id=30000,
        gradient_checkpointing=False,
    )

    print("initializing model")
    model = GPT2LMHeadModel(config)

    convert(
        model=model,
        m0_path="model-v1/80000/mp_rank_00_model_states.pt",
        m1_path="model-v1/80000/mp_rank_01_model_states.pt",
        save_path="model/CPM/",
    )


if __name__ == "__main__":
    main()

