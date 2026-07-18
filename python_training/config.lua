local config = {
    model = {
        vocab_size = 32000,
        hidden_size = 2048,
        num_hidden_layers = 22,
        num_attention_heads = 32,
        num_key_value_heads = 8,
        intermediate_size = 5632,
        seq_len = 512
    },
    training = {
        batch_size = 2, -- Reduced batch size for 1B model to prevent OOM
        lr = 0.0003,
        max_steps = 50000,
        checkpoint_interval = 1000,
        compile = false
    }
}

return config
