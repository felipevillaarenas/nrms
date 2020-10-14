hparams = {
    'batch_size': 32,
    'num_workers': 10,
    'shuffle': False,
    'lr': 5e-4,
    'name': 'ranger',
    'version': 'v1',
    'description': 'NRMS lr=5e-4, with weight_decay',
    'glove_path': './data/external/glove',
    'path_train_data' : './data/raw/MINDsmall_train',
    'path_val_data' : './data/raw/MINDsmall_dev',
    'path_test_data' : './data/raw/MINDsmall_dev',
    'max_vocab_size' : 40000,
    'model': {
        'dct_size': 'auto',
        'nhead': 10,
        'embed_size': 100,
        'encoder_size': 250,
        'v_size': 200
    },
    'data': {
        'pos_k': 50,
        'neg_k': 4,
        'maxlen': 20
    }
}