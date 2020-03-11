config = {
    'multichoice': True,
    'extractor_batch_size': 384,
    'model_name': 'multi_choice',
    'log_path': 'data/log',
    'tokenizer': 'nonword',  # 'nltk' 
    'batch_sizes':  (16, 24, 12),
    'lower': True,
    #'use_inputs': ['images', 'subtitle', 'speaker', 'que_len', 'ans_len', 'sub_len', 'sub_sentence_len', 'visual'],  # We advise not to use description for the challenge
    'use_inputs':['que','answers','subtitle','speaker','images','sample_visual','filtered_visual','filtered_sub','filtered_speaker','filtered_image','que_len','ans_len','sub_len','filtered_visual_len','filtered_sub_len','filtered_image_len','adjacency', 'filtered_person_full', 'filtered_person_full_len', 'q_level_logic'],
    'stream_type': ['script', 'visual_bb'], #
    'cache_image_vectors': True,
    'image_path': 'data/AnotherMissOh/AnotherMissOh_images',
    'visual_path': 'data/AnotherMissOh/AnotherMissOh_Visual.json',
    'data_path': 'data/AnotherMissOh/AnotherMissOh_QA/AnotherMissOhQA_set_script.jsonl',
    'subtitle_path': 'data/AnotherMissOh/AnotherMissOh_script.json',
    'glove_path': "data/glove.6B.300d.txt", # "data/glove.6B.50d.txt"
    'vocab_path': "data/vocab.pickle",
    'video_type': ['shot', 'scene'],
    'feature_pooling_method': 'mean',
    'max_epochs': 20,
    'allow_empty_images': False,
    'num_workers': 40,
    'image_dim': 512,  # hardcoded for ResNet50
    'n_dim': 300,
    'layers': 3,
    'dropout': 0.5,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'loss_name': 'cross_entropy_loss',
    'optimizer': 'adam',
    # 'metrics': ['bleu', 'rouge'],
    'metrics': [],
    'log_cmd': False,
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'text_feature_names': ['subtitle', 'description'],
    'max_sentence_len': 30,
    'mask': False,
    'max_sub_len': 300,
    'max_image_len': 100,
    'shuffle': (False, False, False),
}


debug_options = {
    # 'image_path': './data/images/samples',
}

log_keys = [
    'model_name',
    'feature_pooling_method',
]
