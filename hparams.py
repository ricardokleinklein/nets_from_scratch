hparams = {
	# Network architecture
	'input_size': 64,
	'output_size': 10,
	'hidden_size': [120, 33],
	'non_linear': 0, # 0 - Sigmoid, 1 - Relu (recommended)
	# Training hyperparameters
	'lr': 1e-2,
	'dropout': 0.95,
	'batch_size': 5,
	'test_size': 40,
	'epochs': 15,
}