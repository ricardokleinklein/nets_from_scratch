hparams = {
	# Network architecture
	'input_size': 4,
	'output_size': 3,
	'hidden_size': [50],
	'non_linear': 0, # 0 - Sigmoid, 1 - Relu (recommended)
	# Training hyperparameters
	'lr': 1e-3,
	'dropout': 0.4,
	'batch_size': 5,
	'test_size': 25,
	'epochs': 750,
}