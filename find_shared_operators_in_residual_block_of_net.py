import torch
from models.mobilenet_v2 import MobileNetV2, InvertedResidual


class ChannelPruningEnv(object):
	def __init__(self):
		self.prunable_layer_types = [torch.nn.modules.conv.Conv2d, torch.nn.modules.linear.Linear]
		self.model = MobileNetV2(n_class=1000, input_size=224, width_mult=1.)
		self.prunable_idx = []  # prunable layers index
		self.prunable_ops = []  # prunable operators according to prunable layers
		self.connected_idx = [] # the connected index between prunable layers index
		self.shared_prunable_ops_index = [] # the prunable operators shared action


def find_shared_prunable_ops_index_in_residual_block(env):
	for i, m in enumerate(env.model.modules()):
		if type(m) in env.prunable_layer_types:

			if type(m) == torch.nn.Conv2d and m.groups == m.in_channels:  # depth-wise conv, skip
				continue
			else:  # really prunable
				env.prunable_idx.append(i)
				env.prunable_ops.append(m)
		elif type(m) == InvertedResidual:
			if m.use_res_connect:
				# env.connected_idx.append(i+2)
				# the "2" stand for skip the block(InvertedResidual),conv(Sequential) modules,\
				# which is vary from your net architecture definition.
				# env.connected_idx.append(i+len(list(m.modules()))+2)  # the "2" same as above comment.
				for j, _m in enumerate(m.modules()):
					if type(_m) in env.prunable_layer_types:
						break
				env.connected_idx.append(i+j)
				env.connected_idx.append(i+len(list(m.modules()))+j)
	env.connected_idx = sorted(list(set(env.connected_idx)))

	for c_idx in env.connected_idx:
		op_idx = env.prunable_idx.index(c_idx)
		env.shared_prunable_ops_index.append(op_idx)


if __name__ == "__main__":
	env = ChannelPruningEnv()
	find_shared_prunable_ops_index_in_residual_block(env)

	print("==>prunable_idx:{}".format(env.prunable_idx))
	print("==>connected_idx:{}".format(env.connected_idx))
	print("==>shared_prunable_ops_index:{}".format(env.shared_prunable_ops_index))
