from .checkpointing import load_checkpoint_state, save_checkpoint_state


def facenet_stage1_main():
	from .facenet_stage1 import main

	return main()


def facenet_stage2_main():
	from .facenet_stage2 import main

	return main()


def main_stage_1():
	from .vit_stage1 import main_stage_1 as _main

	return _main()


def main_stage_2():
	from .vit_stage2 import main_stage_2 as _main

	return _main()

__all__ = [
	"facenet_stage1_main",
	"facenet_stage2_main",
	"load_checkpoint_state",
	"main_stage_1",
	"main_stage_2",
	"save_checkpoint_state",
]

