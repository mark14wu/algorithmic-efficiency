from typing import Dict, Iterator, Tuple

from algorithmic_efficiency import spec


count = 0
batch_cache = []
batch_cache_init = False
prefetch_opt = False
def data_selection(
    workload: spec.Workload,
    input_queue: Iterator[Dict[str, spec.Tensor]],
    optimizer_state: spec.OptimizerState,
    current_param_container: spec.ParameterContainer,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparameters,
    global_step: int,
    rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  if not prefetch_opt:
    return next(input_queue)
  global count, batch_cache, batch_cache_init
  if not batch_cache_init:
    batch_cache_init = True
    for i in range(101):
      batch_cache.append(next(input_queue))
  if count < 101:
    assert batch_cache_init, "batch_cache not initialized"
    batch = batch_cache[count]
    count += 1
  else:
    batch = next(input_queue)
  return batch
