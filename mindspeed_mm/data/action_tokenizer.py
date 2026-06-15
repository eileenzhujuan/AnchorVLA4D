"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase, AutoProcessor, AutoTokenizer


class ActionTokenizer:

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 action_start_token_id,
                 bins: int = 256,
                 min_action: int = -1,
                 max_action: int = 1,
                 bin_file=None) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        if bin_file is not None:
            self.bins = np.load(bin_file)
            self.n_bins = self.bins.shape[0] - 1
        else:
            self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "action_token_begin_id" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_id = action_start_token_id
        self.tokenizer_max_size = self.action_token_begin_id + 1 + self.n_bins
        print(self.tokenizer.vocab_size, self.tokenizer_max_size,
              self.action_token_begin_id)

    def __call__(self, action_str: str) -> Union[str, List[str]]:
        action = np.array(eval(action_str))
        return (np.ones(1, np.int32) * self.action_token_begin_id).flatten().tolist() + self.tokenizer.encode('<|im_end|>')
        return (np.ones(action.shape, np.int32) * self.action_token_begin_id).flatten().tolist() + self.tokenizer.encode('<|im_end|>')
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action,
                         a_min=float(self.min_action),
                         a_max=float(self.max_action))
        if self.bins.ndim == 2:
            discretized_action = np.zeros(action.shape, np.int32)
            for _idx in range(self.bins.shape[1]):
                discretized_action[:,
                                   _idx] = np.digitize(action[:, _idx],
                                                       self.bins[:, _idx])
        else:
            discretized_action = np.digitize(action, self.bins)
        return (self.tokenizer_max_size - discretized_action
                ).flatten().tolist() + self.tokenizer.encode('<|im_end|>')

    def decode_token_ids_to_actions(
            self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer_max_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1,
                                      a_min=0,
                                      a_max=self.bin_centers.shape[0] - 1)
        if self.bins.ndim == 2:
            action_values = np.zeros(discretized_actions.shape, np.float32)
            for _idx in range(self.bins.shape[1]):
                action_values[_idx] = self.bin_centers[
                    discretized_actions[_idx], _idx]
        else:
            action_values = self.bin_centers[discretized_actions]
        return action_values

    @property
    def vocab_size(self) -> int:
        return self.n_bins


class FASTTokenizer:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 action_start_token_id,
                 bins: int = 256,
                 min_action: int = -1,
                 max_action: int = 1,
                 time_horizon: int = 1,
                 action_dim: int = 7,
                 fast_tokenizer_path: str = "physical-intelligence/fast") -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action
        self.time_horizon, self.action_dim = time_horizon, action_dim
        self._fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path, trust_remote_code=True)

        # [Contract] Set "action_token_begin_id" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_id = action_start_token_id
        self.tokenizer_max_size = self.action_token_begin_id + 1 + self.n_bins
        print(self.tokenizer.vocab_size, self.tokenizer_max_size,
              self.action_token_begin_id)

    def __call__(self, action_str: str) -> List[int]:

        actions = np.asarray(eval(action_str))
        # Tokenize actions with FAST tokenizer --> map to last tokens in text vocab
        dct_tokens = self._fast_tokenizer(actions[None])
        action_tokens = np.asarray(dct_tokens[0])
        ret_token = self.tokenizer.encode('Action: ') + (self.tokenizer_max_size - action_tokens
                ).tolist() + self.tokenizer.encode('<|im_end|>')
        return ret_token

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        decoded_text = self.tokenizer.decode(action_token_ids)
        if "Action: " not in decoded_text:
            return np.zeros((self.time_horizon, self.action_dim), dtype=np.float32)
        prefix = self.tokenizer.encode('Action: ')
        suffix = self.tokenizer.encode('<|im_end|>')
        action_tokens = action_token_ids[len(prefix): -len(suffix)]
        action_ids = np.asarray(action_tokens)
        discretized_actions = self.tokenizer_max_size - action_ids
        # batch_size is set to 1, by default
        action_values = self._fast_tokenizer.decode([discretized_actions], time_horizon=self.time_horizon, action_dim=self.action_dim)
        return action_values


if __name__ == '__main__':
    time_horizon = 2
    action_dim = 7
    qwenvl_tokenizer = AutoTokenizer.from_pretrained("/data/home/1701214120/models/saved_hf/Qwen2.5-VL-3B-vl-ar-fast-windowx-with-states-chunk20-9500")
    action_tokenizer = FASTTokenizer(qwenvl_tokenizer, action_start_token_id=151679, bins=256,
                                     time_horizon=time_horizon, action_dim=action_dim, fast_tokenizer_path="/data/home/1701214120/githubs/fast")

    action = np.random.rand(time_horizon, action_dim).astype(np.float32)
    action_str = str(action.tolist())
    tokens = action_tokenizer(action_str)
    actions_predict = action_tokenizer.decode_token_ids_to_actions(tokens)