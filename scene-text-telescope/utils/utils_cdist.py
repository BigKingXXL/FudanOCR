import collections
import numpy as np
import torch
import re


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self.alphabet = alphabet  # for `-1` index
        self._ignore_case = ignore_case
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i

    def encode(self, text, to_np=True):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        #print(text)
        if isinstance(text, str):
            #print("isString")
            text = re.sub('[^0-9a-zA-Z]+', '', text)
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            text.insert(0,2)
            text.append(3)
            if to_np:
                return np.array(text)
            return text
        elif isinstance(text, collections.Iterable):
            #print("Is iteer")
            text = list(map(lambda x: self.encode(x, True), text))
        return text

    def decode(self, t, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        outs = []
        for el in t:
            outs.append(''.join([self.alphabet[i] for i in el if i != 0]))
        return outs
        
