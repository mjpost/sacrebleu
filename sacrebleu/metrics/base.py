# -*- coding: utf-8 -*-

# Copyright 2017--2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


from .. import __version__


class BaseScore:
    """A base score class to derive from."""
    def __init__(self, score):
        self.score = score

    def format(self, width=2, score_only=False, signature=''):
        raise NotImplementedError()


class Signature:
    """A convenience class to represent sacreBLEU reproducibility signatures.

    :param args: The resulting `Namespace` returned from `parse_args()`.
    Argument-value pairs from command-line would then be directly added
    to the signature.
    """
    def __init__(self, args):
        # Copy the dictionary
        self.args = dict(args.__dict__)
        self.short = self.args.get('short', False)

        self._abbr = {
            'version': 'v',
            'test': 't',
            'lang': 'l',
            'subset': 'S',
            'origlang': 'o',
        }

        self._sig = {
            # None's will be ignored
            'version': __version__,
            'test': self.args.get('test_set', None),
            'lang': self.args.get('langpair', None),
            'origlang': self.args.get('origlang', None),
            'subset': self.args.get('subset', None),
        }

    def __str__(self):
        """Returns a formatted signature string."""
        pairs = []
        for name in sorted(self._sig.keys()):
            value = self._sig[name]
            if value is not None:
                final_name = self._abbr[name] if self.short else name
                pairs.append('{}.{}'.format(final_name, value))

        return '+'.join(pairs)

    def __repr__(self):
        return self.__str__()
