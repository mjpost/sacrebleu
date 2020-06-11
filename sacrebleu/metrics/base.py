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


def get_signature(args, custom_dict):
    """Returns a signature dictionary by parsing the argparse results
    for common signatures and then updating it with custom signature dict
    for metric-specific data."""

    # Abbreviations for the signature
    # 'name': ('short name', 'arg key from argparse')
    argparse_map = {
        'test': ('t', 'test_set'),
        'lang': ('l', 'langpair'),
        'origlang': ('o', 'origlang'),
        'subset': ('S', 'subset'),
    }

    sig = {}

    # add version
    sig['version'] = ('v', __version__)

    for name, (short_name, arg_name) in argparse_map.items():
        # Add to signature only if available and not None
        value = args.__dict__.get(arg_name, None)
        if value:
            sig[name] = (short_name, value)

    for name, (short_name, value) in custom_dict.items():
        sig[name] = (short_name, value)

    return sig


class BaseScore:
    """A base score class to derive from."""
    def __init__(self, score):
        self.__sig_dict = None
        self.score = score

    def format(self, width=2, signed=True, short=False, score_only=False):
        raise NotImplementedError()

    def set_signature(self, sig_dict):
        self.__sig_dict = sig_dict

    def signature(self, short=False):
        """Returns a formatted signature string.

        :param short: If `True`, short version of the signature is returned.
        :return: A string object representing the signature.
        """
        pairs = []
        for name in sorted(self.__sig_dict.keys()):
            sname, value = self.__sig_dict[name]
            final_name = sname if short else name
            pairs.append('{}.{}'.format(final_name, value))

        return '+'.join(pairs)

    def __str__(self):
        return self.format()

    def __repr__(self):
        return self.format()
