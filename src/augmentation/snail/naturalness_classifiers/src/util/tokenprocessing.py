"""
Copyright 2024 Kyle Luoma

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def make_token_tag(identifier):
    """
    Feature engineering for identifiers, tags each character as a vowel, consonant, number, special character, or other.

    Args:
        identifier (str): The identifier to tag.

    Returns:
        str: A string of tag characters the same length as the input string.
    """
    vowels = ["a", "e", "i", "o", "u"]
    special = ["-", "_", "@"]
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    consonants = [
        "b",
        "c",
        "d",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        "m",
        "n",
        "p",
        "q",
        "r",
        "s",
        "t",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    tags = ""
    for c in identifier.lower():
        if c in vowels:
            tags += "^"
        elif c in special:
            tags += "$"
        elif c in numbers:
            tags += "#"
        elif c in consonants:
            tags += "+"
        else:
            tags += "*"
    return tags
