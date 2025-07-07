# This file contains data class definitions to help with file selection

# Used to build file paths
import os

# Used to create easier-to-read options
from enum import Enum

# Used to define the dataclasses
from dataclasses import dataclass, field

# Used to define the range
from typing import List, Tuple



# Used to define Token types so the right data is used
class TokenType(Enum):
    STATIC = 0
    INCREMENTAL = 1
    PHRASES = 2

# Used to help FilePattern
@dataclass
class Token:
    type: TokenType
    literal: str = ""
    num: int = 0
    increment: Tuple[int, int, int] = (0, 0, 1)
    phrases: List[str] = field(default_factory=list)



# Used to help manage large amounts of files
@dataclass
class FilePattern:
    tokens: list[Token]

    # Returns the full name for the current values
    def get_full_pattern(self):
        parts = []

        for token in self.tokens:
            if token.type == TokenType.STATIC:
                parts.append(token.literal)

            elif token.type == TokenType.INCREMENTAL:
                parts.append(token.literal + str(token.num))

            elif token.type == TokenType.PHRASES:
                parts.append(token.literal + token.phrases[token.num])

        return "".join(parts)

    # Increments the numbers that can be incremented (if there are any), from right to left
    # Returns a 1 if the pattern was incremented, 0 if not (which means the pattern is done or is only STATIC)
    def increment(self):
        for token in reversed(self.tokens):

            if token.type == TokenType.INCREMENTAL or token.type == TokenType.PHRASES:

                if token.num >= token.increment[1]:
                    token.num = token.increment[0]
                    continue

                else:
                    token.num += token.increment[2]
                    return True

        return False

    # Opposite functionality of increment
    def decrement(self):
        for token in reversed(self.tokens):

            if token.type == TokenType.INCREMENTAL or token.type == TokenType.PHRASES:

                if token.num <= token.increment[0]:
                    token.num = token.increment[1]
                    continue

                else:
                    token.num -= token.increment[2]
                    return True

        return False

    # Returns an array of all the current incremental numbers
    def get_incremental_numbers(self):
        numbers = []
        for token in self.tokens:
            if token.type == TokenType.INCREMENTAL or token.type == TokenType.PHRASES:
                numbers.append(token.num)
        return numbers


def get_file_path(file_pattern: FilePattern, folder: str = ""):
    # Used for the working dir, placed here to avoid circular import
    import src.util.options as op
    return os.path.join(op.base_dir, op.data_dir, folder, file_pattern.get_full_pattern())