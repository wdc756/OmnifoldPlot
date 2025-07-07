# This file will mainly be a data container for plot/data options

# Dictionary:
"""



"""

# Used to get the base dir
import os

# Used to create easier-to-read options
from enum import Enum

# Used to create dynamic file names
from src.util.file_patterns import TokenType, Token, FilePattern



# Plot type options - main control
##############################
class PlotType(Enum):
    debug = 0
    ALL = 1

plot_type = PlotType.debug


# Binning options
##############################
bin_start = 0
bin_end = 80
bin_step = 20


# Data options
##############################
base_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = '../../data'
nat_data_dir = 'mock'
syn_data_dir = 'mock'
weights_dir = 'weights'
plots_dir = 'plots'

nat_data_file_pattern = FilePattern([
    Token(TokenType.STATIC, 'mockdata.nat.Logweighted2.N150000.root'),
])
syn_data_file_pattern = FilePattern([
    Token(TokenType.STATIC, 'mockdata.syn'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 2, 1)),
    Token(TokenType.STATIC, '.'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 5, 1)),
    Token(TokenType.STATIC, 'Percent.Logweighted2.N150000.root'),
])
weights_file_pattern = FilePattern([
    Token(TokenType.STATIC, 'Syn'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 2, 1)),
    Token(TokenType.STATIC, '_'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 5, 1)),
    Token(TokenType.STATIC, 'Percent_Test'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 10, 1)),
    Token(TokenType.STATIC, '.npy')
])
plots_file_pattern = FilePattern([
    Token(TokenType.STATIC, 'plot.'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 2, 1)),
    Token(TokenType.STATIC, '.'),
    Token(TokenType.INCREMENTAL, '', 1, (1, 5, 1)),
    Token(TokenType.STATIC, 'Percent.Logweighted2.N150000.png'),
])


#
##############################