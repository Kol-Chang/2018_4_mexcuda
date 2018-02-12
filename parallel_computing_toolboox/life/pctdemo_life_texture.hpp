/* Copyright 2013 The MathWorks, Inc. */

#ifndef PCTDEMO_LIFE_TEXTURE_HPP
#define PCTDEMO_LIFE_TEXTURE_HPP

#include "tmwtypes.h"
bool playGameOfLife(bool const * const pInitialBoard,
                    uint8_T * const d_board1,
                    uint8_T * const d_board2,
                    int const boardDim,
                    size_t const numGenerations);

#endif
