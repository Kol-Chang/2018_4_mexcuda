/* Copyright 2013 The MathWorks, Inc. */
#ifndef PCTDEMO_LIFE_SHMEM_HPP
#define PCTDEMO_LIFE_SHMEM_HPP

bool playGameOfLife(bool const * const pInitialBoard,
                    bool * const d_board1,
                    bool * const d_board2,
                    int const boardDim,
                    size_t const numGenerations);
#endif
