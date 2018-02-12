/**
 * @file pctdemo_life_mex_shmem.cu
 * @brief Example of implementing a stencil operation on the GPU using shared memory.
 *
 * Copyright 2013 The MathWorks, Inc.
 */

#include <algorithm>
#include <cuda_runtime_api.h>
#include "pctdemo_life_shmem.hpp"

size_t const THREAD_BLOCK_DIM = 16;
size_t const SHARED_MEMORY_DIM = THREAD_BLOCK_DIM + 2;

/**
 * @brief Fill shared memory from the current board.
 *
 * @param board the input board to read from
 * @param row   row of the input board for the centre of the stencil
 * @param col   column of the input board for the centre of the stencil
 * @param M     number of rows in the input board
 * @param N     number of columns in the input board
 * @param shmem shared memory input array
 * @param sRow  row in shared memory centre of the stencil
 * @param sCol  column in shared memory for the centre of the stencil
 * @param sM    number of rows in shared memory
 * @param sN    number of columns in shared memory
 */
__device__ inline
void copyIntoSharedMemory(bool const * const board,
                          unsigned int const row, unsigned int const col,
                          unsigned int const M, unsigned int const N,
                          bool shmem[SHARED_MEMORY_DIM][SHARED_MEMORY_DIM],
                          unsigned int const sRow, unsigned int const sCol,
                          unsigned int const sM, unsigned int const sN)
{
    // Work out the linear row and column indices for the previous/next
    // rows/columns, repeating the edge value at the boundary.
    unsigned int const top    = (row > 0) ? row-1 : 0;
    unsigned int const middle = (row < M) ? row : M-1;
    unsigned int const bottom = (row < (M-1)) ? row+1 : M-1;
    unsigned int const left   = (col > 0) ? (col-1)*M : 0;
    unsigned int const centre = (col < N) ? col*M : (N-1)*M;
    unsigned int const right  = (col < (N-1)) ? (col+1)*M : (N-1)*M;

    // Work out the linear indices for the rows/columns of shared memory
    unsigned int const sTop    = sRow-1;
    unsigned int const sMiddle = sRow;
    unsigned int const sBottom = sRow+1;
    unsigned int const sLeft   = sCol-1;
    unsigned int const sCentre = sCol;
    unsigned int const sRight  = sCol+1;

    // First, load each thread's central element
    shmem[sMiddle][sCentre] = board[middle + centre];

    // Now fill the "halo" around the edges
    bool const isTopEdge    = (sRow == 1);
    bool const isBottomEdge = (sRow == sM-2);
    bool const isLeftEdge   = (sCol == 1);
    bool const isRightEdge  = (sCol == sN-2);

    // + top and bottom borders
    if (isTopEdge) {
        shmem[sTop][sCentre] = board[top + centre];
    } else if (isBottomEdge) {
        shmem[sBottom][sCentre] = board[bottom + centre];
    }

    // + left and right borders, including corners
    if (isLeftEdge) {
        shmem[sMiddle][sLeft] = board[middle + left];
        if (isTopEdge) {
            shmem[sTop][sLeft] = board[top + left];
        } else if (isBottomEdge) {
            shmem[sBottom][sLeft] = board[bottom + left];
        }
    } else if (isRightEdge) {
        shmem[sMiddle][sRight] = board[middle + right];
        if (isTopEdge) {
            shmem[sTop][sRight] = board[top + right];
        } else if (isBottomEdge) {
            shmem[sBottom][sRight] = board[bottom + right];
        }
    }
}


/**
 * Calculate the proper value of the element at <row,col> on the
 * next board given the present board state.
 */

__device__
bool gameOfLifeCalculation(bool shmem[SHARED_MEMORY_DIM][SHARED_MEMORY_DIM],
                           unsigned int const sRow, unsigned int const sCol,
                           unsigned int const sM)
{
    // Work out the linear indices for the rows/columns of shared memory
    unsigned int const sTop    = (sRow-1);
    unsigned int const sMiddle = sRow;
    unsigned int const sBottom = (sRow+1);
    unsigned int const sLeft   = (sCol-1);
    unsigned int const sCentre = sCol;
    unsigned int const sRight  = (sCol+1);

    // Work out if this cell should be alive
    bool const alive = shmem[sMiddle][sCentre];
    unsigned int const liveNeighbours = shmem[sTop][sLeft]
        + shmem[sMiddle][sLeft]
        + shmem[sBottom][sLeft]
        + shmem[sTop][sCentre]
        + shmem[sBottom][sCentre]
        + shmem[sTop][sRight]
        + shmem[sMiddle][sRight]
        + shmem[sBottom][sRight];

    // Finally, set the element of "newboard".
    return (alive && (liveNeighbours == 2)) || (liveNeighbours == 3);
}

/**
 * Device function to compute one generation of the game of life. Calculates
 * "newboard" from "board".  Both arrays are of assumed to be size NxN.
 */
__global__
void life(bool * const newboard,
          bool const * const board,
          unsigned int const N)
{
    // Coordinates for this thread within the board
    unsigned int const row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int const col = blockIdx.y * blockDim.y + threadIdx.y;
    // Coordinates for this thread within shared memory (prefix "s")
    unsigned int const sRow = threadIdx.x + 1;
    unsigned int const sCol = threadIdx.y + 1;
    unsigned int const sM = SHARED_MEMORY_DIM; // Num rows of shared memory
    unsigned int const sN = SHARED_MEMORY_DIM; // Num columns of shared memory

    __shared__ bool shmem[sM][sN];

    // 1. Load the old board into shared memory
    copyIntoSharedMemory(board,
                         row, col, N, N,
                         shmem,
                         sRow, sCol, sM, sN);

    // 2. Make sure all shared memory is loaded
    __syncthreads();

    // Only threads inside the grid need to compute a result
    if ((row < N) && (col < N)) {
        //3.  Game of life stencil computation
        newboard[row + col*N] = gameOfLifeCalculation(shmem, sRow, sCol, sM);
    }
}

/**
 * Host function called by MEX gateway. Sets up and calls the device function
 * for each generation.
 */
bool playGameOfLife(bool const * const pInitialBoard,
                    bool * const d_board1, bool * const d_board2,
                    int const boardDim, size_t const numGenerations)
{

    // Copy the initial values from the host to the first workspace gpuArray.
    int const boardBytes = boardDim*boardDim*sizeof(bool);
    cudaMemcpy(d_board1, pInitialBoard, boardBytes, cudaMemcpyHostToDevice);

    // Choose a reasonably sized number of threads in each dimension for the block.
    int const threadsPerBlockEachDim = THREAD_BLOCK_DIM;

    // Compute the thread block and grid sizes based on the board dimensions.
    int const blocksPerGridEachDim = (boardDim + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    dim3 const dimBlock(blocksPerGridEachDim, blocksPerGridEachDim);
    dim3 const dimThread(threadsPerBlockEachDim, threadsPerBlockEachDim);

    // In each iteration, we treat one workspace as the input and one as the output.
    bool *d_bufferIn  = d_board1;
    bool *d_bufferOut = d_board2;

    // If numGenerations = 0, the output is the initial value.
    bool boardOneIsCurrentOutput = true;
    for (size_t ix = 0; ix < numGenerations; ++ix) {
        // Play one generation of the game.
        life<<<dimBlock, dimThread>>>(d_bufferOut, d_bufferIn, boardDim);

        // Swap the input and output workspace pointers for the next generation.
        std::swap(d_bufferOut, d_bufferIn);

        // Keep track of the current output workspace.
        boardOneIsCurrentOutput = !boardOneIsCurrentOutput;
    }
    return boardOneIsCurrentOutput;
}
