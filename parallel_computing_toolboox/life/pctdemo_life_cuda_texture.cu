/**
 * @file pctdemo_life_cuda_texture.cu
 * @brief Example of implementing a stencil operation on the GPU using texture memory.
 *
 * This file contains both host and device code. The MEX function that calls this
 * code is in pctdemo_life_mex_texture.cu.
 * Three features of this file enable the use of texture memory in the device function.
 * (1) The texture reference is declared at the top of the MEX-file.
 * (2) The CUDA device function fetches the input from the texture reference.
 * (3) The MEX-file binds the texture reference to the input buffer.
 *
 * Copyright 2013 The MathWorks, Inc.
 */

#include <algorithm>
#include <cuda_runtime_api.h>
#include "pctdemo_life_texture.hpp"

/**
 * (1) Declare the texture reference.
 * There is no texture reference to bool: we use uint8_T instead.
 */
texture<uint8_T, cudaTextureType1D> texRef;

/**
 * Calculate the proper value of the element at <row,col> on the
 * next board given the present board state.
 */
__device__
uint8_T gameOfLifeCalculation(unsigned int const row,
                              unsigned int const col,
                              unsigned int const N,
                              size_t const offset)
{
    // Work out the linear row and column indices for the previous/next
    // rows/columns, repeating the edge value at the boundary.
    unsigned int const top    = (row > 0) ? row-1 : 0;
    unsigned int const middle = (row < N) ? row : N-1;
    unsigned int const bottom = (row < (N-1)) ? row+1 : N-1;
    unsigned int const left   = (col > 0) ? (col-1)*N : 0;
    unsigned int const centre = (col < N) ? col*N : (N-1)*N;
    unsigned int const right  = (col < (N-1)) ? (col+1)*N : (N-1)*N;

    // Work out if this cell should be alive
    // (2) Fetch the input from the texture reference.
    bool const alive = static_cast<bool>(tex1Dfetch(texRef, offset + middle + centre));
    unsigned int const liveNeighbours =  tex1Dfetch(texRef, offset + top    + left)
        + tex1Dfetch(texRef, offset + middle + left)
        + tex1Dfetch(texRef, offset + bottom + left)
        + tex1Dfetch(texRef, offset + top    + centre)
        + tex1Dfetch(texRef, offset + bottom + centre)
        + tex1Dfetch(texRef, offset + top    + right)
        + tex1Dfetch(texRef, offset + middle + right)
        + tex1Dfetch(texRef, offset + bottom + right);

    // Finally, set the element of "newboard".
    return static_cast<uint8_T>((alive && (liveNeighbours == 2)) || (liveNeighbours == 3));
}


/**
 * One generation of the game of life. Calculates "newboard" from "board".
 * Both arrays are of assumed to be size NxN.
 */
__global__
void life(uint8_T * const newboard,
          uint8_T const * const board,
          unsigned int const N,
          size_t const offset)
{
    // Coordinates for this thread within the board
    unsigned int const row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int const col = blockIdx.y * blockDim.y + threadIdx.y;

    // Only threads inside the grid need to compute a result
    if ((row < N) && (col < N)) {
        // Game of life stencil computation
        newboard[row+col*N] = gameOfLifeCalculation(row, col, N, static_cast<uint32_T>(offset));
    }
}

/**
 * Host function called by MEX gateway. Sets up and calls the device function
 * for each generation.
 */

bool playGameOfLife(bool const * const pInitialBoard,
                    uint8_T * const d_board1,
                    uint8_T * const d_board2,
                    int const boardDim,
                    size_t const numGenerations)
{
    // Choose a reasonably sized number of threads in each dimension for the block.
    int const threadsPerBlockEachDim = 16;

    // Copy the initial values from the host to the first workspace gpuArray.
    int const boardBytes = boardDim*boardDim*sizeof(uint8_T);
    cudaMemcpy(d_board1, pInitialBoard, boardBytes, cudaMemcpyHostToDevice);

    // Compute the thread block and grid sizes based on the board dimensions.
    int const blocksPerGridEachDim = (boardDim + threadsPerBlockEachDim - 1) / threadsPerBlockEachDim;
    dim3 const dimBlock(blocksPerGridEachDim, blocksPerGridEachDim);
    dim3 const dimThread(threadsPerBlockEachDim, threadsPerBlockEachDim);

    // In each iteration, we treat one workspace as the input and one as the output.
    uint8_T * d_bufferIn  = d_board1;
    uint8_T * d_bufferOut = d_board2;

    // If numGenerations = 0, the output is the initial value.
    bool boardOneIsCurrentOutput = true;
    for (size_t ix = 0; ix < numGenerations; ++ix) {
        // (3) Bind the texture reference to the input workspace.
        size_t offset;
        cudaBindTexture(&offset, texRef, d_bufferIn, boardBytes);

        // Play one generation of the game.
        life<<<dimBlock, dimThread>>>(d_bufferOut, d_bufferIn, boardDim, offset);

        // Swap the input and output workspace pointers for the next generation.
        std::swap(d_bufferOut, d_bufferIn);

        // Keep track of the current output workspace.
        boardOneIsCurrentOutput = !boardOneIsCurrentOutput;

        // Undo the current texture binding so we leave things in a good state
        // for the next loop iteration or upon exiting.
        cudaUnbindTexture(texRef);
    }

    return boardOneIsCurrentOutput;
}
