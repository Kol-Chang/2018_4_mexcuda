/**
 * @file pctdemo_life_mex_texture.cpp
 * @brief MEX gateway for a stencil operation.
 * Copyright 2013 The MathWorks, Inc.
 *
 */

#include "tmwtypes.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "pctdemo_life_shmem.hpp"

/**
 * MEX gateway
 */
void mexFunction(int /* nlhs */, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    char const * const errId = "parallel:gpu:pctdemo_life_mex:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    // Initialize the MathWorks GPU API.
    mxInitGPU();

    if (nrhs!=2) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    // We expect to receive as input an initial board, consisting of CPU data of
    // MATLAB class 'logical', and a scalar double specifying the number of
    // generations to compute.
    mxArray const * const initialBoard = prhs[0];
    bool const * const pInitialBoard = static_cast<bool const *>(mxGetData(initialBoard));
    size_t const numGenerations = static_cast<size_t>(mxGetScalar(prhs[1]));

    if (mxGetClassID(initialBoard) != mxLOGICAL_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    mwSize const numDims = mxGetNumberOfDimensions(initialBoard);
    mwSize const * const dimArray = mxGetDimensions(initialBoard);

    // Create two workspace gpuArrays, square real arrays of the same size as
    // the input containing logical data. We will fill these with data, so leave
    // them uninitialized.
    mxGPUArray * const board1 = mxGPUCreateGPUArray(numDims, dimArray,
                                                    mxLOGICAL_CLASS, mxREAL,
                                                    MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray * const board2 = mxGPUCreateGPUArray(numDims, dimArray,
                                                    mxLOGICAL_CLASS, mxREAL,
                                                    MX_GPU_DO_NOT_INITIALIZE);

    bool * const d_board1 = static_cast<bool *>(mxGPUGetData(board1));
    bool * const d_board2 = static_cast<bool *>(mxGPUGetData(board2));

    int const boardDim = static_cast<int>(dimArray[0]);

    bool const boardOneIsOutput = playGameOfLife(pInitialBoard,
                                                 d_board1, d_board2,
                                                 boardDim, numGenerations);

    // Wrap the appropriate workspace up as a MATLAB gpuArray for return.
    if (boardOneIsOutput) {
        plhs[0] = mxGPUCreateMxArrayOnGPU(board1);
    } else {
        plhs[0] = mxGPUCreateMxArrayOnGPU(board2);
    }

    // The mxGPUArray pointers are host-side structures that refer to device
    // data. These must be destroyed before leaving the MEX function.
    mxGPUDestroyGPUArray(board1);
    mxGPUDestroyGPUArray(board2);
}
