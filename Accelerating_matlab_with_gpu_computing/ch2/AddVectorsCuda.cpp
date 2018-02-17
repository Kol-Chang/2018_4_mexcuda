#include "mex.h"
#include "AddVectors.h"
#include "cuda_runtime.h"

__global__  void addVectorsMask(float* A, float*B, float*C, int size)
{
	int i = blockIdx.x;
	if(i >= size)
		return;

	C[i] = A[i] + B[i];
}


void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{	

	if(nrhs != 2)
		mexErrMsgTxt("Invalid number of input arguments");

	if(nlhs != 1)
		mexErrMsgTxt("Invalid number of outputs");

	if(!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]))
		mexErrMsgTxt("input vectors dat type must be single");

	int numRowsA = (int)mxGetM(prhs[0]);
	int numColsA = (int)mxGetN(prhs[0]);
	int numRowsB = (int)mxGetM(prhs[1]);
	int numColsB = (int)mxGetN(prhs[1]);

	if(numRowsA != numRowsB || numColsA != numColsB)
		mexErrMsgTxt("Invalid size. The size of the two vectors must be the same");

	int minSize = (numRowsA<numColsA) ? numRowsA : numColsA;
	int maxSize = (numRowsA>numColsA) ? numRowsA : numColsA;

	if(minSize != 1)
		mexErrMsgTxt("Invalid size. The vector must be one dimentional");

	float *A = (float*)mxGetData(prhs[0]);
	float *B = (float*)mxGetData(prhs[1]);

	plhs[0] = mxCreateNumericMatrix(numRowsA, numColsB, mxSINGLE_CLASS, mxREAL);
	float *C = (float*)mxGetData(plhs[0]);

	addVectors(A, B, C, maxSize);

	
}