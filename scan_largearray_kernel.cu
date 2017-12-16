#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define TILE_SIZE 8
// You can use any other block size you wish.
#define BLOCK_SIZE 4


// Host Helper Functions (allocate your own data structure...)



// Device Functions



// Kernel Functions
__global__ void prescan(unsigned int *g_odata, unsigned int *g_idata, int n, unsigned int *offset,unsigned int width){
	__shared__ unsigned int scan_array[2*BLOCK_SIZE];

	unsigned int tx=threadIdx.x;

	unsigned int index=2*blockDim.x*blockIdx.x+threadIdx.x;
	unsigned int numTiles=width/(BLOCK_SIZE*2);
	if(width%(BLOCK_SIZE*2)){numTiles++;}
	unsigned int rowValspad=numTiles*(BLOCK_SIZE*2);
	//unsigned int row_index=index-(rowValspad-width);
	unsigned int fixIdx=(rowValspad-width)*(blockIdx.x/numTiles);
	unsigned int row_index=index-fixIdx-width*(blockIdx.x/numTiles);
	if(row_index<width){scan_array[tx]=g_idata[index-fixIdx];}
	else{scan_array[tx]=0;}
	if(row_index+blockDim.x<width){scan_array[tx+blockDim.x]=g_idata[index+blockDim.x-fixIdx];}
	else{scan_array[tx+blockDim.x]=0;}

/*
	unsigned int tx=threadIdx.x;
	unsigned int start=2*blockDim.x*blockIdx.x;

	scan_array[tx]=g_idata[tx+start];
	scan_array[tx+blockDim.x]=g_idata[tx+start+blockDim.x];
*/
	__syncthreads();
	int stride=1;
	while(stride<=BLOCK_SIZE){
		int index=(tx+1)*stride*2-1;
		if(index<2*BLOCK_SIZE){
			scan_array[index]+=scan_array[index-stride];
		}
		stride=stride<<1;
		__syncthreads();
	}

	if(tx==0){
		offset[blockIdx.x]=scan_array[2*blockDim.x-1];
		scan_array[2*blockDim.x-1]=0;
	}
	__syncthreads();
	stride=BLOCK_SIZE;
	while(stride>0){
		int index=(tx+1)*stride*2-1;
		if(index<2*BLOCK_SIZE){
			unsigned int temp=scan_array[index];
			scan_array[index]+=scan_array[index-stride];
			scan_array[index-stride]=temp;
		}	
		stride=stride>>1;
		__syncthreads();
	}

	if(row_index<width){g_odata[index-fixIdx]=scan_array[tx];}
        if(row_index+blockDim.x<width){g_odata[index+blockDim.x-fixIdx]=scan_array[tx+blockDim.x];}
/*
	g_odata[tx+start]=scan_array[tx];
	g_odata[tx+start+blockDim.x]=scan_array[tx+blockDim.x];
*/
}

__global__ void addOffset(unsigned int *g_odata, unsigned int *offset, unsigned int width){
	__shared__ unsigned int offTotal;
	unsigned int tx=threadIdx.x;
	/*unsigned int start=2*blockDim.x*(blockIdx.x+1);*/
	
	offTotal=0;	
	unsigned int index=2*blockDim.x*blockIdx.x+threadIdx.x;
        unsigned int numTilesRow=width/(BLOCK_SIZE*2);
        if(width%(BLOCK_SIZE*2)){numTilesRow++;}
        unsigned int rowValspad=numTilesRow*(BLOCK_SIZE*2);
        //unsigned int row_index=index-(rowValspad-width);
        unsigned int fixIdx=(rowValspad-width)*(blockIdx.x/numTilesRow);
        unsigned int row_index=index-fixIdx-width*(blockIdx.x/numTilesRow);
	unsigned int numiter=blockIdx.x%numTilesRow;
	if(blockIdx.x % numTilesRow){
	if(tx==0){
	for(int i=0;i<numiter;i++){
		offTotal+=offset[((blockIdx.x/numTilesRow)*numTilesRow)+i];
	}
	}
	__syncthreads();
	if(row_index<width){g_odata[index-fixIdx]=g_odata[index-fixIdx]+offTotal;}
	if(row_index+blockDim.x<width){g_odata[index+blockDim.x-fixIdx]=g_odata[index+blockDim.x-fixIdx]+offTotal;}
}
}


// **===-------- Modify the body of this function -----------===**
// You may need to make multiple kernel calls. Make your own kernel
// functions in this file, and then call them from here.
// Note that the code has been modified to ensure numElements is a multiple 
// of TILE_SIZE
void prescanArray(unsigned int *outArray, unsigned int *inArray, int numElements)
{
	unsigned int width=8;
	unsigned int numBlocks=width/(2*BLOCK_SIZE);
	if (width % (2*BLOCK_SIZE)){numBlocks++;}
	numBlocks*=(numElements/width);
	//unsigned int numBlocks=numElements/(2*BLOCK_SIZE);
	//if (numElements % BLOCK_SIZE*2) {numBlocks++;}
	unsigned int *offsetArray;
	cudaMalloc((void**)&offsetArray,numBlocks*sizeof(unsigned int));	
	//unsigned int width = 12;
	prescan<<<numBlocks,BLOCK_SIZE>>>(outArray,inArray,numElements,offsetArray,width);
	cudaDeviceSynchronize();
	
	addOffset<<<numBlocks,BLOCK_SIZE>>>(outArray,offsetArray,width);
	cudaDeviceSynchronize();

	cudaFree(offsetArray);




}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
