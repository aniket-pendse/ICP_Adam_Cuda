// Device code for ICP computation 
// Currently working only on performing rotation and translation using cuda 


#ifndef _ICP_KERNEL_H_
#define _ICP_KERNEL_H_
#include <stdio.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include<vector>
#include<time.h>
#include<sys/time.h>
#include<ctime>
#include "dlib/optimization/optimization.h"
#include "dlib/optimization/find_optimal_parameters_abstract.h"
#include "dlib/optimization/optimization_bobyqa.h"
#include "dlib/optimization/find_optimal_parameters.h" 
#include "octree_code/octree.h"
#include "globals.h"


#define TILE_WIDTH 256 

__global__  void PerformRotationKernel( double * data_x, double * data_y, double * data_z, double * transformed_data_x, double * transformed_data_y, double * transformed_data_z, int n)
{

	// Create Matrices in the shared memory 
	
	__shared__ double data_x_s[TILE_WIDTH];
	__shared__ double data_y_s[TILE_WIDTH];
	__shared__ double data_z_s[TILE_WIDTH];
	
	// Thread allocation
	int tx = threadIdx.x ;
	int bx = blockDim.x*blockIdx.x;
	int index = bx + tx;
	// Generate rows and cols of Result point 
	if(index < n)
	{
		data_x_s[tx] = data_x[index];
		data_y_s[tx] = data_y[index];
		data_z_s[tx] = data_z[index];
	}
	
	__syncthreads();         // To ensure all elements of tile are loaded and consumed 

	// carrying out the rotation operation 
	double point_x = data_x_s[tx]*R_constant[0] + data_y_s[tx]*R_constant[1] + data_z_s[tx]*R_constant[2] + t_constant[0];
	double point_y = data_x_s[tx]*R_constant[3] + data_y_s[tx]*R_constant[4] + data_z_s[tx]*R_constant[5] + t_constant[1];
	double point_z = data_x_s[tx]*R_constant[6] + data_y_s[tx]*R_constant[7] + data_z_s[tx]*R_constant[8] + t_constant[2];

	if(index < n)
	{
		transformed_data_x[index] = point_x;
		transformed_data_y[index] = point_y;
		transformed_data_z[index] = point_z;
	}      
}


__global__ void CalculateDistanceIndexEachPoint(double point_x, double point_y, double point_z, double * data_x_d, double * data_y_d, double * data_z_d, int * bin_index_d, double * distance_d, int size_data)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if(index < size_data)
	{
		distance_d[index] = sqrt(pow(data_x_d[index] - point_x,2) + pow(data_y_d[index] - point_y,2) + pow(data_z_d[index] - point_z,2));
		bin_index_d[index] = index;
	}
	
}

__global__ void CalculateBestIndex(double * distance_d, int * bin_index_d, int size_data)
{
	__shared__ double distance_s[2*TILE_WIDTH];
	__shared__ unsigned int bin_smallest_index[2*TILE_WIDTH];
	unsigned int t = threadIdx.x;
	unsigned int start = 2*blockDim.x*blockIdx.x;
	
	if(start + t < size_data)
	{
		distance_s[t] = distance_d[start + t];
		bin_smallest_index[t] = bin_index_d[start + t];
	}
	else
	{
		distance_s[t] = 65535;
		bin_smallest_index[t] = 0;
	}
	if(start + blockDim.x + t < size_data)
	{
		distance_s[blockDim.x + t] = distance_d[start + blockDim.x + t];
		bin_smallest_index[blockDim.x + t] = bin_index_d[start + blockDim.x + t];
	}
	else
	{
		distance_s[blockDim.x + t] = 65535;
		bin_smallest_index[blockDim.x + t] = 0;
	}

	for(unsigned int stride = blockDim.x; stride >= 1; stride >>= 1)
	{
		__syncthreads();
		if(t < stride)
			if(distance_s[t] > distance_s[stride + t])
			{
				bin_smallest_index[t] = bin_smallest_index[stride + t];
				distance_s[t] = distance_s[stride + t];
			}
			
	}

	if(t == 0)
	{
		distance_d[blockIdx.x] = distance_s[t];
		bin_index_d[blockIdx.x] = bin_smallest_index[t];	
	}


}

__global__ void CalculateDistanceAllPoints(double * data_x_d, double * data_y_d, double * data_z_d, double * transformed_data_x_d, double * transformed_data_y_d, double * transformed_data_z_d, int * index_d, double * distance_d, int size_data)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < size_data)
	{
		int index = index_d[i];
		distance_d[i] = sqrt(pow(data_x_d[index] - transformed_data_x_d[i],2) + pow(data_y_d[index] - transformed_data_y_d[i],2) + pow(data_z_d[index] - transformed_data_z_d[i],2));
	}
}

__global__ void CalculateTotalError(double * distance_d, int size_data)
{
	__shared__ double error_s[2*TILE_WIDTH];

	unsigned int t = threadIdx.x;
        unsigned int start = 2*blockDim.x*blockIdx.x;

        if(start + t < size_data)   
    		error_s[t] = distance_d[start + t];
        else
		error_s[t] = 0.0f;
        if(start + blockDim.x + t < size_data)
    		error_s[blockDim.x + t] = distance_d[start + blockDim.x + t];
        else
		error_s[blockDim.x + t] = 0.0f;

	for(unsigned int stride = blockDim.x; stride >= 1; stride >>= 1)
	{
		__syncthreads();
		if(t < stride)
	    		error_s[t] += error_s[t + stride];
	}

	if(t == 0)
		distance_d[blockIdx.x] = error_s[t];

}














#endif // #ifndef _ICP_KERNEL_H_
