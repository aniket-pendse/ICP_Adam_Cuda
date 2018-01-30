// HOST Code to compute ICP for localization 



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
//#include "icp_gold.cpp"

extern "C"

using namespace std;

typedef dlib::matrix<double,0,1> column_vector;

__constant__ double R_constant[9];
__constant__ double t_constant[3];

#include "icp_kernel.cu"

// Function declarations
Matrix AllocateDeviceMatrix(const Matrix M);
Vector AllocateDeviceVector(const Vector V);



// Creating variables to store the measurement and model data
point_cloud_data measurement_data;
point_cloud_data model_data;

//CPU functions
dlib::matrix<double> PerformRotation(dlib::matrix<double> R,dlib::matrix<double> t, dlib::matrix<double> point);

void PerformTransformationToAllPoints(dlib::matrix<double> R, dlib::matrix<double> t, point_cloud_data * data, point_cloud_data * transformed_data, int skips);

void cal_closest_points_cpu(const column_vector &rt);

double findTotalErrorInCloud_cpu(const column_vector &rt);

dlib::matrix<double> compute_gold();
	

///// For initial testing purposes carrying out rotation and translation operation on cuda//////////////////

void cal_closest_points(Matrix rt);
double findTotalErrorInCloudOnDevice(const column_vector &rt_vec);


// Function to carry out Rotation of given point on the device 

void PerformRotationOnDevice(const Matrix R_h, const Matrix t_h, point_cloud_data * data, point_cloud_data * transformed_data)
{
	int size_data = data->size;

//***********Allocate Memory on device********************

	double * data_x_d;
	cudaMalloc((void**)&data_x_d, size_data*sizeof(double));
	double * data_y_d;
	cudaMalloc((void**)&data_y_d, size_data*sizeof(double));
	double * data_z_d;
	cudaMalloc((void**)&data_z_d, size_data*sizeof(double));

	double * transformed_data_x_d;
	cudaMalloc((void**)&transformed_data_x_d, size_data*sizeof(double));
	double * transformed_data_y_d;
	cudaMalloc((void**)&transformed_data_y_d, size_data*sizeof(double));
	double * transformed_data_z_d;
	cudaMalloc((void**)&transformed_data_z_d, size_data*sizeof(double));

	//Allocate temporary memory for x,y,x
	double * temp_x = (double*)malloc(size_data*sizeof(double));
	double * temp_y = (double*)malloc(size_data*sizeof(double));
	double * temp_z = (double*)malloc(size_data*sizeof(double));

//---------------------------------------------------------

//**************Copy data to Device and constant memory******
		
	cudaMemcpy(data_x_d, data->x_coord.data(), size_data*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(data_y_d, data->y_coord.data(), size_data*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(data_z_d, data->z_coord.data(), size_data*sizeof(double), cudaMemcpyHostToDevice);
	
	cudaMemcpyToSymbol(R_constant,R_h.elements,3 * 3*sizeof(double));
	cudaMemcpyToSymbol(t_constant, t_h.elements,3*sizeof(double));
	
//----------------------------------------------------------- 
	 
//******Setup the execution configuration*********************

	dim3 block, grid;
	block.x = TILE_WIDTH;
	block.y = 1;
	block.z = 1;
	
	if(size_data%block.x == 0)
		grid.x = size_data/block.x;
	else
		grid.x = size_data/block.x + 1;
	grid.y = 1;
	grid.z = 1;

//--------------------------------------------------------------

    // Launch the device computation threads!

	PerformRotationKernel<<<grid,block>>>(data_x_d, data_y_d, data_z_d, transformed_data_x_d, transformed_data_y_d, transformed_data_z_d, size_data);
		
	// Transfer Rotated Point from device to host
	
	cudaMemcpy(temp_x, transformed_data_x_d, size_data*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(temp_y, transformed_data_y_d, size_data*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(temp_z, transformed_data_z_d, size_data*sizeof(double), cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < size_data; i++)
	{
		transformed_data->x_coord.push_back(temp_x[i]);
		transformed_data->y_coord.push_back(temp_y[i]);
		transformed_data->z_coord.push_back(temp_z[i]);
		transformed_data->index.push_back(-1);
	}

	//cout<<"Value in transformed data "<<transformed_data->x_coord[0]<<endl;
	//cout<<"Value in Temp x value "<<temp_x[0]<<endl;
	//cout<<"Value in input data "<<data->x_coord[0]<<endl;

	transformed_data->size = data->size;
	
	
       // Free device memory for all
       cudaFree(data_x_d); cudaFree (data_y_d);cudaFree (data_z_d);	
	
}



// The main function 

int main()
{
	
	
	


	ifstream infile1;
  	infile1.open ("icp_model.csv");
	char* pEnd;
	string x,y,z;

	
	// Reading data from the model map data csv file 

	
	 while(!infile1.eof()){
		getline(infile1,x, ',');
		getline(infile1,y, ',');
		getline(infile1,z);
		//getline(infile,index);
		model_data.x_coord.push_back(strtod(x.c_str(),&pEnd));
		model_data.y_coord.push_back(strtod(y.c_str(),&pEnd));
		model_data.z_coord.push_back(strtod(z.c_str(),&pEnd));
		measurement_data.index.push_back(-1);
	}
	

	
	//Remove the last elements
	model_data.x_coord.pop_back();
	model_data.y_coord.pop_back();
	model_data.z_coord.pop_back();
	model_data.size = model_data.size - 1;
	
		
	//cout<<"Range x value "<<range_x<<endl;

	model_data.size = model_data.x_coord.size();
	//cout<<"model data value "<<model_data.size<<endl;
	
	//Rotational function test
	double theta = 0.03;
	double point_x = 0.003;
	double point_y = 0.005;
	double point_z = 0.0;
	Matrix R;
	Matrix t;
	R.width = 3;
	R.height = 3;
	t.width = 1;
	t.height = 3;
	// Allocating memory to the matrices 

	R.elements = (double*)malloc(R.width*R.height*sizeof(double));
	t.elements = (double*)malloc(t.width*t.height*sizeof(double));

	R.elements[0] = cos(theta);R.elements[1]= -sin(theta); R.elements[2]= 0;
	R.elements[3] =sin(theta);  R.elements[4]=cos(theta); R.elements[5]= 0;
	R.elements[6] = 0; R.elements[7]= 0; R.elements[8]= 1;
	
	t.elements[0] = point_x;
	t.elements[1]= point_y;
	t.elements[2] = point_z;
	
	
	// Generate mesasurement datra by rorating the model data
	PerformRotationOnDevice(R, t, &model_data, &measurement_data);

	


	//Calling closest point.
	Matrix  rt;
	rt.width =  1;
	rt.height =  4;
	column_vector rt_lower(4), rt_upper(4), rt_vec(4);
	rt.elements = (double*)malloc(rt.width*rt.height*sizeof(double));
	rt.elements[0] = 0.0;rt.elements[1] = 0.0;
	rt.elements[2] = 0.0;rt.elements[3] = 0;
	
	rt_vec = rt.elements[0], rt.elements[1], rt.elements[2], rt.elements[3];	
	rt_lower = -1.0, -1.0,-1.0,-1.0;
	rt_upper = 1.0, 1.0, 1.0, 1.0;

	double temp_error = 0;	
	double cpu_starttime , cpu_endtime;
	//temp_error = findTotalErrorInCloudOnDevice(rt_vec);
	//cpu_starttime = clock();
	//cal_closest_points(rt);
	//cpu_endtime = clock();
	//cout<<"The time taken for calculation of closest point = "<<((cpu_endtime - cpu_starttime)/CLOCKS_PER_SEC)<<endl;

	double final_error = 0;
	// time measurement variables 

	column_vector rt_gold(5);
	rt_gold = compute_gold();
	
	cout<<"Rt gold "<<rt_gold<<endl;


	cpu_starttime = clock();
	//double cpu_starttime , cpu_endtime;
	for(int i = 0; i<20; i++)
	{
		cout<<"iteration #: "<<i<<endl;
		
		cal_closest_points(rt);
		final_error = find_optimal_parameters(0.01, 0.000000001,100000, rt_vec, rt_lower, rt_upper,findTotalErrorInCloudOnDevice);
		//final_error = findTotalErrorInCloudOnDevice(rt_vec);
		
		cout<<"Rt parameters "<<rt_vec<<endl;
		cout<<"current error: "<<final_error<<endl;
		rt.elements[0] = rt_vec(0);
		rt.elements[1] = rt_vec(1);
		rt.elements[2] = rt_vec(2);
		rt.elements[3] = rt_vec(3);
		
	}
	cpu_endtime = clock();
	cout<<"The time taken for calculation = "<<((cpu_endtime - cpu_starttime)/CLOCKS_PER_SEC)<<endl;
	//cout<<"Error after optimization "<<final_error<<endl;
	





	return 0;
}


//Calculating the closest point
void cal_closest_points(Matrix rt)
{
	point_cloud_data transformed_data;

	Matrix R_h, t_h;
	R_h.width = 3;
	R_h.height = 3;
	t_h.width = 1;
	t_h.height = 3;
	// Allocating memory to the matrices 

	R_h.elements = (double*)malloc(R_h.width*R_h.height*sizeof(double));
	t_h.elements = (double*)malloc(t_h.width*t_h.height*sizeof(double));

	R_h.elements[0] = cos(rt.elements[0]);R_h.elements[1]= -sin(rt.elements[0]); R_h.elements[2]= 0;
	R_h.elements[3] = sin(rt.elements[0]);  R_h.elements[4]= cos(rt.elements[0]); R_h.elements[5]= 0;
	R_h.elements[6] = 0; R_h.elements[7]= 0; R_h.elements[8]= 1;
	
	t_h.elements[0] = rt.elements[1];
	t_h.elements[1] = rt.elements[2];
	t_h.elements[2] = rt.elements[3];


	PerformRotationOnDevice(R_h, t_h, &measurement_data, &transformed_data);

	//cout<<"x of measurement data "<<transformed_data.x_coord[0]<<endl;
	//cout<<"y of measurement data "<<transformed_data.y_coord[0]<<endl;
	//cout<<"z of measurement data "<<transformed_data.z_coord[0]<<endl;		

	//cout<<"x of transformed data "<<transformed_data.x_coord[0]<<endl;
	//cout<<"y of transformed data "<<transformed_data.y_coord[0]<<endl;
	//cout<<"z of transformed data "<<transformed_data.z_coord[0]<<endl;

	//Calculate the closest point
	double * x_coord_model_d;
	cudaMalloc((void**)&x_coord_model_d, model_data.size*sizeof(double));
	cudaMemcpy(x_coord_model_d, model_data.x_coord.data(), model_data.size*sizeof(double), cudaMemcpyHostToDevice);
	
	double * y_coord_model_d;
	cudaMalloc((void**)&y_coord_model_d, model_data.size*sizeof(double));
	cudaMemcpy(y_coord_model_d, model_data.y_coord.data(), model_data.size*sizeof(double), cudaMemcpyHostToDevice);
	
	double * z_coord_model_d;
	cudaMalloc((void**)&z_coord_model_d, model_data.size*sizeof(double));
	cudaMemcpy(z_coord_model_d, model_data.z_coord.data(), model_data.size*sizeof(double), cudaMemcpyHostToDevice);
	
	double * distance_d;
	cudaMalloc((void**)&distance_d, model_data.size*sizeof(double));

	int * index_d;	
	cudaMalloc((void**)&index_d, model_data.size*sizeof(int));

	
	
	for(int i = 0; i < transformed_data.size; i++)
	{	
		dim3 block, grid;
		block.x = TILE_WIDTH;
		block.y = 1;
		block.z = 1;
		if(transformed_data.size%block.x == 0)
			grid.x = transformed_data.size/block.x;
		else
			grid.x = transformed_data.size/block.x + 1;
		grid.y = 1;
		grid.z = 1;
		//cout<<"Check grid "<<grid.x<<endl;
		int size_data = model_data.size;
		double point_x = transformed_data.x_coord[i];
		double point_y = transformed_data.y_coord[i];
		double point_z = transformed_data.z_coord[i];
		
		CalculateDistanceIndexEachPoint<<<grid, block>>>(point_x, point_y, point_z, x_coord_model_d, y_coord_model_d, z_coord_model_d, index_d, distance_d, size_data);
	
		
		while(grid.x > 1)
		{	
			cudaDeviceSynchronize();
			//cout<<"Check grid 2 "<<grid.x<<endl;
			
			if(size_data%block.x == 0)
				grid.x = size_data/block.x;
			else
				grid.x = size_data/block.x + 1;
			
			CalculateBestIndex<<<grid,block>>>(distance_d, index_d, size_data);
			size_data = grid.x;
		}		
		cudaMemcpy(measurement_data.index.data() + i, index_d, sizeof(int), cudaMemcpyDeviceToHost);
	}

	cout<<"Index at some point "<<measurement_data.index[10]<<endl;
	cudaFree(x_coord_model_d);
	cudaFree(y_coord_model_d);
	cudaFree(z_coord_model_d);
	cudaFree(distance_d);
	cudaFree(index_d);
		



}









// Function to find the total error in cloud






double findTotalErrorInCloudOnDevice(const column_vector &rt_vec) 
{
	//iterations++;
	double icp_error = 0.0;
	point_cloud_data transformed_data;
	Matrix R, t;
        R.width = 3;R.height =3;t.height =3;t.width = 1;
	R.elements = (double*)malloc(R.width*R.height*sizeof(double));
	t.elements = (double*)malloc(t.width*t.height*sizeof(double));

	
	R.elements[0] = cos(rt_vec(0));R.elements[1] = -sin(rt_vec(0));R.elements[2] = 0; R.elements[3] = sin(rt_vec(0));R.elements[4] = cos(rt_vec(0));R.elements[5] = 0;
	R.elements[6] = 0; R.elements[7] = 0; R.elements[8] = 1;
	t.elements[0] = rt_vec(1);
	t.elements[1] =  rt_vec(2);
	t.elements[2] =  rt_vec(3);
	//cout<<"Check measurement data element "<<measurement_data.x_coord.at(0)<<endl;
	PerformRotationOnDevice(R, t, &measurement_data, &transformed_data);


	// Creating device variables 
	double * data_x_d;
	double * data_y_d;
	double * data_z_d;
	int * index_d;
	double * transformed_data_x_d;
	double * transformed_data_y_d;
	double * transformed_data_z_d;
	double * distance_d;	
	int size_data = transformed_data.size;
		


	cudaMalloc((void**)&data_x_d, size_data*sizeof(double));
	cudaMalloc((void**)&data_y_d, size_data*sizeof(double));
	cudaMalloc((void**)&data_z_d, size_data*sizeof(double));
	cudaMalloc((void**)&transformed_data_x_d, size_data*sizeof(double));
	cudaMalloc((void**)&transformed_data_y_d, size_data*sizeof(double));
	cudaMalloc((void**)&transformed_data_z_d, size_data*sizeof(double));
	cudaMalloc((void**)&distance_d, size_data*sizeof(double));
	cudaMalloc((void**)&index_d, size_data*sizeof(int));


	cudaMemcpy(data_x_d, model_data.x_coord.data(), size_data*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(data_y_d, model_data.y_coord.data(), size_data*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(data_z_d, model_data.z_coord.data(), size_data*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(transformed_data_x_d, transformed_data.x_coord.data(), size_data*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(transformed_data_y_d, transformed_data.y_coord.data(), size_data*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(transformed_data_z_d, transformed_data.z_coord.data(), size_data*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(index_d, measurement_data.index.data(), size_data*sizeof(int), cudaMemcpyHostToDevice);


	dim3 block, grid;
	block.x = TILE_WIDTH;
	block.y = 1;
	block.z = 1;
	if(size_data%block.x == 0)
		grid.x = size_data/block.x;
	else
		grid.x = size_data/block.x + 1;
	grid.y = 1;
	grid.z = 1;

	CalculateDistanceAllPoints<<<grid, block>>>(data_x_d, data_y_d, data_z_d, transformed_data_x_d, transformed_data_y_d, transformed_data_z_d, index_d, distance_d, size_data);
	while(grid.x > 1)
	{	
		cudaDeviceSynchronize();
		if(size_data%block.x == 0)
			grid.x = size_data/block.x;
		else
			grid.x = size_data/block.x + 1;

		CalculateTotalError<<<grid,block>>>(distance_d, size_data);
		size_data = grid.x;
	}
	
	cudaMemcpy(&icp_error, distance_d, sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(data_x_d);
	cudaFree(data_y_d);
	cudaFree(data_z_d);
	cudaFree(transformed_data_x_d);
	cudaFree(transformed_data_y_d);
	cudaFree(transformed_data_z_d);
	cudaFree(index_d);
	cudaFree(distance_d);
	
	return icp_error;
}

// Function to allocate matrix memory on the device
 
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    
    return Mdevice;
}
/*
Vector AllocateDeviceVector(std::vector<int> V)
{
    std::vector<int> Vdevice = V;
    int size = V.size() * sizeof(int);
    cudaMalloc((void**)&Vdevice, size);
    return Vdevice;
}

*/












	
