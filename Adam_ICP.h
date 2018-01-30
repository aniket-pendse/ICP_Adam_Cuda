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



typedef dlib::matrix<double,0,1> column_vector;

struct point_cloud_data{

	std::vector <double> x_coord;
	std::vector <double> y_coord;
	std::vector <double> z_coord;
	std::vector <int> index;
	int size ;	
};


///////////////////Function definitions////////////////////////////////

void cal_closest_points(const column_vector &rt);

double findTotalErrorInCloud(const column_vector &rt);

column_vector CalculateGradient(const column_vector rt, column_vector gradient);

column_vector AdamOptimizer(const column_vector rt);

double cal_norm(column_vector r);

dlib::matrix<double> pointwise_divide(dlib::matrix<double> m, dlib::matrix<double> v);

void PerformTransformationToAllPoints(dlib::matrix<double> R, dlib::matrix<double> t, point_cloud_data * data, point_cloud_data * transformed_data, int skips);

dlib::matrix<double> PerformRotation(dlib::matrix<double> R,dlib::matrix<double> t, dlib::matrix<double> point);
/////////////////////////////////////////////////////////////////////////////////////////



