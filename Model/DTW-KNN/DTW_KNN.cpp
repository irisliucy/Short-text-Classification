// DTW.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
// **************************************************************
// *
// *  Preproccessing.cpp
// *
// *  Based on LB_Keogh and LSH algorithm, perform fast Dynamic Time Wrapping
// *
// **************************************************************
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <list>
#include <algorithm>
#include <cstdlib>
#include <utility>
#include <random>
#include "RTree_TwoPatterns_HD.h"
#include <queue>
#include <ctime>
#include <iterator>

using namespace std;


#define M 14997//5000 //The number of time series
#define T 128 //The length of a time serie 256
#define D 1//The dimension of a time point
#define bandwidth  0.12*T//Used for Sakoe-Chiba Band restriction, 0<=bandwidth<=T
//#define slope_variance 1 //Used for Itakura parallelogram restriction
#define constraint 4 //Itakura parallelogram window, must be smaller than T
#define PAAReductionFactor 1//the equal amount for each cell, must be a factor of T
#define L 3//The number of LSH function groups
#define K 6//The number of LSH functions in each group
#define W 1//Numerical constant
#define threshold 88//410
#define BlockNum T/PAAReductionFactor//The number of blocks left after using PAA
#define Epsilon 0.03*sqrt(D)*sqrt(T)//Threshold used in RTree RangeSearch
#define R 0.005*sqrt(D)
#define KNN 20
#define FLAG 50
#define LSHfunc 200
#define NUM 0.1*M
#define ratio 0.99
#define SD 0.01
#define ClusterNum 2

// r=0.04   t=140
// r=0.06   t=108

// T=128
// T=512


//compute Euclidean distance between two datasets with length N
float distance(float p[], float q[], int N)
{
	float dist = 0;
	for (int i = 0; i<N; i++)
	{
		dist += (p[i] - q[i])*(p[i] - q[i]);
	}
	return sqrt(dist);
}

//compute Euclidean distance between two points with dimension d
float distance_HD(float p[], float q[], int d)
{
	float dist = 0;
	for (int i = 0; i<d; i++) {
		dist += (p[i] - q[i])*(p[i] - q[i]);
	}
	return dist;
}


//compute Euclidean distance between two points with dimension d
float Euclidean_disatance(float p[], float q[], int d)
{
	float dist = 0;
	for (int i = 0; i<d; i++) {
		dist += (p[i] - q[i])*(p[i] - q[i]);
	}
	return sqrt(dist);
}


//compute Euclidean distance between two series with dimension d
float distance_HD(float** p, float** q)
{
	float dist = 0;
	for (int i = 0; i<T; i++) {
		for (int j = 0; j<D; j++) {
			dist += (p[i][j] - q[i][j])*(p[i][j] - q[i][j]);
		}
	}
	return sqrt(dist);
}

//load dataset from files
float***  load_data(const char* filename,int m, int timepoint, int dimension)

{
    ifstream file(filename); // declare file stream:
    string value;
    string num;
    int i,j,k;
    int count = -1;
    float ***data = new float**[m];
    
    for (i = 0; i<m; i++) {
        data[i] = new float*[timepoint];
        for (j = 0; j<timepoint; j++) {
            data[i][j] = new float[dimension+1];
        }
    }
    
    for ( i = 0; i<m; i++) {
        for ( j = 0; j<timepoint-1; j++) {
            for (k = 0; k<(dimension); k++) {
                getline(file, value, ' ');
                num = string(value, 0, value.length());
                data[i][j][k] = ::atof(num.c_str());
            }
        }
        for (k = 0; k<(dimension)-1; k++) {
            getline(file, value, ' ');
            num = string(value, 0, value.length());
            data[i][j][k] = ::atof(num.c_str());
        }
        getline(file,value);
        num = string(value);
        data[i][j][dimension-1]= atof(value.c_str());
        //getline(file, value, '\n');
    }
    file.close();
    return data;
}

float*  load_labels(const char* filename,int m)
{
    ifstream file(filename); // declare file stream:
    string value;
    string num;
    int i;
    int count = -1;
    float *labels = new float[m];
    
    for ( i = 0; i<m; i++) {
        getline(file,value);
        num = string(value);
        labels[i]= atof(value.c_str());
        //getline(file, value, '\n');
    }
    file.close();
    return labels;
}

//normalize input datasets to the range of [0,1]
void normalization_HD(float***&p) {
	float max[D] = { -INFINITY };
	float min[D] = { INFINITY };

	for (int d = 0; d<D; d++) {
		for (int i = 0; i<M; i++) {
			for (int j = 0; j<T; j++) {
				if (p[i][j][d] >= max[d])
					max[d] = p[i][j][d];
				if (p[i][j][d]<min[d])
					min[d] = p[i][j][d];
			}
		}
	}
	for (int i = 0; i<M; i++) {
		for (int j = 0; j<T; j++) {
			for (int d = 0; d<D; d++) {
				p[i][j][d] = (p[i][j][d] - min[d]) / (max[d] - min[d]);
			}
		}
	}

}


//Basic multiple dimensional DTW
float DTW_HD(float** p, float** q)
{
	float gamma[T][T];
	float dist[T][T];
	for (int i = 0; i<T; i++) {
		for (int j = 0; j<T; j++) {
			dist[i][j] = distance_HD(p[i], q[j], D); //no sqrt
		}
	}
	gamma[0][0] = dist[0][0];
	for (int i = 1; i<T; i++) {
		gamma[0][i] = dist[0][i] + gamma[0][i - 1];
		gamma[i][0] = dist[i][0] + gamma[i - 1][0];
		//gamma[i][0]=INFINITY;
	}
	float temp = 0;
	for (int i = 1; i<T; i++) {
		for (int j = 1; j<T; j++) {
			if ((j - i < bandwidth) && (j - i > -bandwidth))//Rectangle restriction
				gamma[i][j] = min(gamma[i - 1][j - 1], min(gamma[i - 1][j], gamma[i][j - 1])) + dist[i][j];
			else gamma[i][j] = dist[i][j];
			if (gamma[i][j] >= temp) {
				temp = gamma[i][j];
			}
		}
	}
	//cout<<gamma[95][95]<<endl;
	vector<pair<int, int> > pair_vector;
	int i = 0;
	int j = 0;
	while (i<T - 1 && j<T - 1) {
		if (i == T - 2 && j != T - 2) //
			j += 1;
		else if (j == T - 2 && i != T - 2)
			i += 1;
		else if (i == T - 2 && i == T - 2) {
			i += 1;
			j += 1;
		}
		else {
			if (gamma[i + 1][j + 1] - dist[i + 1][j + 1] == gamma[i + 1][j])
				i += 1;
			else if (gamma[i + 1][j + 1] - dist[i + 1][j + 1] == gamma[i][j + 1])
				j += 1;
			else {
				i += 1;
				j += 1;
			}
		}
		//pair_vector.push_back(make_pair(i,j));
		pair_vector.push_back(make_pair(i, j));
	}
	float cost = 0;
	for (int i = 0; i<pair_vector.size(); i++) {
		//cout << "Pair: "<<pair_vector[i].first << ", " << pair_vector[i].second << endl;
		cost = cost + distance_HD(p[pair_vector[i].first], q[pair_vector[i].second], D);
	}
	return sqrt(cost);
}

vector<int> DTW_GroundTruth_Range(float**query, float*** datasets) {
// find series which DTW<value and fall in the r-envelop of query series
	vector<int >candidate;
	float dtw_dist = 0;
	for (int i = 0; i<M; i++) {
		dtw_dist = DTW_HD(query, datasets[i]);
		if (dtw_dist <= Epsilon) 
		{
			bool isTrue = true;
			for (int j = 0; j<T; j++)
			{
				if (Euclidean_disatance(query[j], datasets[i][j], D)> R)
				{
					isTrue = false;
					break;
				}			
			}
			if (isTrue)			
			candidate.push_back(i);
		}
	}
	return candidate;
}

vector<int> DTW_GroundTruth_KNN(float** query, float*** datasets){
    struct sort_int_StoL {
        bool operator()(int left, int right) {
            return left < right;
        }
    };//Int from small to large
    struct sort_pred {
        bool operator()(const std::pair<int,float> &left, const std::pair<int,float> &right) {
            return left.second < right.second;
        }
    };
  
    vector<pair<int,float> > candidate_KNN;
    int count=0;
    for(int m=0;m<M;m++){
        if(count<KNN){ //find 20 K-neigbours
            candidate_KNN.push_back(make_pair(m,DTW_HD(query,datasets[m]))); //initialise first 20
        }
        else{
            sort(candidate_KNN.begin(),candidate_KNN.end(),sort_pred());
            float temp=DTW_HD(query,datasets[m]);
            if(temp<candidate_KNN.back().second){ //insertion
                candidate_KNN.pop_back(); //swap out
                candidate_KNN.push_back(make_pair(m,temp));
            }
        }
        count++;
    }
    vector<int> KNN_output;
    for(vector<pair<int,float> >::iterator it=candidate_KNN.begin();it!=candidate_KNN.end();++it){
        KNN_output.push_back((*it).first);
        cout<<"cost: "<<(*it).second<<endl;
    }
    sort(KNN_output.begin(), KNN_output.end(), sort_int_StoL());
    return KNN_output; // 20 index
}

float accuracy(vector<int> truth, vector<int> results)
{
	float accuracy;
	float count,acc_count;
	acc_count = 0;
	vector<int>::iterator it;
	vector<int>::iterator it2;

	for (it = truth.begin(); it != truth.end(); ++it)
	{   
		count = 0;
		for (it2 = results.begin(); it2 != results.end(); ++it2)
		{
			if ((*it) == (*it2))
			{
				count += 1;
			}
				
		}
		if (count != 0)
			acc_count += 1;
	}
	//cout << acc_count << endl;
	int size=truth.size();
	//cout << truth.size() << endl;
	accuracy = acc_count / size;
	return accuracy;
}

// float accuracy_KNN(vector<int> truth, vector<int> results, int query_id, float*** datasets)
// {
//     float sum_truth,sum_results=0;
    
//     for(int i=0;i<KNN;i++){
//         sum_truth+=DTW_HD(datasets[truth[i]],datasets[query_id]);
//         sum_results+=DTW_HD(datasets[results[i]],datasets[query_id]);
//     }
//     float accuracy=sum_truth/sum_results;
//     return accuracy;
// }

// float accuracy_KNN_classification(vector<int> truth, vector<int> results, int query_id, float*** datasets)
// {
//     int count_truth=0;
//     int count_results=0;
//     //cout<<"Query class type number: "<<datasets[query_id][0][D]<<endl;
//     for(int i=0;i<KNN;i++){
//         if(datasets[query_id][0][D]==datasets[truth[i]][0][D]){
//             count_truth++;
//         }
//         if(datasets[query_id][0][D]==datasets[results[i]][0][D]){
//             count_results++;
//         }
//     }
//     //cout<<"Ground Truth KNN classification number: "<<count_truth<<endl;
//     //cout<<"LSH KNN classification number: "<<count_results<<endl;
    
//     float accuracy=float(count_truth)/float(count_results);
//     return accuracy;
// }


int main()
{
    /*load data*/
    float*** datasets=new float**[M];
    float*** train=load_data("./20news_1000D_train_output.txt",11997,T,D);
    float* train_labels=load_labels("./20news_1000D_train_output.txt",11997);
    float*** test=load_data("./20news_1000D_test_output.txt",3000,T,D);
    float* test_labels=load_labels("./RNN/TwoPatterns_test_labels.txt",3000);
    for(int i=0;i<M;i++){
        datasets[i] = new float*[T];
        for (int j = 0; j<T; j++) {
            datasets[i][j] = new float[D+1];
        }
    }
    for(int i=0;i<11997;i++){
        for(int j=0;j<T;j++){
            train[i][j][D]=train_labels[i];
        }
    }
    for(int i=0;i<3000;i++){
        for(int j=0;j<T;j++){
            test[i][j][D]=test_labels[i];
        }
    }
    for(int i=0;i<11997;i++){
        for (int j = 0; j<T; j++) {
            for(int k=0;k<D+1;k++){
                datasets[i][j][k] = train[i][j][k];
            }
        }
    }
    for(int i=11997;i<M;i++){
        for (int j = 0; j<T; j++) {
            for(int k=0;k<D+1;k++){
                datasets[i][j][k] = test[i-11997][j][k];
            }
        }
    }
    
  
    //normalization
    normalization_HD(datasets);
	cout << "finish normalization" << endl;


	//cout << "LB_PAA Test: " << compute_LB_PAA_HD(datasets[0], datasets[2]) << endl;
 //    clock_t DTW_loopBegin=clock();
 //    double DTW_loopEnd = 0;
 //    for (int i=0; i<M; i++){
 //    	for (int j=0; j< M; j++){
	// 	    clock_t DTW_onceBegin=clock();
	// 	    if (i == j) continue;
	// 		cout << "DTW_HD Test: " << DTW_HD(datasets[i], datasets[j]) << endl;
	// 	    clock_t DTW_onceEnd=clock();
	// 	    double DTW_once = double(DTW_onceEnd - DTW_onceBegin) / CLOCKS_PER_SEC;
	// 	    double DTW_loopEnd += DTW_once;
	// 	    cout << "The time for DTW calculation once: " << DTW_once<< " seconds." << endl;
	// 		cout << "Euclidean distance Test: " << distance_HD(datasets[i], datasets[j])<< endl;
	// 	    cout<<"/****************************************************************************/"<<endl;
 //    	}
	// }
	
	// cout << "The time for DTW calculation once: " << DTW_loop<< " seconds." << endl;
    
    //DTW KNN Query Ground Truth
    cout<<"DTW KNN Query Ground Truth: "<<endl;
    clock_t beginDTWKNN = clock();
    int countDTWKNN=0;
    vector<int> DTW_groundtruth_KNN=DTW_GroundTruth_KNN(datasets[query_id],datasets);
    clock_t endDTWKNN = clock();
    for(vector<int>::iterator it=DTW_groundtruth_KNN.begin();it!=DTW_groundtruth_KNN.end();++it){
        cout<<"Candidate series number for DTW KNN Query ground truth: "<<(*it)<<endl;
        countDTWKNN++;
    }
    cout<<"The total number of candidate series for DTW KNN Query: "<<countDTWKNN<<endl;
    double elapsed_secsDTWKNN = double(endDTWKNN - beginDTWKNN) / CLOCKS_PER_SEC;
    cout<<"The time spent for DTW KNN Query  ground truth: "<< elapsed_secsDTWKNN<<" seconds."<<endl;
    cout<<"/****************************************************************************/"<<endl;

	for (int i = 0; i < M; i++) {
		for (int j = 0; j<T; j++) {
			delete[]datasets[i][j];
		}
		delete[] datasets[i];
	}
	delete[] datasets;

	return 0;
}


