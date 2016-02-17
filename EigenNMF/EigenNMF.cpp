// EigenNMF.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <stdlib.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "opencv2\opencv.hpp"

/*teranishi 2015/12/17
 V ≒ WH となるようにNMF

|| V - WH || を最小化するようにNMFを実行

 V   n x m 行列
 k   分解する要素数
 W  n x k 行列
 H  k x m 行列
 loop_max W, Hの更新回数
 k > 0
*/
using namespace std;
#define buffsize 99999999

double buff[buffsize];

Eigen::MatrixXd readMatrix(const char *filename)
{
	int cols = 0, rows = 0;

	//read numbers from file into buffer
	ifstream infile;
	infile.open(filename);
	while (!infile.eof())
	{
		string line;
		getline(infile, line);
		int temp_cols = 0;
		stringstream stream(line);
		while (!stream.eof())
		{
			stream >> buff[cols*rows + temp_cols++];
		}
		if (temp_cols == 0)
		{
			continue;
		}
		if (cols == 0)
		{
			cols = temp_cols;
		}
		rows++;
	}
	infile.close();

	Eigen::MatrixXd result(rows, cols);
	for (int i = 0;i < rows;i++)
	{
		for (int j = 0;j < cols;j++)
		{
			result(i, j) = buff[cols*i + j];
			//if (result(i, j) == 0 )result(i, j) = 1.0E-100;
			if (result(i, j) == 0|| result(i, j)>=40)result(i, j) = 1.0E-100;
		}
	}
	//cout << rows <<":"<< cols << endl;
	Eigen::MatrixXd transResult(cols, rows);
	//cout << "default:" <<endl<< result << endl;
	transResult = result.transpose();
	//cout <<"change:"<<endl <<transResult << endl;
	//cout << endl;
	return transResult;
}

void writeMatrix(Eigen::MatrixXd Wtrain, int k, int motionRows, int motionCols, int motion,string filename)
{

	Eigen::MatrixXd WtrainMotion = Wtrain.block(0, 0, motion, k);
	Eigen::MatrixXd WtrainLanguage = Wtrain.block(motion, 0, Wtrain.rows() - motion, k);


	ofstream ofsM("WtrainMotion.txt", std::ios::out | std::ios::app);
	ofstream ofsL("WtrainLanguage.txt", std::ios::out | std::ios::app);
	//WtrainMotion = WtrainMotion.transpose();

	for (int i = 0;i < WtrainMotion.cols();i++)
	{
		stringstream  ss;
		ss <<filename<< i << ".png";
		int heightCount = 0;
		int widthCount = 0;
		cv::Mat test(motionRows, motionCols, CV_8UC1);
		for (int j = 0;j < WtrainMotion.rows();j++)
		{
			ofsM << WtrainMotion(j, i) << "\t";
			//cout << WtrainMotion(j, i) << endl;
			if (widthCount == motionCols)
			{
				heightCount++;
				widthCount = 0;
			}
			//画像出力のため(test:100=input:1)の比率で画素値を計算
			//cout << j << ":(" << widthCount << ":" << heightCount << ")->" << WtrainMotion(j, i) << endl;
			test.at<unsigned char>(heightCount, widthCount) = static_cast<unsigned char>(WtrainMotion(j, i) * 100);//static_cast<double>(WtrainMotion(i, j));
			widthCount++;
		}
		ofsM << "\n";
		cv::imwrite(ss.str(), test);
	}
	for (int i = 0;i < WtrainLanguage.cols();i++)
	{
		for (int j = 0;j < WtrainLanguage.rows();j++)
		{
			ofsL << WtrainLanguage(j, i) << "\t";
		}
		ofsL << "\n";
	}
}

Eigen::MatrixXd NMFtrain(const Eigen::MatrixXd &V, unsigned int k, int loop_max)
{
	// W, Hをランダムな非負数で初期化
	Eigen::MatrixXd W = Eigen::MatrixXd::Random(V.rows(), k).cwiseAbs();
	//cout << W << endl;
	Eigen::MatrixXd H = Eigen::MatrixXd::Random(k, V.cols()).cwiseAbs();
	//cout << H << endl;
	//cout << W*H << endl;
	// W, Hを乗法的更新アルゴリズムで更新->二乗誤差
	for (unsigned int i = 0; i < loop_max; ++i) {
		H.array() = H.array() * (W.transpose() * V).array() / (W.transpose() * W * H).array();
		W.array() = W.array() * (V * H.transpose()).array() / (W * H * H.transpose()).array();
		//cout << H << endl;
		//cout << W << endl;
		//cout << "W*H=:" << endl;
		//cout << i << endl;
		//cout << W*H << endl;
		Eigen::MatrixXd ErrorMat = (V - (W*H)).array().pow(2);
		double Error = ErrorMat.sum();
		
		if (i == 9 || i == 49 || i == 99 || i == 499 || i == 999 || i == 9999 || i == 1999 || i == 2999 || i == 3999 || i == 4999 || i == 5999 || i == 6999 || i == 7999 || i == 8999 || i == 9999)
		{
			cout << i + 1 << endl;
			cout << Error << endl;
		}
		
	}

	//cout << "Wtrain=:" << endl;
	//cout << W << endl;
	//cout << "Htrain=:" << endl;
	//cout << H << endl;
	//cout << "W*H=:" << endl;
	//cout << W*H << endl;
	//cout << "Vtrain=:" << endl;
	//cout << V << endl;

	//二乗誤差の和を求める
	//Eigen::MatrixXd error = ((V) - (W * H));



	return W;
}

Eigen::MatrixXd NMFtest(Eigen::MatrixXd &Wtrain, const Eigen::MatrixXd &Vtest, unsigned int k, unsigned int motion, int loop_max)
{
	// W, Hをランダムな非負数で初期化
	Eigen::MatrixXd Htest = Eigen::MatrixXd::Random(k, Vtest.cols()).cwiseAbs();
	Eigen::MatrixXd WtrainMotion = Wtrain.block(0, 0, motion, k);
	Eigen::MatrixXd WtrainLanguage = Wtrain.block(motion, 0, Wtrain.rows() - motion, k);

	// Wtrain->motionを固定してHtestを乗法的更新アルゴリズムで求める->二乗誤差
	for (unsigned int i = 0; i < loop_max; ++i) {
		//W.array() = W.array() * (V * H.transpose()).array() / (W * H * H.transpose()).array();
		Htest.array() = Htest.array() * (WtrainMotion.transpose() * Vtest).array() / (WtrainMotion.transpose() * WtrainMotion * Htest).array();
		Eigen::MatrixXd ErrorMat = (Vtest - (Wtrain*Htest)).array().pow(2);
		double Error = ErrorMat.sum();
		//if (i == 9 || i == 49 || i == 99 || i == 499 || i == 999 ||i==1999||i==2999||i==3999||i==4999||i==5999||i==6999||i==7999||i==8999|| i == 9999)
		/*
		if (i == 0 || i == 1 || i == 2 || i == 3 || i == 4 || i == 5 || i == 6 || i == 7 || i == 8 || i == 9)
		{
			cout << i + 1 << endl;
			cout << Error << endl;
		}
		*/
	}
	//cout << "Wtrain->motion=:" << endl;
	//cout << WtrainMotion << endl;
	//cout << "Htest=:" << endl;
	//cout << Htest << endl;
	//cout << "W*H=:" << endl;
	//cout << WtrainMotion*Htest << endl;
	//cout << "Vtest->motion=:" << endl;
	//cout << Vtest << endl;

	//Htestを固定してWtrain->languageからVtest->languageを求める
	Eigen::MatrixXd VTestLanguage = WtrainLanguage * Htest;

	//WtrainMotionを記録

	return VTestLanguage;
}
int main()
{
	/*
	Eigen::MatrixXd TrainData(6, 8);
	TrainData <<
		9.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		9.0, 8.0, 0.0, 0.0, 6.0, 9.0, 9.0, 8.0,
		0.0, 0.0, 8.0, 7.0, 0.0, 0.0, 9.0, 8.0,
		1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
		0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0;
	Eigen::MatrixXd Wtrain = NMFtrain(TrainData, k, loop);

	Eigen::MatrixXd TestData(3, 4);
	TestData <<
		7.0, 0.0, 0.0, 0.0,
		7.0, 0.0, 8.0, 7.0,
		0.0, 10.0, 0.0, 7.0;
	*/

	int motionRows = 37;
	int motionCols = 61;
	int k =5;
	int MotionPart = (motionRows * motionCols);//今回は37*61のデータを使用しているため
	int loop = 1000;
	int LanguagePart = 7;
	int trainNum = 400;

	Eigen::MatrixXd TrainData = readMatrix("all2.txt");
	writeMatrix(TrainData, trainNum, motionCols, motionRows, MotionPart,"train");
	//cout << TrainData << endl;
	Eigen::MatrixXd Wtrain = NMFtrain(TrainData, k, loop);
	Eigen::MatrixXd TestData = readMatrix("cup2Test.txt");
	
	writeMatrix(Wtrain, k, motionCols, motionRows, MotionPart,"dictionary");
	//writeMatrix(TestData, 5);
	//cout << TestData << endl;
	Eigen::MatrixXd LinguisticPart = NMFtest(Wtrain, TestData, k, MotionPart, loop);

	cout << "Vtest->linguistic=" << endl;
	cout << LinguisticPart << endl;

	return 0;
}

