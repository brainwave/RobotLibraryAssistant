#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
using namespace std;
using namespace cv;

Mat src_1;
Mat src_2;
Mat src_new;

Mat src_orig;
Mat src_trans;
char* window_name="Image";

void surfAlgo (int, void*)
{
	int key=0, i=0;
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq *imageKeyPoints=0, *imageDescriptors = 0;
	vector<KeyPoint> keyPoints_orig, keyPoints_trans;
	Mat descriptor_orig, descriptor_trans;
	FlannBasedMatcher matcher;
	std::vector<DMatch>matches;
	double max_dist=0;double min_dist=100;
	std::vector < DMatch > good_matches;

	SurfFeatureDetector detector(5500);
	SurfDescriptorExtractor extractor;

	cvtColor (src_2,src_trans,CV_BGR2GRAY);
	cvtColor (src_1,src_orig,CV_BGR2GRAY);

	detector.detect(src_orig, keyPoints_orig);
	detector.detect(src_trans, keyPoints_trans);

	extractor.compute(src_orig, keyPoints_orig, descriptor_orig);
	extractor.compute(src_trans, keyPoints_trans, descriptor_trans);
	
	matcher.match(descriptor_orig, descriptor_trans, matches);
	
	for (int count=0;count<descriptor_orig.rows;count++)
	{
		double dist=matches[i].distance;
		if( dist < min_dist ) min_dist=dist;
		if (dist > max_dist) max_dist = dist;
	}

	for(int count = 0; count < descriptor_orig.rows; count++)
	{
		if( matches[count].distance < 3*min_dist )
			{
				good_matches.push_back( matches[count]);
			}
	}

	//drawKeypoints(src_trans, keyPoints_trans, src_trans, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//drawKeypoints(src_orig, keyPoints_orig, src_orig, Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);	
	
	//drawMatches ( src_orig, keyPoints_orig, src_trans, keyPoints_trans, matches, src_trans);

	drawMatches( src_orig, keyPoints_orig, src_trans, keyPoints_trans,good_matches,src_new,Scalar(255,0,255),Scalar(0,0,0),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow(window_name, src_new);
}	

int main(int argc, char** argv)
	
{
		
	VideoCapture vidcap= VideoCapture(argv[1]);
	cvNamedWindow (window_name,CV_WINDOW_AUTOSIZE); 
	
	vidcap.read(src_1);
		
	vidcap>>src_2;
	
	while(true)
	{
		vidcap >> src_2;
		if(src_2.empty())
			break;
		else
			surfAlgo(0,0);
		waitKey(10);
	}
	
	waitKey(0);	
	return 0;
}


