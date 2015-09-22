//Required Header Files
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

//Global Variables
Mat src_1;
Mat src_2;
Mat src_new;

Mat src_orig;
Mat src_trans;
char* window_name="Image";

VideoWriter vidout;

//Definitoin of surfAlgo - Includes Flann Matching Implementation

void surfAlgo (int, void*)
{
	//Variables
	int key=0, i=0;
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq *imageKeyPoints=0, *imageDescriptors = 0;
	vector<KeyPoint> keyPoints_orig, keyPoints_trans;
	Mat descriptor_orig, descriptor_trans;
	double max_dist=0;double min_dist=100;
	std::vector < DMatch > good_matches;
	
	FlannBasedMatcher matcher;
	std::vector<DMatch>matches;

	//A low number of points, as we need to do IP on a moving webcam with limited computation
	SurfFeatureDetector detector(3000);
	SurfDescriptorExtractor extractor;

	//Grayscale Conversion, for easier IP
	cvtColor (src_2,src_trans,CV_BGR2GRAY);
	cvtColor (src_1,src_orig,CV_BGR2GRAY);

	//use Surf feature extraction
	detector.detect(src_orig, keyPoints_orig);
	detector.detect(src_trans, keyPoints_trans);

	extractor.compute(src_orig, keyPoints_orig, descriptor_orig);
	extractor.compute(src_trans, keyPoints_trans, descriptor_trans);
	
	matcher.match(descriptor_orig, descriptor_trans, matches);
	
	//Store only good points
	//This approach is inferior to use of masks, but chosen in favour of quicker implementation
	
	for (int count=0;count<descriptor_orig.rows;count++)
	{
		double dist=matches[i].distance;
		if( dist < min_dist ) min_dist=dist;
		if (dist > max_dist) max_dist = dist;
	}

	
	for(int count = 0; count < descriptor_orig.rows; count++)
	{
		if( matches[count].distance < 1.2*min_dist )
			{
				good_matches.push_back( matches[count]);
			}
	}

	drawMatches( src_orig, keyPoints_orig, src_trans, keyPoints_trans,good_matches,src_new,Scalar(255,0,255),Scalar(0,0,0),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	/*
	This section of the code is in the making

	
	//drawKeypoints(src_trans, keyPoints_trans, src_trans, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//drawKeypoints(src_orig, keyPoints_orig, src_orig, Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);	
	
	//drawMatches ( src_orig, keyPoints_orig, src_trans, keyPoints_trans, matches, src_trans);


	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int count=0; count<matches.size();count++)
	{
		obj.push_back(keyPoints_trans[ matches[count].queryIdx].pt);
		scene.push_back(keyPoints_orig[ matches[count].trainIdx].pt);
	}

	
	Mat H = findHomography( obj, scene, CV_RANSAC );
	std::vector<Point2f> object_corners(4);
  	std::vector<Point2f> scene_corners(4);


	object_corners[0] = cvPoint (0,0);
       	object_corners[1] = cvPoint (src_trans.cols,0);
	object_corners[2]= cvPoint (src_trans.cols,src_trans.rows);
	object_corners[3] = cvPoint (0,src_trans.rows);

	perspectiveTransform( object_corners, scene_corners, H);

	line (src_new, scene_corners[0]+Point2f(src_trans.cols,0), scene_corners[1]+Point2f(src_trans.cols,0),Scalar(0,255,0),4);
	line (src_new, scene_corners[1]+Point2f(src_trans.cols,0), scene_corners[2]+Point2f(src_trans.cols,0),Scalar(0,255,0),4);
	line (src_new, scene_corners[2]+Point2f(src_trans.cols,0), scene_corners[3]+Point2f(src_trans.cols,0),Scalar(0,255,0),4);
	line (src_new, scene_corners[3]+Point2f(src_trans.cols,0), scene_corners[0]+Point2f(src_trans.cols,0),Scalar(0,255,0),4);
*/

	//Resize and write to file
resize(src_new,src_new,Size(960,540),0,0,CV_INTER_LINEAR);
if(!src_new.empty())
{
	vidout<<src_new;
}
}	

int main(int argc, char** argv)
	
{
	
//Open video stream and window	
	VideoCapture vidcap= VideoCapture(argv[1]);
	cvNamedWindow (window_name,CV_WINDOW_AUTOSIZE); 

	int fourcc=CV_FOURCC('m','p','4','v');

	int fps=24;

	Size S= Size(960,540);
		
	vidout.open("/Users/brainwave/Desktop/tesvid.mov",fourcc,fps,S,true);
	if(!vidout.isOpened())
		{
			cout<<"error in file opening";
			return(0);
		}

//Source image
	src_1=imread(argv[2]);

	vidcap>>src_2;
//Perform continuous matching
	while(true)
	{
		vidcap >> src_2;
		if(src_2.empty())
			break;
		else
			surfAlgo(0,0);
	}

	vidout.release();
	waitKey(0);	
	return 0;
}


