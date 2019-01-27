#ifndef __FILETOOLS_H__
#define __FILETOOLS_H__

#include <string>  
#include <vector> 
using std::vector;
using std::string;

string createpath(string path, string dir);
string createfile(string filename);
string show_path();

/**
@brief getFiles �õ�·���������ļ���·��
@param path �ļ���·��
@param files ����path�µ������ļ�·��
*/
void getFiles(string path, vector<string>& files);

#endif //__FILETOOLS_H__
