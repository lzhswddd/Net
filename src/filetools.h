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
@brief getFiles 得到路径下所有文件的路径
@param path 文件夹路径
@param files 保存path下的所有文件路径
*/
void getFiles(string path, vector<string>& files);

#endif //__FILETOOLS_H__
