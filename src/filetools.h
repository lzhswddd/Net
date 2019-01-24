#ifndef __FILETOOLS_H__
#define __FILETOOLS_H__

#include <string>  
#include <vector> 
using namespace std;

string createpath(string path, string dir);
string createfile(string filename);
string show_path();
void getFiles(string path, vector<string>& files);

#endif //__FILETOOLS_H__
