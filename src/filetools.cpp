#include "filetools.h"
#include <iostream> 
#include <io.h>
#include <direct.h>  
#include <stdio.h> 

void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	intptr_t hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

string createpath(string path, string dir)
{
	path = path + "\\" + dir + "\\";
	return path;
}

string createfile(string filename)
{
	return filename.substr(filename.rfind('\\') + 1);
}

//运行路径
string show_path()
{
	const int MAX_PATH = 256;
	char buffer[MAX_PATH];
	_getcwd(buffer, MAX_PATH);
	return string(buffer);
}
