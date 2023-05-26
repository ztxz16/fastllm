#pragma once
#include<windows.h>
#include<string>

std::string utf2Gb(const char* szU8)
{
	int wcsLen = ::MultiByteToWideChar(CP_UTF8, NULL, szU8, strlen(szU8), NULL, 0);
	wchar_t* wszString = new wchar_t[wcsLen + 1];
	::MultiByteToWideChar(CP_UTF8, NULL, szU8, strlen(szU8), wszString, wcsLen);
	wszString[wcsLen] = '\0';

	wcsLen = WideCharToMultiByte(CP_ACP, 0, wszString, -1, NULL, 0, NULL, NULL);
	char* gb2312 = new char[wcsLen + 1];
	memset(gb2312, 0, wcsLen + 1);
	WideCharToMultiByte(CP_ACP, 0, wszString, -1, gb2312, wcsLen, NULL, NULL);

	if (wszString)
		delete[] wszString;
	std::string gbstr(gb2312);
	if (gb2312)
		delete[] gb2312;
	return gbstr;
}

std::string Gb2utf(std::string ws)
{
	//int dwNum = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), -1, 0, 0, 0, 0);
	int dwNum = MultiByteToWideChar(CP_ACP, 0, ws.c_str(), -1, NULL, 0);

	wchar_t* wstr = new wchar_t[dwNum + 1];
	memset(wstr, 0, dwNum + 1);
	MultiByteToWideChar(CP_ACP, 0, ws.c_str(), -1, wstr, dwNum);

	dwNum = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
	char* utf8 = new char[dwNum + 1];
	memset(utf8, 0, dwNum + 1);
	WideCharToMultiByte(CP_UTF8, 0, wstr, -1, utf8, dwNum, NULL, NULL);
	if (wstr)
		delete[] wstr;
	std::string str(utf8);
	if (utf8)
		delete[]  utf8;
	return str;
}