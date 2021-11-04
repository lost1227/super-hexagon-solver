// SuperHexagonSolver.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>

#include "ParsedFrame.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    filesystem::path indir("C:\\Users\\jordan\\Documents\\git\\super-hexagon-solver\\in");
    filesystem::path outdir("C:\\Users\\jordan\\Documents\\git\\super-hexagon-solver\\out");

    filesystem::path outfile;

    Mat image;
    Mat bw;
    Mat plotted;
    Mat vis;

    filesystem::remove_all(outdir);
    filesystem::create_directory(outdir);

    filesystem::directory_iterator initer(indir);
    for (auto& dirent : initer) {
        if (!dirent.is_regular_file() || dirent.path().extension() != ".png") {
            continue;
        }
        const filesystem::path &infile = dirent.path();

        outfile = outdir / infile.filename();

        image = imread(infile.string());
        if (image.data == NULL)
        {
            cout << "Could not open or find the image " << infile.string() << endl;
            continue;
        }

        cvtColor(image, bw, COLOR_BGR2GRAY);

        ParsedFrame frame(bw);

        cvtColor(frame.getThresh(), plotted, COLOR_GRAY2BGR);

        vconcat(image, plotted, vis);

        imwrite(outfile.string(), vis);
    }
    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
