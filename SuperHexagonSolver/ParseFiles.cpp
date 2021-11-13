#include "RunModes.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>

#include "ParsedFrame.h"

using namespace cv;
using namespace std;

int parseFiles() {
    filesystem::path indir("C:\\Users\\jordan\\Documents\\git\\super-hexagon-solver\\in");
    filesystem::path outdir("C:\\Users\\jordan\\Documents\\git\\super-hexagon-solver\\out");

    filesystem::path outfile;

    Mat image;
    Mat plotted;
    Mat vis;

    filesystem::remove_all(outdir);
    filesystem::create_directory(outdir);

    filesystem::directory_iterator initer(indir);
    for (auto& dirent : initer) {
        if (!dirent.is_regular_file() || dirent.path().extension() != ".png") {
            continue;
        }
        const filesystem::path& infile = dirent.path();

        outfile = outdir / infile.filename();

        image = imread(infile.string());
        if (image.data == NULL)
        {
            cout << "Could not open or find the image " << infile.string() << endl;
            continue;
        }
        ParsedFrame frame(image);

        plotted = frame.showPlottedPath();

        vconcat(image, plotted, vis);

        imwrite(outfile.string(), vis);
    }

    return 0;
}