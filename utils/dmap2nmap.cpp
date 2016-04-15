// this code loads a displacement map and converts it to a normal map
// 
// g++ dmap2nmap.cpp -o dmap2nmap -lX11 -lpthread

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "CImg.h"
using namespace cimg_library;

/* Create a normal map from a displacement/bump map */
void
dmap2nmap(const CImg<uint8_t>& dmap, CImg<uint8_t>& nmap, float scale = 0.1f)
{
	int w = dmap.width();
	int h = dmap.height();

	nmap.resize(w, h, /*depth*/1, /*channels*/3);
	for (int i = 0; i < w; ++i)
	for (int j = 0; j < h; ++j) {
		uint8_t px_l = dmap.atXY(i - 1, j); // in [0,255]
		uint8_t px_r = dmap.atXY(i + 1, j); // in [0,255]
		uint8_t px_b = dmap.atXY(i, j + 1); // in [0,255]
		uint8_t px_t = dmap.atXY(i, j - 1); // in [0,255]
		float z_l = (float)px_l / 255.f; // in [0, 1]
		float z_r = (float)px_r / 255.f; // in [0, 1]
		float z_b = (float)px_b / 255.f; // in [0, 1]
		float z_t = (float)px_t / 255.f; // in [0, 1]
		float slope_x = (float)w * 0.5f * scale * (z_r - z_l);
		float slope_y = (float)h * 0.5f * scale * (z_t - z_b);
		float nrm_sqr = 1.f + slope_x * slope_x + slope_y * slope_y;
		float nrm_inv = 1.0 / sqrt(nrm_sqr);
		float nx = -slope_x * nrm_inv;
		float ny = -slope_y * nrm_inv;
		float nz = nrm_inv;
		float tmp1 = 0.5 * nx + 0.5; // in [0, 1]
		float tmp2 = 0.5 * ny + 0.5; // in [0, 1]

		nmap(i, j, 0, 0) = (uint8_t)(tmp1 * 255);
		nmap(i, j, 0, 1) = (uint8_t)(tmp2 * 255);
		nmap(i, j, 0, 2) = (uint8_t)(nz * 255);
	}
}

void usage(const char *appname)
{
	printf("\n%s -- displacement/bump map to normal map converter\n\n",
	       appname);
	printf("Usage\n"
	       " %s [OPTIONS] path_to_dmap\n\n", appname);
	printf("Options\n"
	       "-h --help\n"
	       "    Print help\n\n"
	       "--scale value\n"
	       "    Set the slope scaling factor\n"
	       "    (default is 0.01)\n\n"
	       "--clamp_to_border\n"
	       "    Set the image sample to clamp border pixels\n"
	       "    (default repeats)\n\n"
	       );
}

int main(int argc, const char **argv)
{
	const char *filename = NULL;
	bool clamp_to_border = false;
	float scale = 0.01f;
	CImg<uint8_t> dmap, nmap;

	// parse command line
	for (int i = 1; i < argc; ++i) {
		if (!strcmp("-h", argv[i]) || !strcmp("--help", argv[i])) {
			usage(argv[0]);
			return EXIT_SUCCESS;
		} else if (!strcmp("--clamp_to_border", argv[i])) {
			clamp_to_border = true;
		} else if (!strcmp("--scale", argv[i])) {
			scale = atof(argv[++i]);
		} else {
			filename = argv[i];
		}
	}

	// make sure an image file was specified
	if (!filename) {
		usage(argv[0]);
		return EXIT_FAILURE;
	}

	// load file
	dmap = CImg<uint8_t>(filename);

	// set up the sampler properly
	dmap.resize(dmap.width(),
	            dmap.height(),
	            dmap.depth(),
	            dmap.spectrum(),
	            /*nearest neighbour*/1,
	            clamp_to_border ? /*clamp*/1 : /*repeat*/2
	            );

	// convert
	dmap2nmap(dmap, nmap, scale);

	// save
	nmap.save("nmap.png");

	return EXIT_SUCCESS;
}


