// this code loads a displacement map and converts it to a normal map
// 
// g++ nmap2leanmap_biased.cpp -o nmap2leanmap_biased -I/usr/include/OpenEXR/ -lX11 -lpthread -lIlmImf -lz -lImath -lHalf -lIex -lIlmThread
//
// note: the data is biased to ensure that the components are all positive (Mitsuba does not support negative floating point texels)
//

#include <algorithm>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define cimg_use_openexr
#include "CImg.h"
using namespace cimg_library;

#define DJ_BRDF_IMPLEMENTATION 1
#include "../dj_brdf.h" 

#define BIAS (25.f)

/* Create a LEAN map from a normal map */
void
nmap2leanmap(
	const CImg<uint8_t>& nmap,
	CImg<float>& leanmap_1,
	CImg<float>& leanmap_2,
	float base_roughness = 1e-5
) {
	int w = nmap.width();
	int h = nmap.height();

	leanmap_1.resize(w, h, /*depth*/1, /*channels*/4);
	leanmap_2.resize(w, h, /*depth*/1, /*channels*/4);
	for (int i = 0; i < w; ++i)
	for (int j = 0; j < h; ++j) {
		uint8_t px_r = nmap.atXY(i, j, 0, 0); // in [0,255]
		uint8_t px_g = nmap.atXY(i, j, 0, 1); // in [0,255]
		uint8_t px_b = nmap.atXY(i, j, 0, 2); // in [0,255]
		float tmp1 = ((float)px_r / 255.f) * 2.0f - 1.0f; // in [-1, 1]
		float tmp2 = ((float)px_g / 255.f) * 2.0f - 1.0f; // in [-1, 1]
		float tmp3 = ((float)px_b / 255.f); // in [0, 1]
		float slope_x = -tmp1 / tmp3;
		float slope_y = -tmp2 / tmp3;
		float slope_x_sqr = slope_x * slope_x;
		float slope_y_sqr = slope_y * slope_y;
		float slope_xy = slope_x * slope_y;
		float base_roughness_sqr = 0.5f * base_roughness * base_roughness;

		if(-slope_x > BIAS || -slope_y > BIAS) {
			printf("slope = %f %f\n", slope_x, slope_y);
		}

		leanmap_1(i, j, 0, 0) = slope_x + BIAS; // E1
		leanmap_1(i, j, 0, 1) = slope_y + BIAS; // E2
		leanmap_1(i, j, 0, 2) = 1.f;
		leanmap_1(i, j, 0, 3) = 1.f;
		leanmap_2(i, j, 0, 0) = slope_x_sqr + base_roughness_sqr; // E3
		leanmap_2(i, j, 0, 1) = slope_y_sqr + base_roughness_sqr; // E4
		leanmap_2(i, j, 0, 2) = slope_xy + BIAS * BIAS;    // E5
		leanmap_2(i, j, 0, 3) = 1.f;
	}
}

/* Check the LEAN maps */
void
check_lean_maps(
	const CImg<float>& leanmap_1,
	const CImg<float>& leanmap_2
) {
	int w = leanmap_1.width();
	int h = leanmap_1.height();

	for (int i = 0; i < w; ++i)
	for (int j = 0; j < h; ++j) {
		float E1 = leanmap_1.atXY(i, j, 0, 0);
		float E2 = leanmap_1.atXY(i, j, 0, 1);
		float E3 = leanmap_2.atXY(i, j, 0, 0);
		float E4 = leanmap_2.atXY(i, j, 0, 1);
		float E5 = leanmap_2.atXY(i, j, 0, 2);
		djb::beckmann::lrep lrep(E1, E2, E3, E4, E5);
		djb::microfacet::params params;
		djb::beckmann::lrep_to_params(lrep, &params);
	}
}

/* Export to OpenEXR */
//void save_exr(const C)

void usage(const char *appname)
{
	printf("\n%s -- displacement/bump map to normal map converter\n\n",
	       appname);
	printf("Usage\n"
	       " %s [OPTIONS] path_to_dmap\n\n", appname);
	printf("Options\n"
	       "-h --help\n"
	       "    Print help\n\n"
	       "--base-roughness value\n"
	       "    Set the base roughness\n"
	       "    (default is 1e-5)\n\n"
	       );
}

int main(int argc, const char **argv)
{
	const char *filename = NULL;
	bool clamp_to_border = false;
	float base_roughness = 1e-5;
	CImg<uint8_t> nmap;
	CImg<float> leanmap_1, leanmap_2;

	// parse command line
	for (int i = 1; i < argc; ++i) {
		if (!strcmp("-h", argv[i]) || !strcmp("--help", argv[i])) {
			usage(argv[0]);
			return EXIT_SUCCESS;
		} else if (!strcmp("--base-roughness", argv[i])) {
			base_roughness = atof(argv[++i]);
			printf("note: base roughness set to %f\n", base_roughness);
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
	nmap = CImg<uint8_t>(filename);

	// convert
	nmap2leanmap(nmap, leanmap_1, leanmap_2, base_roughness);
	check_lean_maps(leanmap_1, leanmap_2);

	// save
	leanmap_1.save_exr("leanmap_1.exr");
	leanmap_2.save_exr("leanmap_2.exr");

	return EXIT_SUCCESS;
}


