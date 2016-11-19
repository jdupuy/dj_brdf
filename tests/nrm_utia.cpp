/* 
 * This code tests if the utia BRDFs given as arguments integrate to one or 
 * less, i.e., if they are energy conserving.
 *
 * g++ -O3 nrm_utia.cpp -o nrm_utia
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define DJ_BRDF_IMPLEMENTATION 1
#include "../dj_brdf.h"

void usage(char *appname)
{
	printf("%s path_to_utia_file1 path_to_utia_file1 ...\n", appname);
}

bool furnace(const djb::brdf& fr) {
	const int ntheta = 64;
	const int nphi = 256;
	const float dtheta = (M_PI / 2.0) / (float)ntheta;
	const float dphi   = (M_PI * 2.0) / (float)nphi;
	const float dtheta_dphi = dtheta * dphi;
	bool success = true;

	for (int i1 = 0; i1 < ntheta && success; ++i1)
	for (int i2 = 0; i2 < nphi && success; ++i2) {
		float u1 = (float)i1 / (float)ntheta;
		float u2 = (float)i2 / (float)nphi;
		float theta = u1 * M_PI / 2.0;
		float phi = u2 * M_PI * 2.0;
		djb::vec3 o = djb::vec3(theta, phi);
		djb::vec3 nint = djb::vec3(0);

		for (int j1 = 0; j1 < ntheta; ++j1) {
			float u = (float)j1 / (float)ntheta;
			float theta = u * M_PI / 2.0;
			for (int j2 = 0; j2 < ntheta; ++j2) {
				float u = (float)j2 / (float)nphi;
				float phi = u * M_PI * 2.0;
				djb::vec3 i = djb::vec3(theta, phi);
				nint+= fr.evalp(i, o) * sin(theta);
			}
		}
		nint*= dtheta_dphi;
		success = (nint.x <= 1.0f && nint.y <= 1.0f && nint.z <= 1.0f);
	}
	return success;
}

int main(int argc, char **argv)
{
	bool success = true;
	if (argc == 1) {
		usage(argv[0]);
		return EXIT_SUCCESS;
	}

	for (int i = 1; i < argc && success; ++i) {
		printf("Testing %s...\n", argv[i]);
		success = furnace(djb::utia(argv[i]));
		printf("%s\n", success ? "=> ok" : "=> FAILURE");
	}

	if (success) return EXIT_SUCCESS;
	return EXIT_FAILURE;
}

