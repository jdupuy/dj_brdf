// this code exports the roughness parameters for Beckmann and GGX 
// microfacet distributions fitted from MERL material

// compile
// g++ -Wall -O3 -I../ merl_params.cpp -o merl_params -lm

// run
// ./merl_params path_to_merl.binary

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define DJ_BRDF_IMPLEMENTATION 1
#include "dj_brdf.h"

static void usage(const char *appname)
{
	printf("%s - GGX and Beckmann Parameters for MERL BRDFs\n\n",
	       appname);
	printf("Usage\n"
	       "  %s merl1.binary merl2.binary ...\n\n", appname);
	printf("Options\n"
	       "  -h\n"
	       "     Print help\n\n"
	       );
}

int main(int argc, char **argv)
{
	FILE *pf = NULL;
	int i;

	// check params
	if (argc < 2) {
		usage(argv[0]);
		return EXIT_SUCCESS;
	}

	// parse command line
	for (i = 1; i < argc; ++i) {
		if (!strcmp("-h", argv[i])) {
			usage(argv[0]);
			return EXIT_SUCCESS;
		}
	}

	// create file
	pf = fopen("params.txt", "w");
	assert(pf && "fopen failed");
	fprintf(pf, "# MERL Beckmann GGX\n");
	for (i = 1; i < argc; ++i) {
		const char *input = argv[i];
		djb::merl merl(input);
		djb::tabular tab(djb::microfacet::GAF_SMITH, merl, 90, true);
		djb::ggx *ggx = djb::tabular::to_ggx(tab);
		djb::gaussian *gaussian = djb::tabular::to_gaussian(tab);
		char name[64];
		double alpha_x, alpha_g;
		const double sqrt_2 = sqrt(2.f);

		sscanf(strrchr(input, '/') + 1, "%[^.]s", name);
		ggx->get_roughness(&alpha_x, NULL, NULL);
		gaussian->get_roughness(&alpha_g, NULL, NULL);
		fprintf(pf, "%s %.3f %.3f\n", name, alpha_g * sqrt_2, alpha_x);

		delete ggx;
		delete gaussian;
	}

	fclose(pf);

	return EXIT_SUCCESS;
}
