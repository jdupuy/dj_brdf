// this code exports the roughness parameters for Beckmann and GGX 
// microfacet distributions fitted from MERL material

// compile
// g++ -O3 -I../ merl_params.cpp -o merl_params -lm

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
		djb::tabular tab(merl, 90);
		djb::microfacet::params params_beckmann = 
			djb::tabular::fit_beckmann_parameters(tab);
		djb::microfacet::params params_ggx = 
			djb::tabular::fit_ggx_parameters(tab);
		char name[64];
		float beckmann, ggx, dummy;

		sscanf(strrchr(input, '/') + 1, "%[^.]s", name);
		params_beckmann.get_ellipse(&beckmann, &dummy, NULL);
		params_ggx.get_ellipse(&ggx, &dummy, NULL);
		fprintf(pf, "%s %.3f %.3f\n", name, beckmann, ggx);
	}

	fclose(pf);

	return EXIT_SUCCESS;
}
