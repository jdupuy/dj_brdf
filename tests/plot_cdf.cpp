// g++ -O3 plot_cdf.cpp -o plot_cdf

#include <stdio.h>
#include <stdlib.h>

#define DJ_BRDF_IMPLEMENTATION 1
#include "../dj_brdf.h" 

void plot_cdf(const djb::radial& fr, FILE *pf)
{
	int cnt = 90;

	for (int i = 1; i < cnt; ++i) {
		float u = (float)i / (float)cnt;
		float theta = u * M_PI / 2.0;
		float cdf = fr.cdf_radial(tan(theta));

		fprintf(pf, "%f %f\n", theta * 180.0 / M_PI, cdf);
	}
}


int main(int argc, char **argv)
{
	djb::beckmann beckmann(djb::fresnel::ideal(), false);
	FILE *pf = fopen("eval_cdf_beckmann.txt", "w");
	plot_cdf(beckmann, pf);
	fclose(pf);

	djb::tabular beckmann_tab(beckmann, 180);
	pf = fopen("eval_cdf_beckmann_tab.txt", "w");
	plot_cdf(beckmann_tab, pf);
	fclose(pf);

	djb::ggx ggx(djb::fresnel::ideal(), false);
	pf = fopen("eval_cdf_ggx.txt", "w");
	plot_cdf(ggx, pf);
	fclose(pf);

	djb::tabular ggx_tab(ggx, 180);
	pf = fopen("eval_cdf_ggx_tab.txt", "w");
	plot_cdf(ggx_tab, pf);
	fclose(pf);

	return 0;
}
