// g++ -O3 plot_qf.cpp -o plot_qf

#include <stdio.h>
#include <stdlib.h>

#define DJ_BRDF_IMPLEMENTATION 1
#include "../dj_brdf.h" 

void plot_qf(const djb::radial& fr, FILE *pf)
{
	int cnt = 90;

	for (int i = 1; i < cnt; ++i) {
		float u = (float)i / (float)cnt;
		float qf = fr.qf_radial(u);

		fprintf(pf, "%f %f\n", u, qf);
	}
}


int main(int argc, char **argv)
{
	djb::beckmann beckmann(djb::fresnel::ideal(), false);
	FILE *pf = fopen("eval_qf_beckmann.txt", "w");
	plot_qf(beckmann, pf);
	fclose(pf);

	djb::tabular beckmann_tab(beckmann, 180);
	pf = fopen("eval_qf_beckmann_tab.txt", "w");
	plot_qf(beckmann_tab, pf);
	fclose(pf);

	djb::ggx ggx(djb::fresnel::ideal(), false);
	pf = fopen("eval_qf_ggx.txt", "w");
	plot_qf(ggx, pf);
	fclose(pf);

	djb::tabular ggx_tab(ggx, 180);
	pf = fopen("eval_qf_ggx_tab.txt", "w");
	plot_qf(ggx_tab, pf);
	fclose(pf);

	return 0;
}
